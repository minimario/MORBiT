import higher
import numpy as np
import hypergrad as hg
from generate_task import Sine

# amplitude: [0.1, 5]
# phase: [0, pi]
# A * sin(x-phase)

# data uniformly in [-5, 5]
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

import torch
import torch.nn as nn


train_tasks = 5
test_tasks = 10
sine_task = Sine(train_tasks, test_tasks)




def simplex_projection(s):
    """Projection onto the unit simplex."""
    # if np.sum(s) <=1 and np.alltrue(s >= 0):
        # return s
    # Code taken from https://gist.github.com/daien/1272551
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(s)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - 1) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    return np.maximum(s-theta, 0)

class Learner():
    def __init__(self, sine, gamma=0.003, update_lambdas=False):
        self.gamma = gamma
        self.update_lambdas = update_lambdas
        self.task = sine
        self.embedding_dim = 10
        self.features = nn.Sequential(
            nn.Linear(1, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            # nn.Linear(40, 40),
            # nn.ReLU(),
            nn.Linear(40, self.embedding_dim) # (40, 10)
        )
        self.heads = [nn.Linear(self.embedding_dim, 1) for _ in range(sine.total_tasks)]

        self.fmodel = higher.monkeypatch(self.features, copy_initial_weights=True)
        self.fheads = [higher.monkeypatch(self.heads[i], copy_initial_weights=True) for i in range(len(self.heads))]
        # lambda1 = lambda epoch: 0.01 / 2 ** (epoch // 100)
        
        # self.feature_optimizer = torch.optim.lr_scheduler.LambdaLR(torch.optim.SGD(list(self.features.parameters())), lambda1)
        self.feature_optimizer = torch.optim.Adam(list(self.features.parameters()))
        self.head_optimizers = [torch.optim.Adam(list(self.heads[i].parameters())) for i in range(sine.total_tasks)]

        self.lambdas = 1/sine.train_tasks * np.ones(sine.train_tasks)


    def outer_loss(self, task_id):
        def loss(params, hparams):
            xs, ys = self.task.sample_batch(task_id, 10)
            embedding = self.fmodel(xs, params=hparams)
            y_preds = self.fheads[task_id](embedding, params=params)
            loss_fn = nn.MSELoss()
            total_loss = loss_fn(y_preds, ys)
            overall_loss = self.lambdas[task_id] * total_loss
            return overall_loss
        return loss

    def inner_loss(self, task_id):
        def loss(params, hparams):
            xs, ys = self.task.sample_batch(task_id, 10)
            embedding = self.fmodel(xs, params=hparams)
            y_preds = self.fheads[task_id](embedding, params=params)
            loss_fn = nn.MSELoss()
            total_loss = loss_fn(y_preds, ys)
            return total_loss
        return hg.GradientDescent(loss, step_size=0.01)
                
    def test_full_batch(self, mode='train'):
        if mode == 'train':
            task_list = np.arange(self.task.train_tasks)
        else:
            task_list = np.arange(self.task.test_tasks) + self.task.train_tasks
        losses = []
        for task in task_list:
            xs, ys = self.task.sample_batch(task, 100000)
            embedding = self.features(xs)
            y_preds = self.heads[task](embedding)
            loss_fn = nn.MSELoss()
            total_loss = loss_fn(y_preds, ys)
            losses.append(total_loss.item())
        return losses        

    def inner_loop(self, n_inner=2, tasks_per_batch=5, n_shots=10, mode="train", verbose=True):
        # sample list of task id's
        if mode == "train": 
            # task_list = np.random.choice(self.task.train_tasks, tasks_per_batch, replace=False)
            task_list = np.arange(self.task.train_tasks)
        elif mode == "test":
            # task_list = np.random.choice(self.task.test_tasks, tasks_per_batch, replace=False) + self.task.train_tasks
            task_list = np.arange(self.task.test_tasks) + self.task.train_tasks

        for task_id in task_list: 
            self.head_optimizers[task_id].zero_grad()
            
            # get inner loop data
            xs, ys = self.task.sample_batch(task_id, n_shots)

            # calculate value of outer loop
            embedding = self.features(xs)

            # optimize weights for inner loop
            for it_inner in range(n_inner):
                # regularization
                l2_reg_constant = 0.01
                l2_reg = 0
                for p in self.heads[task_id].parameters():
                    l2_reg += p.norm(2)

                # network loss
                y_preds = self.heads[task_id](embedding)
                loss_fn = nn.MSELoss()
                total_loss = loss_fn(y_preds, ys) + l2_reg_constant * l2_reg

                hg.fixed_point(params=list(self.heads[task_id].parameters()), 
                    hparams=list(self.features.parameters()),
                    K=100, 
                    fp_map=self.inner_loss(task_id),
                    outer_loss=self.outer_loss(task_id))

                # backpropagate
                total_loss.backward(retain_graph=True)
                self.head_optimizers[task_id].step()
        
        # validation (don't get to tune inner loop)
        self.feature_optimizer.zero_grad()
        for a in task_list:
            self.head_optimizers[a].zero_grad()
        losses = []
        for a in task_list:
            xs, ys = self.task.sample_batch(a, n_shots)
            embedding = self.features(xs)
            y_preds = self.heads[a](embedding)
            # print("pred: ", y_preds.detach().numpy().flatten())
            # print("act: ", ys.numpy().flatten())
            loss_fn = nn.MSELoss()
            total_loss = loss_fn(y_preds, ys)
            if verbose:
                print(f"task {a}, loss: {total_loss}")
            losses.append(total_loss)
        return losses, task_list

    def learn(self):
        all_losses = []
        all_max_losses = []
        for it_outer in range(4000):
            # perform inner loop
            task_losses, tasks = self.inner_loop(tasks_per_batch=10, verbose=False)
            # print(f"losses!: {np.array([i.item() for i in task_losses])}")

            # calculate overall loss (using lambda weights)
            overall_loss = 0
            for i, t in enumerate(tasks):
                overall_loss += self.lambdas[t] * task_losses[i]

            # backpropagation and lambda
            overall_loss.backward()
            self.feature_optimizer.step()

            # (avg with, max with, avg w/o, max w/o)
            # (4.037, 12.028, 4.025, 12.815)

            if self.update_lambdas:
                # print(self.lambdas)
                for i, t in enumerate(tasks):
                    mu_lambda = 3
                    # print("pull of task losses: ", task_losses[i].item()*self.gamma / ((it_outer + 1) ** (3/5)))
                    # print("pull of reg: ", -mu_lambda*(self.lambdas[t] - 1/self.task.train_tasks) * self.gamma / ((it_outer + 1) ** (3/5)))
                    self.lambdas[t] += (task_losses[i].item() - mu_lambda * (self.lambdas[t] - 1/self.task.train_tasks)) * self.gamma

                    # self.lambdas[t] += (task_losses[i].item() - mu_lambda * (self.lambdas[t] - 1/self.task.train_tasks)) * self.gamma / ((it_outer + 1) ** (3/5))
                    # self.lambdas[t] += task_losses[i].item() * self.gamma / ((it_outer + 1) ** 0.2)

                self.lambdas = simplex_projection(self.lambdas)

                # lambda_i * t_i + 1/2*|lambda - uniform|^2 
                # grad_i = t_i + 2 * (lambda_i - 1/)

                # 1) lambda regularization [done]
                # 2) use full loss rather than stochastic loss [done]
                # 3) tune learning rate schedules
                # 4) check testing
                # 5) SGD instead of adam

            # log average loss for graphing purposes
            true_losses = self.test_full_batch(mode='train')
            all_losses.append(sum(true_losses) / len(true_losses))
            all_max_losses.append(max(true_losses))

            if it_outer % 100 == 0:
                print(f"iteration {it_outer}")
                print(f"losses!: {np.array([i.item() for i in task_losses])}")
                print(f"true losses: {true_losses}")
                # print(f"tasks: {tasks)
                print(f"lambdas: {self.lambdas[:50]}")
        
        return all_losses, all_max_losses

    def evaluate(self, task_size = 100, mode="test"):
        loss = 0
        max_loss = 0
        all_losses = []
        all_max_losses = []
        for i in range(5):
            self.inner_loop(mode="test", n_inner=10, verbose=False)
            
        true_losses = self.test_full_batch(mode='test')
        print("average loss: ", sum(true_losses) / len(true_losses))
        print("max loss: ", max(true_losses))

        return sum(true_losses) / len(true_losses), max(true_losses)
        # return all_losses, all_max_losses
        # for i in range(task_size):
            # batch_loss, tasks = self.inner_loop(mode=mode, tasks_per_batch=10, n_inner=10, verbose=False)
            # loss += sum([i.item() for i in batch_loss])/len(batch_loss)
            # max_loss += max([i.item() for i in batch_loss])

        # print(f"test loss {loss / task_size}")
        # print(f"max test loss {max_loss / task_size}")


print(sine_task.a)
learner_lambdas = Learner(sine_task, update_lambdas=True)
loss_lambdas, max_loss_lambdas = learner_lambdas.learn()

learner_nolambdas = Learner(sine_task, update_lambdas=False)
loss_nolambdas, max_loss_nolambdas = learner_nolambdas.learn()

# from matplotlib import pyplot as plt
# plt.plot(range(len(loss_lambdas)), loss_lambdas, label='lambdas')
# plt.plot(range(len(loss_nolambdas)), loss_nolambdas, label='no lambdas')
# plt.legend()
# plt.title("avg loss")
# plt.show()

from matplotlib import pyplot as plt
plt.plot(range(len(max_loss_lambdas)), max_loss_lambdas, label='lambdas')
plt.plot(range(len(max_loss_nolambdas)), max_loss_nolambdas, label='no lambdas')
plt.legend()
plt.title("max loss")
plt.show()

print(learner_lambdas.evaluate())
print(learner_nolambdas.evaluate())


# print(learner_lambdas.evaluate(mode="train"))
# print(learner_nolambdas.evaluate(mode="test"))
# print(learner_nolambdas.evaluate(mode="train"))
