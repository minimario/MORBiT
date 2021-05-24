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
    u = np.sort(s)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(s-theta, 0)

class Learner():
    def __init__(self, sine, alpha=0.001, beta=0.05, gamma=0.003, update_lambdas=False):
        self.gamma = gamma
        self.update_lambdas = update_lambdas
        self.task = sine
        self.embedding_dim = 10
        self.features = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            # nn.Linear(40, 40),
            # nn.ReLU(),
            nn.Linear(100, self.embedding_dim) # (40, 10)
        )
        self.heads = [nn.Linear(self.embedding_dim, 1) for _ in range(sine.total_tasks)]

        self.fmodel = higher.monkeypatch(self.features, copy_initial_weights=True)
        self.fheads = [higher.monkeypatch(self.heads[i], copy_initial_weights=True) for i in range(len(self.heads))]
        # lambda1 = lambda epoch: 0.01 / 2 ** (epoch // 100)
        
        # self.feature_optimizer = torch.optim.lr_scheduler.LambdaLR(torch.optim.SGD(list(self.features.parameters())), lambda1)
        self.feature_optimizer = torch.optim.Adam(list(self.features.parameters()), alpha)
        self.head_optimizers = [torch.optim.Adam(list(self.heads[i].parameters()), beta) for i in range(sine.total_tasks)]
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

            # regularization
            l2_reg_constant = 0.01
            l2_reg = 0
            for p in self.fheads[task_id].parameters():
                l2_reg += p.norm(2)
            return total_loss + l2_reg_constant * l2_reg

        return hg.GradientDescent(loss, step_size=0.01)
                
    def test_full_batch(self, mode='train'):
        if mode == 'train': task_list = np.arange(self.task.train_tasks)
        else: task_list = np.arange(self.task.test_tasks) + self.task.train_tasks
        losses = []
        for task_id in task_list:
            xs, ys = self.task.sample_full_batch(task_id, 100)
            losses.append(self.get_loss(task_id, xs, ys, reg=False).item())
        return losses        

    def get_loss(self, task, xs, ys, reg=False, embedding=None):
        if embedding == None:
            embedding = self.features(xs)

        y_preds = self.heads[task](embedding)
        loss_fn = nn.MSELoss()
        total_loss = loss_fn(y_preds, ys)

        if reg:
            l2_reg_constant = 0.01
            l2_reg = 0
            for p in self.heads[task].parameters():
                l2_reg += p.norm(2)
            total_loss += l2_reg * l2_reg_constant

        return total_loss

    def validation(self, task_list, n_shots):
        # clear all the gradients
        self.feature_optimizer.zero_grad()
        for a in task_list: self.head_optimizers[a].zero_grad()

        # calculate loss
        losses = []
        for task_id in task_list:
            xs, ys = self.task.sample_batch(task_id, n_shots)
            losses.append(self.get_loss(task_id, xs, ys, reg=False))
        return losses

    def inner_loop(self, n_inner=2, tasks_per_batch=5, n_shots=10, mode="train", verbose=True):
        # sample list of task id's
        if mode == "train": task_list = np.arange(self.task.train_tasks)
        elif mode == "test": task_list = np.arange(self.task.test_tasks) + self.task.train_tasks

        for task_id in task_list:             
            xs, ys = self.task.sample_batch(task_id, n_shots) # sample data
            embedding = self.features(xs) # outer loop
            for it_inner in range(n_inner): # inner loop
                self.head_optimizers[task_id].zero_grad() # clear gradients
                total_loss = self.get_loss(task_id, xs, ys, reg=True, embedding=embedding) # fwd
                total_loss.backward(retain_graph=True) # backwards
                self.head_optimizers[task_id].step() # update inner weights
        
        return task_list

    def learn(self):
        all_avg_losses = []
        all_max_losses = []
        for it_outer in range(4000):
            self.feature_optimizer.zero_grad()
            # perform inner loop

            # print("bef")
            # print("loss before: ")
            # print(self.inner_loss(0)(list(self.heads[0].parameters()), list(self.features.parameters())))
            # print(list(self.heads[0].parameters())[0].detach().numpy().squeeze())            
            task_list = self.inner_loop(n_inner=2, n_shots=10, verbose=False)
            task_losses = self.validation(task_list, n_shots=10)
            # print(self.inner_loss(0)(list(self.heads[0].parameters()), list(self.features.parameters())))
            # print("loss after")
            # print(list(self.heads[0].parameters())[0].detach().numpy().squeeze())
            # print("af")

            # print(f"losses!: {np.array([i.item() for i in task_losses])}")

            # calculate overall loss (using lambda weights)
            overall_loss = 0
            for i, t in enumerate(task_list):
                overall_loss += self.lambdas[t] * task_losses[i]

            for task_id in task_list:
                hg.fixed_point(params=list(self.heads[task_id].parameters()), 
                    hparams=list(self.features.parameters()),
                    K=10, 
                    fp_map=self.inner_loss(task_id),
                    outer_loss=self.outer_loss(task_id))
                    
            # task_losses = []
            # for task_id in tasks:
                # task_losses.append(self.outer_loss(task_id))

            # backpropagation and lambda
            # overall_loss.backward()
            self.feature_optimizer.step()

            if self.update_lambdas:
                # print(self.lambdas)
                for i, t in enumerate(task_list):
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
            all_avg_losses.append(sum(true_losses) / len(true_losses))
            all_max_losses.append(max(true_losses))

            if it_outer % 100 == 0:
                print(f"iteration {it_outer}")
                print(f"losses!: {np.array([i.item() for i in task_losses])}")
                print(f"true losses: {true_losses}")
                # print(f"tasks: {tasks)
                print(f"lambdas: {self.lambdas[:50]}")
        
        return all_avg_losses, all_max_losses

    def evaluate(self):
        for i in range(5):
            self.inner_loop(mode="test", n_inner=10, verbose=False)
            
        true_losses = self.test_full_batch(mode='test')
        print("average loss: ", sum(true_losses) / len(true_losses))
        print("max loss: ", max(true_losses))

        return sum(true_losses) / len(true_losses), max(true_losses)
        # return all_avg_losses, all_max_losses
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

data = {}
data["hg_lambdas"] = max_loss_lambdas
data["hg_nolambdas"] = max_loss_nolambdas
import pickle
with open("hg.pkl", "wb") as f:
    pickle.dump(data, f)

from matplotlib import pyplot as plt
plt.plot(range(len(max_loss_lambdas)), max_loss_lambdas, label='lambdas')
plt.plot(range(len(max_loss_nolambdas)), max_loss_nolambdas, label='no lambdas')
plt.legend()
plt.title("max loss")
plt.show()
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
