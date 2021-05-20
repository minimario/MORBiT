import numpy as np

# amplitude: [0.1, 5]
# phase: [0, pi]
# A * sin(x-phase)

# data uniformly in [-5, 5]
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

import torch
import torch.nn as nn

train_tasks = 10
test_tasks = 10

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

class Sine():
    def __init__(self, train_tasks, test_tasks):
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.total_tasks = train_tasks + test_tasks
        self.a = np.random.uniform(0.1, 5, (self.total_tasks,))
        self.phi = np.random.uniform(0, np.pi, (self.total_tasks,))

    def sample_batch(self, task_id, batch_size):
        x = np.random.uniform(-5, 5, (batch_size,))
        y = self.a[task_id] * np.sin(x - self.phi[task_id])
        return torch.tensor(x).unsqueeze(1).type(torch.float32), torch.tensor(y).unsqueeze(1).type(torch.float32)

class Learner():
    def __init__(self, sine, gamma=0.1, update_lambdas=False):
        self.gamma = gamma
        self.update_lambdas = update_lambdas
        self.task = sine
        self.embedding_dim = 10
        self.features = nn.Sequential(
            nn.Linear(1, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, self.embedding_dim)
        )
        self.heads = [nn.Linear(self.embedding_dim, 1) for _ in range(sine.total_tasks)]

        self.feature_optimizer = torch.optim.Adam(list(self.features.parameters()))
        self.head_optimizers = [torch.optim.Adam(list(self.heads[i].parameters())) for i in range(sine.total_tasks)]

        self.lambdas = 1/sine.train_tasks * np.ones(sine.train_tasks)

    def inner_loop(self, n_inner=10, tasks_per_batch=5, n_shots=100, mode="train", verbose=True):
        # sample list of task id's
        if mode == "train": 
            # task_list = np.random.choice(self.task.train_tasks, tasks_per_batch, replace=False)
            task_list = np.arange(self.task.train_tasks)
        elif mode == "test":
            task_list = np.random.choice(self.task.test_tasks, tasks_per_batch, replace=False) + self.task.train_tasks
        
        for task in task_list: 
            self.head_optimizers[task].zero_grad()
            
            # get inner loop data
            xs, ys = self.task.sample_batch(task, n_shots)

            # calculate value of outer loop
            embedding = self.features(xs)

            # optimize weights for inner loop
            for it_inner in range(n_inner):
                # regularization
                l2_reg_constant = 0.01
                l2_reg = 0
                for p in self.heads[task].parameters():
                    l2_reg += p.norm(2)

                # network loss
                y_preds = self.heads[task](embedding)
                loss_fn = nn.MSELoss()
                total_loss = loss_fn(y_preds, ys) + l2_reg_constant * l2_reg

                # backpropagate
                total_loss.backward(retain_graph=True)
                self.head_optimizers[task].step()
        
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
        for it_outer in range(1000):
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

            if self.update_lambdas:
                for i, t in enumerate(tasks):
                    self.lambdas[t] += task_losses[i] * self.gamma
                self.lambdas = simplex_projection(self.lambdas)

            # log average loss for graphing purposes
            all_losses.append(overall_loss.item())
            all_max_losses.append(max(list(map(lambda x:x.item(), task_losses))))

            if it_outer % 100 == 0:
                print(f"iteration {it_outer}")
                print(f"losses!: {np.array([i.item() for i in task_losses])}")
                # print(f"tasks: {tasks)
                print(f"lambdas: {self.lambdas[:50]}")
        
        return all_losses, all_max_losses

    def evaluate(self, task_size = 100, mode="test"):
        loss = 0
        max_loss = 0
        for i in range(task_size):
            batch_loss, tasks = self.inner_loop(mode=mode, tasks_per_batch=10, n_inner=20, verbose=False)
            loss += sum([i.item() for i in batch_loss])/len(batch_loss)
            max_loss += max([i.item() for i in batch_loss])

        print(f"test loss {loss / task_size}")
        print(f"max test loss {max_loss / task_size}")

sine_task = Sine(train_tasks, test_tasks)
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

print(learner_lambdas.evaluate(mode="test"))
print(learner_lambdas.evaluate(mode="train"))
print(learner_nolambdas.evaluate(mode="test"))
print(learner_nolambdas.evaluate(mode="train"))
