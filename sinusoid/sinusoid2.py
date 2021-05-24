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

train_tasks = 20
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
    def __init__(self, sine, alpha=0.003, beta=0.005, gamma=0.003, update_lambdas=False):
        self.gamma = gamma
        self.update_lambdas = update_lambdas
        self.task = sine
        self.embedding_dim = 10
        self.neurons_per_hidden = 80
        self.features = nn.Sequential(
            nn.Linear(1, self.neurons_per_hidden),
            nn.ReLU(),
            nn.Linear(self.neurons_per_hidden, self.neurons_per_hidden),
            nn.ReLU(),
            # nn.Linear(40, 40),
            # nn.ReLU(),
            nn.Linear(self.neurons_per_hidden, self.embedding_dim) # (40, 10)
        )
        self.heads = [nn.Linear(self.embedding_dim, 1) for _ in range(sine.total_tasks)]
        
        self.feature_optimizer = torch.optim.SGD(list(self.features.parameters()), alpha)
        # lambda1 = lambda epoch: 1/(epoch+1)**0.1
        # self.feature_scheduler = torch.optim.lr_scheduler.StepLR(self.feature_optimizer, step_size=100, gamma=0.99)
        # self.feature_scheduler = torch.optim.lr_scheduler.LambdaLR(self.feature_optimizer, lr_lambda=[lambda1])
        
        self.head_optimizers = [torch.optim.Adam(list(self.heads[i].parameters()), beta) for i in range(sine.total_tasks)]
        self.lambdas = 1/sine.train_tasks * np.ones(sine.train_tasks)
                
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

    def inner_loop(self, n_inner=2, n_shots=10, mode="train", verbose=True):
        # sample list of task id's
        if mode == "train": task_list = np.arange(self.task.train_tasks)
        elif mode == "test": task_list = np.arange(self.task.test_tasks) + self.task.train_tasks

        for task_id in task_list:             
            xs, ys = self.task.sample_batch(task_id, n_shots) # sample data
            embedding = self.features(xs) # outer loop
            for it_inner in range(n_inner): # inner loop
                self.head_optimizers[task_id].zero_grad() # clear gradients
                # print(list(self.heads[task_id].parameters())[0].detach().numpy().squeeze())
                total_loss = self.get_loss(task_id, xs, ys, reg=True, embedding=embedding) # fwd
                total_loss.backward(retain_graph=True) # backwards
                self.head_optimizers[task_id].step() # update inner weights
                # print(list(self.heads[task_id].parameters())[0].detach().numpy().squeeze())
                # print("-")        
        return task_list

    def learn(self, n_outer=2000, n_inner=2, n_shots=10):
        all_avg_losses = []
        all_max_losses = []
        for it_outer in range(n_outer):
            self.feature_optimizer.zero_grad()

            task_list = self.inner_loop(n_inner, n_shots, verbose=False)
            task_losses = self.validation(task_list, n_shots)

            # calculate overall loss (using lambda weights)
            overall_loss = 0
            for i, t in enumerate(task_list):
                overall_loss += self.lambdas[t] * task_losses[i]
            overall_loss.backward()
            self.feature_optimizer.step()

            if self.update_lambdas:
                for i, t in enumerate(task_list):
                    mu_lambda = 3
                    # print("pull of task losses: ", task_losses[i].item()*self.gamma / ((it_outer + 1) ** (3/5)))
                    # print("pull of reg: ", -mu_lambda*(self.lambdas[t] - 1/self.task.train_tasks) * self.gamma / ((it_outer + 1) ** (3/5)))
                    self.lambdas[t] += (task_losses[i].item() - mu_lambda * (self.lambdas[t] - 1/self.task.train_tasks)) * self.gamma
                    # self.lambdas[t] += (task_losses[i].item() - mu_lambda * (self.lambdas[t] - 1/self.task.train_tasks)) * self.gamma / ((it_outer + 1) ** (3/5))
                    # self.lambdas[t] += task_losses[i].item() * self.gamma / ((it_outer + 1) ** 0.2)

                self.lambdas = simplex_projection(self.lambdas)

            # log average loss for graphing purposes
            true_losses = self.test_full_batch(mode='train')
            all_avg_losses.append(sum(true_losses) / len(true_losses))
            all_max_losses.append(max(true_losses))

            # if it_outer % 1000 == 0 and it_outer != 0:
                # from matplotlib import pyplot as plt
                # plt.plot(range(len(all_max_losses)), all_max_losses, label='lambdas')
                # plt.legend()
                # plt.title("max loss")
                # plt.show()
            
            # if it_outer % 100 == 0:
            #     print(f"iteration {it_outer}")
            #     print(f"losses!: {np.array([i.item() for i in task_losses])}")
            #     print(f"true losses: {np.array(true_losses)}")
            #     # print(f"tasks: {tasks)
            #     print(f"lambdas: {self.lambdas[:50]}")
        
        return all_avg_losses, all_max_losses

    def evaluate(self):
        for i in range(5):
            self.inner_loop(mode="test", n_inner=2, verbose=False)
            
        true_losses = self.test_full_batch(mode='test')
        print("average loss: ", sum(true_losses) / len(true_losses))
        print("max loss: ", max(true_losses))

        return sum(true_losses) / len(true_losses), max(true_losses)

import pickle
from matplotlib import pyplot as plt

def run_with_step_size(alpha=0.003, beta=0.005, gamma=0.003, n_outer=4000, n_inner=2, n_shots=10, run_id=0):
    learner_lambdas = Learner(sine_task, alpha=alpha, beta=beta, gamma=gamma, update_lambdas=True)
    loss_lambdas, max_loss_lambdas = learner_lambdas.learn(n_outer, n_inner, n_shots)

    learner_no_lambdas = Learner(sine_task, alpha=alpha, beta=beta, gamma=gamma, update_lambdas=False)
    loss_no_lambdas, max_loss_no_lambdas = learner_no_lambdas.learn(n_outer, n_inner, n_shots)

    data = {}
    data["avg_lambdas"] = loss_lambdas
    data["max_lambdas"] = max_loss_lambdas
    data["avg_no_lambdas"] = loss_no_lambdas
    data["max_no_lambdas"] = max_loss_no_lambdas
    with open(f"runs/{alpha}_{beta}_{gamma}_{run_id}.pkl", "wb") as f:
        pickle.dump(data, f)

    plt.clf()
    plt.plot(range(len(max_loss_lambdas)), max_loss_lambdas, label='lambdas')
    plt.plot(range(len(max_loss_lambdas)), max_loss_no_lambdas, label='no lambdas')
    plt.legend()
    plt.title("max loss")
    plt.savefig(f"figs/{alpha}_{beta}_{gamma}_{run_id}.png")

run_id = 0
for alpha in [0.003]:
    for beta in [0.005]:
        for gamma in [0.003]:
            run_with_step_size(alpha, beta, gamma, 10000, 2, 10, run_id)
            run_id += 1
