import pickle
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
import torch
import torch.nn as nn

from generate_task import Sine

def simplex_projection(s):
    """Projection onto the unit simplex."""
    u = np.sort(s)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(s-theta, 0)

class Learner():
    def __init__(self, sine, alpha=0.003, beta=0.005, gamma=0.003, decay=0.9, update_lambdas=False, use_schedule=True):
        self.task = sine # task

        # training details
        self.lambdas = 1/sine.train_tasks * np.ones(sine.train_tasks) # initialize lambdas as uniform
        self.gamma = gamma # step size for lambda update
        self.update_lambdas = update_lambdas # whether ro update lambdas

        # network details
        self.embedding_dim = 10 # output dimension of embedding network
        self.neurons_per_hidden = 80 # number of hidden neurons
        self.features = nn.Sequential( # embedding network
            nn.Linear(1, self.neurons_per_hidden),
            nn.ReLU(),
            nn.Linear(self.neurons_per_hidden, self.neurons_per_hidden),
            nn.ReLU(),
            nn.Linear(self.neurons_per_hidden, self.embedding_dim)
        )
        self.heads = [nn.Linear(self.embedding_dim, 1) for _ in range(sine.total_tasks)] # task-specific network

        # optimizers and schedulers
        self.use_schedule = use_schedule # whether to use a scheduler
        self.feature_optimizer = torch.optim.SGD(list(self.features.parameters()), alpha)
        self.feature_scheduler = torch.optim.lr_scheduler.StepLR(self.feature_optimizer, step_size=100, gamma=decay)        
        self.head_optimizers = [torch.optim.Adam(list(self.heads[i].parameters()), beta) for i in range(sine.total_tasks)]
                
    def test_full_batch(self, mode='train'):
        """
        Function to approximate the true loss of the current network
        """
        if mode == 'train': task_list = np.arange(self.task.train_tasks)
        else: task_list = np.arange(self.task.test_tasks) + self.task.train_tasks
        losses = []
        for task_id in task_list:
            xs, ys = self.task.sample_full_batch(task_id, 100)
            losses.append(self.get_loss(task_id, xs, ys, reg=False).item())
        return losses        

    def get_loss(self, task, xs, ys, reg=False, embedding=None):
        """
        Function to get the loss of the network from (x, y) pairs
        """
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
        """
        meta-validation loop
        """

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
        """
        inner loop for few-shot learning
        samples n_shots samples from each task, then runs n_inner steps
        """

        # get the list of tasks (all tasks)
        if mode == "train": task_list = np.arange(self.task.train_tasks)
        elif mode == "test": task_list = np.arange(self.task.test_tasks) + self.task.train_tasks

        for task_id in task_list:             
            xs, ys = self.task.sample_batch(task_id, n_shots) # sample data
            embedding = self.features(xs) # outer loop
            for _ in range(n_inner): # inner loop
                self.head_optimizers[task_id].zero_grad() # clear gradients
                # print(list(self.heads[task_id].parameters())[0].detach().numpy().squeeze())
                total_loss = self.get_loss(task_id, xs, ys, reg=True, embedding=embedding) # fwd
                total_loss.backward(retain_graph=True) # backwards
                self.head_optimizers[task_id].step() # update inner weights
                # print(list(self.heads[task_id].parameters())[0].detach().numpy().squeeze())
                # print("-")        
        return task_list

    def learn(self, n_outer=2000, n_inner=2, n_shots=10):
        """
        main loop
        """
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
            if self.use_schedule:
                self.feature_scheduler.step()
                if it_outer % 100 == 0: 
                    print(self.feature_scheduler.get_last_lr())

            if self.update_lambdas:
                for i, t in enumerate(task_list):
                    mu_lambda = 3
                    # print("pull of task losses: ", task_losses[i].item()*self.gamma / ((it_outer + 1) ** (3/5)))
                    # print("pull of reg: ", -mu_lambda*(self.lambdas[t] - 1/self.task.train_tasks) * self.gamma / ((it_outer + 1) ** (3/5)))
                    self.lambdas[t] += (task_losses[i].item() - mu_lambda * (self.lambdas[t] - 1/self.task.train_tasks)) * self.gamma
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
            
            if it_outer % 100 == 0:
                print(f"iteration {it_outer}")
                print(f"     losses: {np.array([i.item() for i in task_losses])}")
                print(f"true losses: {np.array(true_losses)}")
                print(f"    lambdas: {self.lambdas[:50]}")
        
        return all_avg_losses, all_max_losses

    def evaluate(self):
        """
        testing loss (this function is not used)
        """
        for i in range(5):
            self.inner_loop(mode="test", n_inner=2, verbose=False)
            
        true_losses = self.test_full_batch(mode='test')
        print("average loss: ", sum(true_losses) / len(true_losses))
        print("max loss: ", max(true_losses))

        return sum(true_losses) / len(true_losses), max(true_losses)

train_tasks = 20
test_tasks = 10
sine_task = Sine(train_tasks, test_tasks)
def run_with_step_size(lambdas=True, use_schedule=True, 
                       alpha=0.003, beta=0.003, gamma=0.003, decay=0.9,
                       n_outer=4000, n_inner=2, n_shots=10, run_id=-1):
    data = {}
    learner_lambdas = Learner(sine_task, alpha=alpha, beta=beta, gamma=gamma, decay=decay, update_lambdas=lambdas, use_schedule=use_schedule)
    loss_lambdas, max_loss_lambdas = learner_lambdas.learn(n_outer, n_inner, n_shots)
    data["lambdas"] = lambdas
    data["avg"] = loss_lambdas
    data["max"] = max_loss_lambdas

    with open(f"runs/{lambdas}_{run_id}.pkl", "wb") as f:
        pickle.dump(data, f)

n_outer = 10
for i in range(20):
    try:
        run_with_step_size(True, False, 0.007, 0.005, 0.003, 1, n_outer, 2, 10, 2*i) # with lambdas
        run_with_step_size(False, False, 0.007, 0.011, 0.003, 1, n_outer, 2, 10, 2*i+1) # without lambdas
    except:
        pass