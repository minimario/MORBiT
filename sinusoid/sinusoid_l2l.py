import numpy as np
import learn2learn as l2l

# amplitude: [0.1, 5]
# phase: [0, pi]
# A * sin(x-phase)

# data uniformly in [-5, 5]
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.set_printoptions(formatter={'double': lambda x: "{0:0.3f}".format(x)})

import torch
import torch.nn as nn

train_tasks = 5
test_tasks = 10

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
    def __init__(self, sine, gamma=0.001):
        self.gamma = gamma
        # self.update_lambdas = update_lambdas

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

        self.meta_lr = 0.01
        self.fast_lr = 0.1
        for i in range(sine.total_tasks):
            self.heads[i] = l2l.algorithms.MAML(self.heads[i], lr=self.fast_lr)

        self.all_parameters = list(self.features.parameters())
        self.optimizer = torch.optim.Adam(self.all_parameters, lr=self.meta_lr)

        # self.lambdas = 1/sine.train_tasks * np.ones(sine.train_tasks)

    def fast_adapt(self, xs_train, ys_train, xs_val, ys_val,
                learner,
                features,
                loss,
                reg_lambda,
                adaptation_steps):
        adaptation_data = features(xs_train)
        adaptation_labels = ys_train

        evaluation_data = features(xs_val)
        evaluation_labels = ys_val

        for step in range(adaptation_steps):
            l2_reg = 0
            for p in learner.parameters():
                l2_reg += p.norm(2)
            # print(np.array([i.item() for i in learner(adaptation_data)]))
            # print(np.array([i.item() for i in adaptation_labels]))
            train_error = loss(learner(adaptation_data), adaptation_labels) + reg_lambda*l2_reg 
            learner.adapt(train_error)

        predictions = learner(evaluation_data)
        valid_error = loss(predictions, evaluation_labels)

        return valid_error

    def learn(self):
        iters = 1000
        all_losses = [] # TODO: update
        all_max_losses = [] # TODO: update

        for iteration in range(iters):
            self.optimizer.zero_grad()

            task_losses = []
            task_list = np.arange(self.task.train_tasks)
            total_loss = 0
            for task_num in task_list:
                xs_train, ys_train = self.task.sample_batch(task_num, batch_size=10)
                xs_val, ys_val = self.task.sample_batch(task_num, batch_size=10)
                
                learner = self.heads[task_num].clone()
                evaluation_loss = self.fast_adapt(xs_train, ys_train, xs_val, ys_val,
                                            learner, 
                                            self.features,
                                            nn.MSELoss(),
                                            0.01, # lambda reg
                                            10) # inner loop steps
                total_loss += evaluation_loss
                task_losses.append(total_loss.item())

            total_loss /= self.task.train_tasks
            total_loss.backward()
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss {total_loss}")
        
        self.optimizer.step()
        
        return all_losses, all_max_losses


sine_task = Sine(train_tasks, test_tasks)
learner_lambdas = Learner(sine_task)
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
