import numpy as np
import torch

np.random.seed(123)

class Sine():
    def __init__(self, train_tasks, test_tasks):
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.total_tasks = train_tasks + test_tasks
        self.a = np.random.uniform(0.1, 5, (self.total_tasks,))
        self.freq = np.random.uniform(1, 3, (self.total_tasks, ))
        self.phi = np.random.uniform(0, np.pi, (self.total_tasks,))

    def sample_batch(self, task_id, batch_size):
        x = np.random.uniform(-5, 5, (batch_size,))
        y = self.a[task_id] * np.sin(self.freq[task_id] * x - self.phi[task_id])
        return torch.tensor(x).unsqueeze(1).type(torch.float32), torch.tensor(y).unsqueeze(1).type(torch.float32)

    def sample_full_batch(self, task_id, batch_size):
        x = np.linspace(-5, 5, batch_size)
        y = self.a[task_id] * np.sin(self.freq[task_id] * x - self.phi[task_id])
        return torch.tensor(x).unsqueeze(1).type(torch.float32), torch.tensor(y).unsqueeze(1).type(torch.float32)
