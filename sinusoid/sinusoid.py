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
total_tasks = train_tasks + test_tasks
a = np.random.uniform(0.1, 5, (total_tasks,))
phi = np.random.uniform(0, np.pi, (total_tasks,))

features = nn.Sequential(
    nn.Linear(1, 40),
    nn.ReLU(),
    nn.Linear(40, 40),
    nn.ReLU(),
    nn.Linear(40, 10)
)
heads = [nn.Linear(10, 1) for _ in range(total_tasks)]

feature_optimizer = torch.optim.Adam(list(features.parameters()))
head_optimizers = [torch.optim.Adam(list(heads[i].parameters())) for i in range(total_tasks)]

loss = nn.MSELoss()

def sample_batch(task_id, batch_size):
    x = np.random.uniform(-5, 5, (batch_size,))
    y = a[task_id] * np.sin(x - phi[task_id])
    return torch.tensor(x).unsqueeze(1).type(torch.float32), torch.tensor(y).unsqueeze(1).type(torch.float32)

lambdas = 1/train_tasks * np.ones(train_tasks)
def inner_loop(n_inner=5, batch_size=5, mode="train", verbose=True):
    if mode == "train": 
        # a_list = np.random.choice(train_tasks, batch_size, replace=False)
        a_list = np.arange(10)
    elif mode == "val":
        a_list = np.random.choice(test_tasks, batch_size, replace=False) + train_tasks
    
    for a in a_list: 
        head_optimizers[a].zero_grad()
        
        # get data
        shots = 10
        xs, ys = sample_batch(a, shots)

        # do outer loop
        embedding = features(xs)

        # inner loop optimization
        for it_inner in range(n_inner):
            # regularization
            l2_reg_constant = 0.01
            l2_reg = 0
            for p in heads[a].parameters():
                l2_reg += p.norm(2)

            # network loss
            y_preds = heads[a](embedding)
            total_loss = loss(y_preds, ys) / ys.shape[0] + l2_reg_constant * l2_reg
            # backpropagate
            total_loss.backward(retain_graph=True)
            head_optimizers[a].step()
    
    # validation (don't get to tune inner loop)
    feature_optimizer.zero_grad()
    for a in a_list:
        head_optimizers[a].zero_grad()
    losses = []
    for a in a_list:
        xs, ys = sample_batch(a, shots)
        embedding = features(xs)
        y_preds = heads[a](embedding)
        total_loss = loss(y_preds, ys) / ys.shape[0]
        if verbose:
            print(f"task {a}, loss: {total_loss}")
        losses.append(total_loss)
    return losses, a_list

# losses = []
gamma = 0.1
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

update_lambdas = False
for it_outer in range(1000):
    losses, tasks = inner_loop(batch_size=10, verbose=False)
    overall_loss = 0
    for i, t in enumerate(tasks):
        overall_loss += lambdas[t] * losses[i]
    overall_loss.backward()
    feature_optimizer.step()

    if update_lambdas:
        for i, t in enumerate(tasks):
            lambdas[t] += losses[i] * gamma
        lambdas = simplex_projection(lambdas)

    if it_outer % 100 == 0:
        print(f"iteration {it_outer}")
        # print(f"losses!: {np.array([i.item() for i in losses])}")
        # print(f"tasks: {tasks)
        # print(f"lambdas: {lambdas[:10]}")
        
    # losses.append(train_loss.item())

    # all_parameters = list(features.parameters())
    # for i in range(total_tasks):
    #     all_parameters += list(heads[i].parameters())

    # torch.autograd.functional.hessian(lambda x:heads[0](features(x)), tuple(all_parameters))

test_loss = 0
test_max_loss = 0
test_sz = 100
for i in range(test_sz):
    batch_loss, tasks = inner_loop(mode="val", batch_size=10, n_inner=20, verbose=False)
    test_loss += sum([i.item() for i in batch_loss])/10
    test_max_loss += max([i.item() for i in batch_loss])

train_loss = 0
train_max_loss = 0
for i in range(test_sz):
    batch_loss, tasks = inner_loop(mode="train", batch_size=10, n_inner=20, verbose=False)
    train_loss += sum([i.item() for i in batch_loss])/10
    train_max_loss += max([i.item() for i in batch_loss])

print(f"update lambdas: {update_lambdas}")
print(f"train loss {train_loss / test_sz}")
print(f"train max loss {train_max_loss / test_sz}")
print(f"test loss {test_loss / test_sz}")
print(f"max test loss {test_max_loss / test_sz}")

# from matplotlib import pyplot as plt
# plt.plot(range(len(losses)), losses)
# plt.show()


