import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import glob
pkl_files = glob.glob("runs/*.pkl")

n_iterations = 10
best_lambdas = [[] for _ in range(n_iterations)]
best_no_lambdas = [[] for _ in range(n_iterations)]
ct = 0
for i in pkl_files:
    with open(i, "rb") as f:
        data = pickle.load(f)

    run = i.split('_')[1].split('.')[0]
    run = int(run)
    
    if data['lambdas']:
        max_lambdas = data['max']
        min_so_far_lambdas = [min(max_lambdas[0:i+1]) for i in range(len(max_lambdas))]
        for j in range(n_iterations):
            best_lambdas[j].append(min_so_far_lambdas[j])
    
    else:
        max_no_lambdas = data['max']
        min_so_far_no_lambdas = [min(max_no_lambdas[0:i+1]) for i in range(len(max_no_lambdas))]
        for j in range(n_iterations):
            best_no_lambdas[j].append(min_so_far_no_lambdas[j])


for o in [0, 1]:
    medians = []
    fq = []
    tq = []
    for i in range(n_iterations):
        if o == 1:
            a = best_lambdas[i]
        else:
            a = best_no_lambdas[i]
        a.sort()
        l = len(a)
        medians.append(a[l//2])
        fq.append(a[l//4])
        tq.append(a[3*l//4])
        
        if o == 1:
            c = 'r'
            label='Task-Robust'
        else:
            c = 'b'
            label='Standard'
    fq = np.log(np.array(fq))
    tq = np.log(np.array(tq))
    medians = np.log(np.array(medians))
    plt.title("Maximum MSE Loss with Task-Robust vs. Standard Training")
    plt.xlabel("Epoch")
    plt.ylabel("Best MSE Loss Until Epoch")
    plt.fill_between(np.arange(1, len(medians)+1), tq, fq, color=c, label=label, alpha=0.2)
    plt.plot(np.arange(1, len(medians)+1), medians, color=c)

plt.legend()
plt.show()
