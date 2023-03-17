# -*- coding: utf-8 -*-
"""
This software is the implementation of the following article submitted to TPAMI:
	Castellini A., Masillo F., Azzalini D., Amigoni F., Farinelli A., Adversarial Data Augmentation for HMM-based Anomaly Detection
In this stage, the software is intended for reviewers' use only.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

args_list = sys.argv
if "--result_dir" in args_list:
	folder = sys.argv[args_list.index("--result_dir")+1]
else:
	print("Mandatory parameter --result_dir not found please check input parameters")
	sys.exit()
if "--adv_method" in args_list:
	adv_method = sys.argv[args_list.index("--adv_method")+1]
else:
	print("Mandatory parameter --adv_method not found please check input parameterss")
	sys.exit()
if "--train_sizes" in args_list:
	train_sizes = sys.argv[args_list.index("--train_sizes")+1].split(',')
else:
	print("Mandatory parameter --train_sizes not found please check input parameters")
	sys.exit() 



train_sizes = [int(t) for t in train_sizes]
f1s = [] * len(train_sizes)


for train_size in sorted(train_sizes):
    f1 = []
    data = pd.read_csv(f"{folder}/f1_scores_train_size_{train_size}_adv_method_{adv_method}.csv")
    for col in data.columns:
        f1.append(np.nanmean(data[col]))
    f1s.append(f1)

plt.figure()
for i, col in enumerate(data.columns):
    to_plot = []
    for j, train_size in enumerate(sorted(train_sizes)):
        to_plot.append(f1s[j][i])
    if len(train_sizes) == 1:
        plt.scatter(train_sizes, to_plot, label=col)
    else:
        plt.plot(to_plot, label=col)

plt.title('F1 score')
if len(train_sizes) != 1:
    plt.xticks(range(len(train_sizes)), train_sizes)
plt.legend()
plt.tight_layout()
plt.savefig(f'{folder}/plot_{adv_method}-AUG.pdf')
