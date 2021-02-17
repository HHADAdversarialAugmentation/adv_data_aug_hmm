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

f1_before = []
f1_after = []
for train_size in sorted(train_sizes):
	data = pd.read_csv(f"{folder}/f1_scores_train_size_{train_size}_adv_method_{adv_method}.csv")
	f1_before.append(np.mean(data['F1 before']))
	f1_after.append(np.mean(data['F1 after']))
    
plt.figure()
plt.plot(f1_before, label='original')
plt.plot(f1_after, label='augmented')
plt.title('F1 score')
plt.xticks(range(len(train_sizes)), train_sizes)
plt.legend()
plt.tight_layout()
plt.savefig(f'{folder}/plot.pdf')
          
    


