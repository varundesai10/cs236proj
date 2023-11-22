import os

sigma_list = [0.1]
random_seed_list = [0,1,2,3,4,5,6,7,8,9]
num_steps = [5,2]
N = 1000

for steps in num_steps:
    for sigma in sigma_list:
        for random_seed in random_seed_list:
            print(f"Running for {sigma= }, {random_seed= }, {steps= }, {N= }")
            run_cmd = f"bash densepure_cifar10.sh {sigma} {steps} {random_seed} {N}"
            os.system(run_cmd)
