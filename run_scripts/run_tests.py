import os

sigma_list = [0.1, 0.25, 0.35]
random_seed_list = [0,1,2,3,4,5,6,7,8,9,10]
num_steps = 2
N = 100

for sigma in sigma_list:
    for random_seed in random_seed_list:
        print(f"Running for {sigma= }, {random_seed= }, {num_steps= }, {N= }")
        run_cmd = f"bash densepure_cifar10.sh {sigma} {num_steps} {random_seed} {N}"
        os.system(run_cmd)
