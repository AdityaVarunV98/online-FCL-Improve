import os

directory_path = "./output/supcon_cifar100_1000_bs_10/FCL/cifar100/w_favg/overlap/5clients/5tasks/30/5/resnet/sgd/01/1000/10/3/random/balanced_uncertainty/bregman/bottomk/"

if os.path.isdir(directory_path):
    print(f"The directory '{directory_path}' exists.")
else:
    print(f"The directory '{directory_path}' does not exist.")

# Or, if you just need to know if it exists at all (file or directory)
if os.path.exists(directory_path):
    print(f"The path '{directory_path}' exists.")
else:
    print(f"The path '{directory_path}' does not exist.")