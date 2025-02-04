from pylab import plt

data_256 = {'exp_name': 'hnet_12-4_fmlp_c-32_bs-256_lr-3e-3', 'seed': 0, 'eval': {'epoch_1': {'cifar10': 89.71, 'cifar100': 45.1, 'imagenet1k': 20.69}, 'epoch_2': {'cifar10': 89.99, 'cifar100': 47.73, 'imagenet1k': 22.59}, 'epoch_5': {'cifar10': 89.97, 'cifar100': 49.08, 'imagenet1k': 23.37}, 'epoch_10': {'cifar10': 90.08, 'cifar100': 49.23, 'imagenet1k': 22.84}, 'epoch_20': {'cifar10': 89.15, 'cifar100': 47.93, 'imagenet1k': 21.36}}}
data_512 = {'exp_name': 'hnet_12-4_fmlp_c-32_bs-512_lr-1e-2', 'seed': 0, 'eval': {'epoch_1': {'cifar10': 89.79, 'cifar100': 44.31, 'imagenet1k': 20.74}, 'epoch_2': {'cifar10': 90.68, 'cifar100': 48.19, 'imagenet1k': 23.29}, 'epoch_5': {'cifar10': 90.97, 'cifar100': 51.7, 'imagenet1k': 25.56}, 'epoch_10': {'cifar10': 90.48, 'cifar100': 53.25, 'imagenet1k': 26.27}, 'epoch_20': {'cifar10': 90.36, 'cifar100': 52.6, 'imagenet1k': 25.46}}}
data_1024 = {'exp_name': 'hnet_12-4_fmlp_c-32_bs-1024_lr-2e-2', 'seed': 0, 'eval': {'epoch_1': {'cifar10': 87.4, 'cifar100': 35.19, 'imagenet1k': 13.89}, 'epoch_2': {'cifar10': 89.8, 'cifar100': 47.11, 'imagenet1k': 22.18}, 'epoch_5': {'cifar10': 90.1, 'cifar100': 51.13, 'imagenet1k': 24.63}, 'epoch_10': {'cifar10': 10.0, 'cifar100': 1.0, 'imagenet1k': 0.1}, 'epoch_20': {'cifar10': 10.0, 'cifar100': 1.0, 'imagenet1k': 0.1}}}
data_4096 = {'exp_name': 'hnet_12-4_fmlp_c-32_bs-4096_lr-3e-2', 'seed': 0, 'eval': {'epoch_1': {'cifar10': 56.09, 'cifar100': 15.07, 'imagenet1k': 3.02}, 'epoch_2': {'cifar10': 82.3, 'cifar100': 30.88, 'imagenet1k': 10.58}, 'epoch_5': {'cifar10': 89.84, 'cifar100': 42.25, 'imagenet1k': 19.05}, 'epoch_10': {'cifar10': 90.05, 'cifar100': 47.94, 'imagenet1k': 22.17}, 'epoch_20': {'cifar10': 89.43, 'cifar100': 49.5, 'imagenet1k': 22.72}}} 
data_16384 = {'exp_name': 'hnet_12-4_fmlp_c-32_bs-16384_lr-5e-2', 'seed': 0, 'eval': {'epoch_1': {'cifar10': 48.91, 'cifar100': 9.31, 'imagenet1k': 1.24}, 'epoch_2': {'cifar10': 54.04, 'cifar100': 11.29, 'imagenet1k': 1.94}, 'epoch_5': {'cifar10': 56.89, 'cifar100': 13.46, 'imagenet1k': 2.95}, 'epoch_10': {'cifar10': 10.0, 'cifar100': 1.0, 'imagenet1k': 0.1}, 'epoch_20': {'cifar10': 10.0, 'cifar100': 1.0, 'imagenet1k': 0.1}}}

data = [data_256, data_512, data_1024, data_4096, data_16384]
accs = [[ subitem["imagenet1k"] for subitem in list(item["eval"].values())] for item in data ]
accs = [max(item) for item in accs]
print(accs)

batch_sizes = [int(pow(2, i)) for i in [8, 9, 10, 12, 14]]

plt.plot(batch_sizes, accs)
plt.show()