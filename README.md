# Practical One-Shot Federated Learning for Cross-Silo Setting
This is the code for paper "Practical One-Shot Federated Learning for Cross-Silo Setting" [[PDF]](https://arxiv.org/pdf/2010.01017.pdf).

## Dependencies
* PyTorch 1.6.0
* torchvision 0.2.2
* pandas 0.24.2
* xgboost 1.0.2
* scikit-learn 0.22.1
* numpy 1.18.1
* scipy 1.4.1
* requests 0.23.0


## Sample Scripts
FedKT on MNIST using a CNN with heterogenous partition and 10 parties: `sh mnist_fedkt.sh`.

FedKT on SVHN using a CNN with heterogenous partition and 10 parties: `sh svhn_fedkt.sh`.



## Parameters

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model`                     | The model architecture. Options: `tree` (random forest), `gbdt_tree`, `mlp`, `simple-cnn`, `vgg-9` .|
| `alg` | The training algorithm. Options: `fedkt`, `fedavg`, `fedprox`, `scaffold`, `local_training`, `pate`
| `dataset`      | Dataset to use. Options: `a9a`, `cod-rna`, `mnist`, `celeba`. |
| `lr` | Learning rate for the local models. |
| `stu_lr` | Learning rate for the student models and the final model of FedKT. |
| `batch-size` | Batch size. |
| `epochs` | Number of local training epochs for FedAvg and FedProx. |
| `stu_epochs` | Number of training epochs for the models in FedKT. |
| `n_parties` | Number of parties. |
| `n_partition` | The number of partition in each party for FedKT. |
| `n_teacher_each_partition` | The number of teacher models in each partition for FedKT. |
| `comm_round`    | Number of communication rounds to use in FedAvg and FedProx. |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition. |
| `mu` | The proximal term parameter for FedProx. |
| `gamma` | The privacy parameter for FedKT-L1 and FedKT-L2. |
| `dp_level` | set to 1 to run FedKT-L1 and 2 to run FedKT-L2. |
| `max_tree_depth` | The tree depth for random forest and gbdt. |
| `n_stu_trees` | The number of trees for random forest and gbdt. |
| `datadir` | The path of the dataset. |
| `logdir` | The path to store the logs. |
| `device` | Specify the device to run the program. |
| `seed` | The initial seed. |
