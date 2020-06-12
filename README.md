# Model-Agnostic Round-Optimal Federated Learning via Knowledge Transfer
This is the code for paper "Model-Agnostic Round-Optimal Federated Learning via Knowledge Transfer".

Authors: [Qinbin Li](https://qinbinli.com), [Bingsheng He](https://www.comp.nus.edu.sg/~hebs/), [Dawn Song](https://people.eecs.berkeley.edu/~dawnsong/)
## Dependencies
* PyTorch 1.1.0
* torchvision 0.2.2
* pandas 0.24.2
* xgboost 1.0.2
* scikit-learn 0.22.1
* numpy 1.18.1
* scipy 1.4.1
* requests 0.23.0


## Sample Commands
FedKT on MNIST using a CNN with heterogenous partition and 10 parties.
```
python experiments.py --model=simple-cnn \
--dataset=mnist \
--alg=fedkt \
--lr=0.01 \
--batch-size=32 \
--epochs=100 \
--stu_epochs=100 \
--n_parties=10 \
--partition=hetero-dir \
--n_partition=5 \
--n_teacher_each_partition=10\
--beta=0.5\
--device='cuda:0'\
--datadir='./data/mnist/' \
--logdir='./logs/mnist/'
```


FedProx on MNIST using a MLP with heterogenous partition and 10 parties.
```
python experiments.py --model=mlp \
--net_config="784, 100, 100, 10" \
--dataset=mnist \
--alg=fedprox \
--comm_round=40 \
--lr=0.01 \
--mu=1 \
--batch-size=32 \
--epochs=10 \
--n_parties=10 \
--partition=hetero-dir \
--beta=0.5\
--device='cuda:0'\
--datadir='./data/mnist/' \
--logdir='./logs/mnist/'
```

## Parameters

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model`                     | The model architecture. Options: `tree` (random forest), `gbdt`, `mlp`, `simple-cnn`, `vgg-9` .|
| `alg` | The training algorithm. Options: `fedkt`, `fedavg`, `fedprox`, `local_training`, `pate`
| `dataset`      | Dataset to use. Options: `a9a`, `real-sim`, `mnist`, `celeba`. |
| `lr` | Learning rate for the local models. |
| `stu_lr` | Learning rate for the student models and the final model of FedKT. |
| `batch-size` | Batch size. |
| `epochs` | Number of local training epochs for FedAvg and FedProx. |
| `stu_epochs` | Number of training epochs for the models in FedKT. |
| `n_parties` | Number of parties. |
| `partition`  | Data partitioning strategy. Options: `hetero-dir` (heterogenous data partition)or `homo` (homogenous data partition). |
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