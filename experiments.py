import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import argparse
import logging
import os
import copy
import datetime
import xgboost as xgb


from model import *
from datasets import MNIST_truncated, CIFAR10_truncated, SVHN_custom, FashionMNIST_truncated, CustomTensorDataset, CelebA_custom
from trees import *

libsvm_datasets = {
    "a9a": "binary_cls",
    'real-sim': "binary_cls",
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP', metavar='N',
                        help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained in a training process')
    parser.add_argument('--n_parties', type=int, default=2, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--n_teacher_each_partition', type=int, default=1,
                        help='number of local nets in a partitioning of a party')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='which type of communication strategy is going to be used: fedenb/fedavg')
    parser.add_argument('--comm_round', type=int, default=50,
                            help='how many round of communications we should use')
    parser.add_argument('--trials', type=int, default=1, help="Number of trials for each run")
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=True, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--max_tree_depth', type=int, default=6, help='Max tree depth for the tree model')
    parser.add_argument('--n_ensemble_models', type=int, default=10, help="Number of the models in the final ensemble")
    parser.add_argument('--lambda_boost', type=float, default = 0.0, help="the regularization term for the FedBoost method")
    parser.add_argument('--train_local_student', type=int, default=1, help="whether use PATE to train local student models before aggregation")
    parser.add_argument('--auxiliary_data_portion', type=float, default=0.5, help="the portion of test data that is used as the auxiliary data for PATE")
    parser.add_argument('--stu_epochs', type=int, default=100, help='Number of epochs for the student model')
    parser.add_argument('--with_unlabeled', type=int, default=1, help='Whether there is public unlabeled data')
    parser.add_argument('--stu_lr', type=float, default=0.01, help='The learning rate for the student model')
    parser.add_argument('--is_local_split', type=int, default=1, help='Whether split the local data for local model training')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--ensemble_method', type=str, default='max_vote', help='Choice: max_vote or averaging')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--n_partition', type=int, default=1, help='The partition times of each party')
    parser.add_argument('--gamma', type=float, default=None, help='The parameter for differential privacy')
    parser.add_argument('--privacy_analysis_file_name', type=str, default=None, help='The file path to save the information for privacy analysis')
    parser.add_argument('--n_stu_trees', type=int, default=100, help='The number of trees in a student model')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--optimizer', type=str, default='adam', help='the optimizer')
    parser.add_argument('--vote_threshold', type=float, default=None, help='a voting threshold to filter the query')
    parser.add_argument('--local_training_epochs', type=int, default=None, help='the number of epochs for the local trainig alg')
    parser.add_argument('--dp_level', type=int, default=0, help='1 represents add dp on the server side. 2 represents add dp on the party side')
    parser.add_argument('--partition_step_size', type=int, default=6, help='how many groups of partitions we will have')
    parser.add_argument('--local_points', type=int, default=5000, help='the approximate fixed number of data points we will have on each local worker')
    parser.add_argument('--partition_step', type=int, default=0, help='how many sub groups we are going to use for a particular training process')
    parser.add_argument('--query_portion', type=float, default=0.5, help='how many queries are used to train the final model')
    parser.add_argument('--local_query_portion', type=float, default=0.5, help='how many queries are used to train the student models')
    parser.add_argument('--query_filter_threshold', type=float, default=0.6, help='the threshold to filter the queries')
    parser.add_argument('--filter_query', type=int, default=0, help='Whether to filter the query or not')
    parser.add_argument('--max_z', type=int, default=1, help='the maximum partition that may be influenced when changing a single record')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--fedkt_seed', type=int, default=None, help='the seed before run fedkt')
    args = parser.parse_args()
    return args


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_mnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_fmnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FashionMNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = FashionMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_svhn_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    svhn_train_ds = SVHN_custom(datadir, train=True, download=True, transform=transform)
    svhn_test_ds = SVHN_custom(datadir, train=False, download=True, transform=transform)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    # X_train = X_train.data.numpy()
    # y_train = y_train.data.numpy()
    # X_test = X_test.data.numpy()
    # y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)

def load_celeba_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    celeba_train_ds = CelebA_custom(datadir, split='train', target_type="attr", download=True, transform=transform)
    celeba_test_ds = CelebA_custom(datadir, split='test', target_type="attr", download=True, transform=transform)

    gender_index = celeba_train_ds.attr_names.index('Male')
    y_train =  celeba_train_ds.attr[:,gender_index:gender_index+1].reshape(-1)
    y_test = celeba_test_ds.attr[:,gender_index:gender_index+1].reshape(-1)

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (None, y_train, None, y_test)

def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(dataset, datadir, logdir, partition, n_parties, beta=0.4):
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'svhn':
        X_train, y_train, X_test, y_test = load_svhn_data(datadir)
    elif dataset == 'celeba':
        X_train, y_train, X_test, y_test = load_celeba_data(datadir)
    elif dataset in libsvm_datasets:
        # X_train, y_train = load_svmlight_file(datadir + dataset + '.train')
        # X_test, y_test = load_svmlight_file(datadir + dataset + '.test')
        X, y = load_svmlight_file(datadir + dataset)
        y_i_transform = np.zeros(y.size)
        for i in range(y.size):
            if y[i] == y[0]:
                y_i_transform[i] = 1
        y=np.copy(y_i_transform)
        X_train, X_test, y_train, y_test = train_test_split(X, y)



    n_train = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "hetero-dir":
        min_size = 0
        min_require_size = 10
        if dataset == 'mnist' or dataset == 'fmnist' or dataset == 'cifar10' or dataset == 'svhn':
            K = 10
        elif dataset in libsvm_datasets or dataset == 'celeba':
            K = 2
            # min_require_size = 100

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # print("proportions1: ", proportions)
                # print("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                # print("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # print("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # print("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break


        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]



    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)

    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def init_nets(net_configs, dropout_p, n_parties, args, n_teacher_each_partition = 1):

    n_total_nets = n_parties * n_teacher_each_partition
    nets = {net_i: None for net_i in range(n_total_nets)}

    for net_i in range(n_total_nets):
        if args.model == "mlp":
            input_size = net_configs[0]
            output_size = net_configs[-1]
            hidden_sizes = net_configs[1:-1]
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
        elif args.model == "vgg":
            net = vgg11()
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == "mnist" or args.dataset == 'fmnist':
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
        elif args.model == "vgg-9":
            if args.dataset == "mnist":
                net = ModerateCNNMNIST()
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN()
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2)
        elif args.model == "resnet":
            net = ResNet50()
        elif args.model == "vgg16":
            net = vgg16()
        elif args.model == 'lr':
            if args.dataset == 'a9a':
                net = LogisticRegression(123,2)
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def init_weights(m):
  if type(m)==nn.Linear or type(m)==nn.Conv2d:
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)

def get_trainable_parameters(net):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    # print("net.parameter.data:", list(net.parameters()))
    paramlist=list(trainable)
    N=0
    for params in paramlist:
        N+=params.numel()
        # print("params.data:", params.data)
    X=torch.empty(N,dtype=torch.float64)
    X.fill_(0.0)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            X[offset:offset+numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset+=numel
    # print("get trainable x:", X)
    return X


def put_trainable_parameters(net,X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset+=numel

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target, _) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            out = model(x)
            _, pred_label = torch.max(out.data, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)

def prepare_weight_matrix(n_classes, weights: dict):
    weights_list = {}

    for net_i, cls_cnts in weights.items():
        cls = np.array(list(cls_cnts.keys()))
        cnts = np.array(list(cls_cnts.values()))
        weights_list[net_i] = np.array([0] * n_classes, dtype=np.float32)
        weights_list[net_i][cls] = cnts
        weights_list[net_i] = torch.from_numpy(weights_list[net_i]).view(1, -1)

    return weights_list


def prepare_uniform_weights(n_classes, net_cnt, fill_val=1):
    weights_list = {}

    for net_i in range(net_cnt):
        temp = np.array([fill_val] * n_classes, dtype=np.float32)
        weights_list[net_i] = torch.from_numpy(temp).view(1, -1)

    return weights_list


def prepare_sanity_weights(n_classes, net_cnt):
    return prepare_uniform_weights(n_classes, net_cnt, fill_val=0)


def normalize_weights(weights):
    Z = np.array([])
    eps = 1e-6
    weights_norm = {}

    for _, weight in weights.items():
        if len(Z) == 0:
            Z = weight.data.numpy()
        else:
            Z = Z + weight.data.numpy()

    for mi, weight in weights.items():
        weights_norm[mi] = weight / torch.from_numpy(Z + eps)

    return weights_norm


def get_weighted_average_pred(models: list, weights: dict, x, device="cpu"):
    out_weighted = None

    # Compute the predictions
    for model_i, model in enumerate(models):
        #logger.info("Model: {}".format(next(model.parameters()).device))
        #logger.info("data device: {}".format(x.device))
        out = F.softmax(model(x), dim=-1)  # (N, C)
        # print("model(x):", model(x))
        # print("out:", out)

        weight = weights[model_i].to(device)


        if out_weighted is None:
            weight = weight.to(device)
            out_weighted = (out * weight)
        else:
            out_weighted += (out * weight)

    return out_weighted


def get_pred_votes(models, x, device="cpu"):
    # print("input x:", x)
    # Compute the predictions
    votes=torch.LongTensor([]).to(device)
    for model_i, model in enumerate(models):
        #logger.info("Model: {}".format(next(model.parameters()).device))
        #logger.info("data device: {}".format(x.device))
        out = F.softmax(model(x), dim=-1)  # (N, C)
        _, pred_label = torch.max(out,1)
        votes=torch.cat((votes, pred_label),dim=0)

    return votes


def compute_ensemble_accuracy(models: list, dataloader, n_classes, ensemble_method="max_vote", train_cls_counts=None,
                              uniform_weights=False, sanity_weights=False, device="cpu"):

    correct, total = 0, 0
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    was_training = [False]*len(models)
    for i, model in enumerate(models):
        if model.training:
            was_training[i] = True
            model.eval()
    if ensemble_method == "averaging":
        if uniform_weights is True:
            weights_list = prepare_uniform_weights(n_classes, len(models))
        elif sanity_weights is True:
            weights_list = prepare_sanity_weights(n_classes, len(models))
        else:
            weights_list = prepare_weight_matrix(n_classes, train_cls_counts)

        weights_norm = normalize_weights(weights_list)

    with torch.no_grad():
        for batch_idx, (x, target, _) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            target = target.long()
            if ensemble_method == "averaging":
                out = get_weighted_average_pred(models, weights_norm, x, device=device)
                _, pred_label = torch.max(out, 1)
            elif ensemble_method == "max_vote":
                votes = get_pred_votes(models, x, device=device)
                pred_label, _ = torch.mode(votes.view(-1, x.data.size()[0]), dim=0)


            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    #logger.info(correct, total)

    conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    for i, model in enumerate(models):
        if was_training[i]:
            model.train()

    return correct / float(total), conf_matrix

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
        #                        amsgrad=True)
    elif args_optimizer == 'adam_ams':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target, _) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            #for adam l2 reg
            # l2_reg = torch.zeros(1)
            # l2_reg.requires_grad = True

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        # logger.info('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

        if epoch % 10 == 0:
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
            train_acc = compute_accuracy(net, train_dataloader, device=device)
            test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc



def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, model, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
        #                        amsgrad=True)
    elif args_optimizer == 'adam_ams':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target, _) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            #for adam l2 reg
            # l2_reg = torch.zeros(1)
            # l2_reg.requires_grad = True

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            #for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
            loss += fed_prox_reg


            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        # logger.info('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if epoch % 10 == 0:
            train_acc = compute_accuracy(net, train_dataloader, device=device)
            test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc


def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def load_model(model, model_index, rank=0, device="cpu"):
    #
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model

def local_train_net(nets, args, net_dataidx_map, X_train = None, y_train = None, X_test = None, y_test = None, remain_test_dl = None, local_split=False, device="cpu"):
    # save local dataset
    # local_datasets = []
    n_teacher_each_partition = args.n_teacher_each_partition
    avg_acc = 0.0
    if local_split:
        split_datasets = []
        for party_id in range(args.n_parties):
            np.random.shuffle(net_dataidx_map[party_id])
            split_datasets.append(np.array_split(net_dataidx_map[party_id], args.n_teacher_each_partition))

    for net_id, net in nets.items():
        if not local_split:
            dataidxs = net_dataidx_map[net_id//n_teacher_each_partition]
        else:
            dataidxs = list(split_datasets[net_id//n_teacher_each_partition][net_id%n_teacher_each_partition])

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        if args.dataset in libsvm_datasets:
            party_id = net_id // n_teacher_each_partition
            train_ds_local = CustomTensorDataset(torch.tensor(X_train[net_dataidx_map[party_id]].toarray(), dtype=torch.float32),
                                                 torch.tensor(y_train[net_dataidx_map[party_id]], dtype=torch.long))
            public_ds = CustomTensorDataset(torch.tensor(X_test[:public_data_size].toarray(), dtype=torch.float32),
                                           torch.tensor(y_test[:public_data_size], dtype=torch.long))
            remain_test_ds = CustomTensorDataset(torch.tensor(X_test[public_data_size:].toarray(), dtype=torch.float32),
                                                torch.tensor(y_test[public_data_size:], dtype=torch.long))
            train_dl_local = data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            remain_test_dl = data.DataLoader(dataset=remain_test_ds, batch_size=32, shuffle=False)
        else:
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

        # local_datasets.append((train_dl_local, test_dl_local))

        # switch to global test set here

        # if remain_test_dl is not None:
        #     test_dl_global = remain_test_dl
        if args.alg == 'local_training':
            n_epoch = args.local_training_epochs
        else:
            n_epoch = args.epochs

        trainacc, testacc = train_net(net_id, net, train_dl_local, remain_test_dl, n_epoch, args.lr, args.optimizer, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_fedprox(nets, global_model, args, net_dataidx_map, X_train = None, y_train = None, X_test = None, y_test = None, remain_test_dl = None, local_split=False, device="cpu"):
    # save local dataset
    # local_datasets = []
    n_teacher_each_partition = args.n_teacher_each_partition
    avg_acc = 0.0
    if local_split:
        split_datasets = []
        for party_id in range(args.n_parties):
            np.random.shuffle(net_dataidx_map[party_id])
            split_datasets.append(np.array_split(net_dataidx_map[party_id], args.n_teacher_each_partition))

    for net_id, net in nets.items():
        if not local_split:
            dataidxs = net_dataidx_map[net_id//n_teacher_each_partition]
        else:
            dataidxs = list(split_datasets[net_id//n_teacher_each_partition][net_id%n_teacher_each_partition])

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        if args.dataset in libsvm_datasets:
            party_id = net_id//n_teacher_each_partition
            train_ds_local = CustomTensorDataset(torch.tensor(X_train[net_dataidx_map[party_id]].toarray(), dtype=torch.float32),
                                                 torch.tensor(y_train[net_dataidx_map[party_id]], dtype=torch.long))
            public_ds = CustomTensorDataset(torch.tensor(X_test[:public_data_size].toarray(), dtype=torch.float32),
                                           torch.tensor(y_test[:public_data_size], dtype=torch.long))
            remain_test_ds = CustomTensorDataset(torch.tensor(X_test[public_data_size:].toarray(), dtype=torch.float32),
                                                torch.tensor(y_test[public_data_size:], dtype=torch.long))
            train_dl_local = data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            remain_test_dl = data.DataLoader(dataset=remain_test_ds, batch_size=32, shuffle=False)
        else:
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)


        if args.alg == 'local_training':
            n_epoch = args.local_training_epochs
        else:
            n_epoch = args.epochs

        trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, remain_test_dl, n_epoch, args.lr, args.optimizer, args.mu, args.model, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_on_a_party(nets, args, net_dataidx_map, party_id, X_train = None, y_train = None, X_test = None, y_test = None, remain_test_dl = None, local_split=0, device="cpu"):
    # save local dataset
    # local_datasets = []
    n_teacher_each_partition = args.n_teacher_each_partition
    if local_split:
        split_datasets = []
        np.random.shuffle(net_dataidx_map[party_id])
        split_datasets = np.array_split(net_dataidx_map[party_id], args.n_teacher_each_partition)


    for net_id, net in nets.items():
        if not local_split:
            dataidxs = net_dataidx_map[party_id]
        else:
            dataidxs = list(split_datasets[net_id])

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        if args.dataset in libsvm_datasets:
            train_ds_local = CustomTensorDataset(torch.tensor(X_train[net_dataidx_map[party_id]].toarray(), dtype=torch.float32),
                                                 torch.tensor(y_train[net_dataidx_map[party_id]], dtype=torch.long))
            public_ds = CustomTensorDataset(torch.tensor(X_test[:public_data_size].toarray(), dtype=torch.float32),
                                           torch.tensor(y_test[:public_data_size], dtype=torch.long))
            remain_test_ds = CustomTensorDataset(torch.tensor(X_test[public_data_size:].toarray(), dtype=torch.float32),
                                                torch.tensor(y_test[public_data_size:], dtype=torch.long))
            train_dl_local = data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            test_dl_global = data.DataLoader(dataset=remain_test_ds, batch_size=32, shuffle=False)
        else:
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

        # local_datasets.append((train_dl_local, test_dl_local))

        # switch to global test set here
        if remain_test_dl is not None:
            test_dl_global = remain_test_dl
        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_global, args.stu_epochs, args.lr, args.optimizer, device=device)
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)

    nets_list = list(nets.values())
    return nets_list



def central_train_net_on_a_party(nets, args, X_train = None, y_train = None, X_test = None, y_test = None, remain_test_dl = None, local_split=0, device="cpu"):
    # save local dataset
    # local_datasets = []
    n_teacher_each_partition = args.n_teacher_each_partition

    dataidx_arr = np.arange(len(y_train))
    np.random.shuffle(dataidx_arr)
    # partition the local data to n_local_models parts
    dataidx = np.array_split(dataidx_arr, n_teacher_each_partition)

    for net_id, net in nets.items():

        # logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        if args.dataset in libsvm_datasets:
            train_ds_local = CustomTensorDataset(torch.tensor(X_train[net_dataidx_map[party_id]].toarray(), dtype=torch.float32),
                                                 torch.tensor(y_train[net_dataidx_map[party_id]], dtype=torch.long))
            public_ds = CustomTensorDataset(torch.tensor(X_test[:public_data_size].toarray(), dtype=torch.float32),
                                           torch.tensor(y_test[:public_data_size], dtype=torch.long))
            remain_test_ds = CustomTensorDataset(torch.tensor(X_test[public_data_size:].toarray(), dtype=torch.float32),
                                                torch.tensor(y_test[public_data_size:], dtype=torch.long))
            train_dl_local = data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            test_dl_global = data.DataLoader(dataset=remain_test_ds, batch_size=32, shuffle=False)
        else:
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidx[net_id])
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

        # local_datasets.append((train_dl_local, test_dl_local))

        # switch to global test set here
        if remain_test_dl is not None:
            test_dl_global = remain_test_dl
        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_global, args.epochs, args.lr, args.optimizer, device=device)
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)

    nets_list = list(nets.values())
    return nets_list

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset in ('mnist', 'fmnist', 'cifar10', 'svhn'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

        elif dataset == 'fmnist':
            dl_obj = FashionMNIST_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor()])
            transform_test = transforms.Compose([
                transforms.ToTensor()])

        elif dataset == 'svhn':
            dl_obj = SVHN_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(), normalize])


        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)


    elif dataset == 'celeba':
        dl_obj = CelebA_custom
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # transform_train = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, split='train', target_type="attr", transform=transform_train, download=True)
        test_ds = dl_obj(datadir, split='test', target_type="attr", transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl, train_ds, test_ds


def get_prediction_labels(models, n_classes, dataloader, gamma=None, method="max_vote", train_cls_counts=None,
                          uniform_weights=True, sanity_weights=False, is_subset=0, is_final_student = False, device="cpu"):
    # correct, total = 0, 0
    true_labels_list, pred_labels_list = [], []

    was_training = [False]*len(models)
    for i, model in enumerate(models):
        if model.training:
            was_training[i] = True
            model.eval()

    if method == "averaging":
        if uniform_weights is True:
            weights_list = prepare_uniform_weights(n_classes, len(models))
        elif sanity_weights is True:
            weights_list = prepare_sanity_weights(n_classes, len(models))
        else:
            weights_list = prepare_weight_matrix(n_classes, train_cls_counts)

        weights_norm = normalize_weights(weights_list)


    vote_counts_save = np.empty((n_classes, 0), dtype=int)
    correct, total = 0, 0
    top2_counts_differ_one = 0
    with torch.no_grad():
        for batch_idx, (x, target, index) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            target = target.long()

            if method == "averaging":
                out = get_weighted_average_pred(models, weights_norm, x, device=device)
                _, pred_label = torch.max(out, 1)
            elif method == "max_vote":
                votes = get_pred_votes(models, x, device=device)
                votes_view = votes.view(-1, x.data.size()[0])

                vote_counts_real = torch.LongTensor([]).to(device)
                for class_id in range(n_classes):
                    vote_count_real = (votes_view == class_id).sum(dim=0)
                    vote_counts_real = torch.cat((vote_counts_real, vote_count_real), dim=0)
                # print("vote_counts_real:", vote_counts_real)
                # print("vote counts view:", vote_counts_real.view(-1,x.data.size()[0]))
                vote_counts_save = np.append(vote_counts_save, vote_counts_real.view(-1,x.data.size()[0]).to("cpu").numpy(), axis=1)
                # print("vote_counts_save:", vote_counts_save)
                if gamma is None or gamma == 0:
                    pred_label, _ = torch.mode(votes_view, dim=0)
                else:
                    vote_counts = torch.FloatTensor([]).to(device)

                    for class_id in range(n_classes):
                        vote_count = (votes_view==class_id).sum(dim=0).float()
                        for i in range(len(vote_count)):
                            vote_count[i]+=np.random.laplace(loc=0.0, scale=float(1.0/gamma))
                        vote_counts=torch.cat((vote_counts,vote_count),dim=0)

                    # print("vote_counts:", vote_counts.to("cpu"))
                    # print("vote_counts view:", vote_counts.view(-1,x.data.size()[0]).to("cpu"))

                    _, pred_label=torch.max(vote_counts.view(-1,x.data.size()[0]),0)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list.append(list(pred_label.numpy()))
                true_labels_list.append(list(target.data.numpy()))

                if is_subset:
                    dataloader.dataset.dataset.target[index]=torch.LongTensor(list(pred_label.numpy()))
                else:
                    dataloader.dataset.target[index]=torch.LongTensor(list(pred_label.numpy()))

            else:
                pred_labels_list.append(list(pred_label.cpu().numpy()))
                true_labels_list.append(list(target.data.cpu().numpy()))

                if is_subset:
                    dataloader.dataset.dataset.target[index] = torch.LongTensor(list(pred_label.cpu().numpy()))
                else:
                    dataloader.dataset.target[index] = torch.LongTensor(list(pred_label.cpu().numpy()))
            # print("target:", target)
            # target = torch.LongTensor(list(pred_label.numpy())).cpu()

    vote_counts_save = np.transpose(vote_counts_save)


    top1_class_counts = np.zeros(500)
    top2_class_counts = np.zeros(500)
    top_diff_counts = np.zeros(500)
    for row in vote_counts_save:
        top2_counts = row[np.argsort(row)[-2:]]
        if top2_counts[1] - top2_counts[0] <= 1:
            top2_counts_differ_one+=1
        top_diff_counts[top2_counts[1] - top2_counts[0]] += 1
        top1_class_counts[top2_counts[1]] += 1
        top2_class_counts[top2_counts[0]] += 1

    return pred_labels_list, top2_counts_differ_one, vote_counts_save


def train_a_student(tea_nets, public_dataloader, public_ds, remain_test_dataloader, stu_net, n_classes, args,
                    gamma=None, is_subset=0, is_final_student=False, filter_query=0,device = 'cpu'):
    public_labels, top2_counts_differ_one, vote_counts_save = get_prediction_labels(tea_nets, n_classes, public_dataloader, gamma=gamma,
                                          method=args.ensemble_method, is_subset=is_subset, is_final_student= is_final_student, device=device)
    if filter_query:
        confident_query_idx = []
        for idx, row in enumerate(vote_counts_save):
            top2_counts = row[np.argsort(row)[-2:]]
            if top2_counts[1] - top2_counts[0] > 2:
                confident_query_idx.append(idx)
        print("len confident query idx:", len(confident_query_idx))
        logger.info("len confident query idx: %d" % len(confident_query_idx))
        # local_query_ds = data.Subset(public_ds, confident_query_idx)
        public_dataloader = data.DataLoader(dataset=public_ds, batch_size=32, sampler=data.SubsetRandomSampler(confident_query_idx))



    logger.info('len public_labels: %d' % len(public_labels))
    logger.info('Training student network')
    logger.info('n_public: %d' % len(public_dataloader))
    stu_net.to(device)

    train_acc = compute_accuracy(stu_net, public_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, stu_net.parameters()), lr=args.stu_lr, weight_decay=args.reg)
    elif args.optimizer == 'adam_ams':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, stu_net.parameters()), lr=args.stu_lr, weight_decay=args.reg,
                               amsgrad=True)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0


    for epoch in range(args.stu_epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target, _) in enumerate(public_dataloader):

            x, target = x.to(device), target.to(device)




            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = stu_net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

            # test_acc = compute_accuracy(stu_net, remain_test_dataloader, device=device)
            # test_acc, conf_matrix = compute_accuracy(stu_net, test_dataloader, get_confusion_matrix=True, device=device)

            # logger.info('>> Test accuracy in epoch %d: %f' % (epoch, test_acc))

        # logger.info('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if epoch % 10 == 0:

            train_acc = compute_accuracy(stu_net, public_dataloader, device=device)
            test_acc = compute_accuracy(stu_net, remain_test_dataloader, device=device)

            logger.info('>> Training accuracy: %f  Test accuracy: %f' % (train_acc, test_acc))


    logger.info(' ** Training complete **')
    return train_acc, top2_counts_differ_one, vote_counts_save



if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    # logging.basicConfig(filename='test.log', level=logger.info, filemode='w')
    # logging.info("test")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s-%d-%d' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"),args.init_seed, args.trials)
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)


    for n_exp in range(args.trials):
        seed = n_exp + args.init_seed
        logger.info("#" * 100)
        logger.info("Executing Trial %d with seed %d" % (n_exp, seed))
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info("Partitioning data")
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
            args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

        n_classes = len(np.unique(y_train))

        if args.model == 'lr':
            public_data_size = int(len(y_test) * args.auxiliary_data_portion)
            query_data_size = int(len(y_test) * args.query_portion)
            local_query_data_size = int(len(y_test) * args.local_query_portion)
            if args.dataset in libsvm_datasets:
                train_ds_global = CustomTensorDataset(torch.tensor(X_train.toarray(), dtype=torch.float32),
                                                     torch.tensor(y_train, dtype=torch.long))
                public_ds = CustomTensorDataset(torch.tensor(X_test[:public_data_size].toarray(), dtype=torch.float32),
                                                     torch.tensor(y_test[:public_data_size], dtype=torch.long))
                remain_test_ds = CustomTensorDataset(torch.tensor(X_test[public_data_size:].toarray(), dtype=torch.float32),
                                               torch.tensor(y_test[public_data_size:], dtype=torch.long))
                train_dl_global = data.DataLoader(dataset=train_ds_global, batch_size=args.batch_size, shuffle=True)
                public_dl = data.DataLoader(dataset=public_ds, batch_size=32, shuffle=True)
                remain_test_dl = data.DataLoader(dataset=remain_test_ds, batch_size=32, shuffle=False)
                query_dl = data.DataLoader(dataset=public_ds, batch_size=32,
                                           sampler=data.SubsetRandomSampler(list(range(query_data_size))))
                local_query_dl = data.DataLoader(dataset=public_ds, batch_size=32,
                                                 sampler=data.SubsetRandomSampler(list(range(local_query_data_size))))

        elif args.dataset not in libsvm_datasets:

            train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                              args.datadir,
                                                                                              args.batch_size,
                                                                                              32)

            print("len train_dl_global:", len(train_ds_global))

            public_data_size = int(len(test_ds_global) * args.auxiliary_data_portion)
            query_data_size = int(len(test_ds_global) * args.query_portion)
            local_query_data_size = int(len(test_ds_global) * args.local_query_portion)
            remain_data_size = len(test_ds_global) - public_data_size
            # unquery_size = len(test_ds_global) - query_data_size
            # local_unquery_size = len(test_ds_global) - local_query_data_size
            public_ds, remain_test_ds = data.random_split(test_ds_global,
                                                          [public_data_size, remain_data_size])

            public_dl = data.DataLoader(dataset=public_ds, batch_size=32, shuffle=True)
            # query_dl = data.DataLoader(dataset=query_ds, batch_size=32, shuffle=False)
            # local_query_dl = data.DataLoader(dataset=local_query_ds, batch_size=32, shuffle=False)
            remain_test_dl = data.DataLoader(dataset=remain_test_ds, batch_size=32, shuffle=False)

            query_dl = data.DataLoader(dataset=public_ds, batch_size=32,
                                       sampler=data.SubsetRandomSampler(list(range(query_data_size))))
            local_query_dl = data.DataLoader(dataset=public_ds, batch_size=32,
                                             sampler=data.SubsetRandomSampler(list(range(local_query_data_size))))

        if args.alg == 'fedavg':
            logger.info("Initializing nets")
            args.n_teacher_each_partition = 1
            args.is_local_split = 0
            nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
            # print("nets:", nets)
            # torch.manual_seed(seed)
            global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
            global_model = global_models[0]
            # global_model = copy.deepcopy(nets[0])
            # print("global model:", global_model)



            for round in range(args.comm_round):
                logger.info("in comm round:" + str(round))

                # print("nets0 paras before:", list(nets[0].parameters()))
                # send back global model

                global_para = get_trainable_parameters(global_model)
                # global_para = get_trainable_parameters(nets[0])

                if args.is_same_initial:
                    for net_id, net in nets.items():
                        put_trainable_parameters(net, global_para)
                else:
                    if round!=0:
                        for net_id, net in nets.items():
                            put_trainable_parameters(net, global_para)


                # print("global paras:", list(global_model.parameters()))
                # print("nets0 paras after:", list(nets[0].parameters()))



                local_train_net(nets, args, net_dataidx_map, X_train, y_train, X_test, y_test, remain_test_dl = remain_test_dl, local_split=False, device=device)
                # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

                # update global model
                total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]
                weights = [get_trainable_parameters(nets[i].cpu()) for i in range(args.n_parties)]
                # print("avg freqs:", fed_avg_freqs)
                # print("weights:", weights)
                average_weight = sum(weights[i] * fed_avg_freqs[i] for i in range(args.n_parties))
                # print("average_weight:", average_weight)
                put_trainable_parameters(global_model, average_weight)

                # global model accuracy
                # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

                # print("len test_ds_global:", len(test_ds_global))


                logger.info('global n_training: %d' % len(train_dl_global))
                logger.info('global n_test: %d' % len(remain_test_dl))

                train_acc = compute_accuracy(global_model, train_dl_global)
                test_acc, conf_matrix = compute_accuracy(global_model, remain_test_dl, get_confusion_matrix=True)

                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)


        elif args.alg == 'fedprox':
            logger.info("Initializing nets")
            args.n_teacher_each_partition = 1
            args.is_local_split = 0
            nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
            # print("nets:", nets)
            # torch.manual_seed(seed)
            global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
            global_model = global_models[0]
            # global_model = copy.deepcopy(nets[0])
            # print("global model:", global_model)


            for round in range(args.comm_round):
                logger.info("in comm round:" + str(round))

                # print("nets0 paras before:", list(nets[0].parameters()))
                # send back global model

                global_para = get_trainable_parameters(global_model)
                # global_para = get_trainable_parameters(nets[0])

                if args.is_same_initial:
                    for net_id, net in nets.items():
                        put_trainable_parameters(net, global_para)
                else:
                    if round!=0:
                        for net_id, net in nets.items():
                            put_trainable_parameters(net, global_para)


                # print("global paras:", list(global_model.parameters()))
                # print("nets0 paras after:", list(nets[0].parameters()))



                local_train_net_fedprox(nets, global_model, args, net_dataidx_map, X_train, y_train, X_test, y_test, remain_test_dl = remain_test_dl, local_split=False, device=device)
                global_model.to('cpu')
                # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

                # update global model
                total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]
                weights = [get_trainable_parameters(nets[i].cpu()) for i in range(args.n_parties)]
                # print("avg freqs:", fed_avg_freqs)
                # print("weights:", weights)
                average_weight = sum(weights[i] * fed_avg_freqs[i] for i in range(args.n_parties))
                # print("average_weight:", average_weight)
                put_trainable_parameters(global_model, average_weight)

                # global model accuracy
                # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

                # print("len test_ds_global:", len(test_ds_global))


                logger.info('global n_training: %d' % len(train_dl_global))
                logger.info('global n_test: %d' % len(remain_test_dl))

                train_acc = compute_accuracy(global_model, train_dl_global)
                test_acc, conf_matrix = compute_accuracy(global_model, remain_test_dl, get_confusion_matrix=True)

                logger.info('>> Global Model Train accuracy: %f' % train_acc)
                logger.info('>> Global Model Test accuracy: %f' % test_acc)

        elif args.alg == 'local_training':
            args.n_teacher_each_partition = 1
            args.is_local_split = 0
            avg_acc = 0.0
            local_acc_list = []
            if args.model == 'tree':
                # logger.info("Initializing trees")
                public_data_size = int(len(y_test) * args.auxiliary_data_portion)
                for party_id in range(args.n_parties):
                    local_forest = RandomForestClassifier(max_depth = args.max_tree_depth, n_estimators = args.n_stu_trees)
                    local_forest.fit(X_train[net_dataidx_map[party_id]], y_train[net_dataidx_map[party_id]])
                    local_acc = local_forest.score(X_test[public_data_size:], y_test[public_data_size:])
                    logger.info("In party %d local test acc: %f" % (party_id, local_acc))
                    avg_acc += local_acc
                    local_acc_list.append(local_acc)
                avg_acc /= args.n_parties
                logger.info("average test acc: %f" % avg_acc)
                logger.info("min test acc: %f" % min(local_acc_list))
                logger.info("max test acc: %f" % max(local_acc_list))
            elif args.model == 'gbdt':
                public_data_size = int(len(y_test) * args.auxiliary_data_portion)
                param = {'max_depth': args.max_tree_depth, 'objective': 'binary:logistic', 'gamma':1, 'lambda':1, 'eta':0.1}
                for party_id in range(args.n_parties):
                    gbdt = xgb.XGBClassifier(max_depth=args.max_tree_depth, n_estimators = args.n_stu_trees, learning_rate = args.lr, gamma = 1, reg_lambda = 1, tree_method='hist')
                    gbdt.fit(X_train[net_dataidx_map[party_id]], y_train[net_dataidx_map[party_id]])
                    local_acc = gbdt.score(X_test[public_data_size:], y_test[public_data_size:])
                    logger.info("In party %d local test acc: %f" % (party_id, local_acc))
                    avg_acc += local_acc
                    local_acc_list.append(local_acc)
                    # dtrain = xgb.DMatrix(X_train[net_dataidx_map[party_id]], y_train[net_dataidx_map[party_id]])
                    # gbdt = xgb.train(param, dtrain, args.n_stu_trees)
                avg_acc /= args.n_parties
                logger.info("average test acc: %f" % avg_acc)
                logger.info("min test acc: %f" % min(local_acc_list))
                logger.info("max test acc: %f" % max(local_acc_list))
            else:
                # logger.info("Initializing nets")
                if args.local_training_epochs is None:
                    args.local_training_epochs = args.stu_epochs
                nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)

                local_train_net(nets, args, net_dataidx_map, X_train, y_train, X_test, y_test, remain_test_dl=remain_test_dl, local_split=False,
                                device=device)
        elif args.alg == 'all_in':
            if args.model == 'tree':
                public_data_size = int(len(y_test) * args.auxiliary_data_portion)
                forest = RandomForestClassifier(max_depth = args.max_tree_depth, n_estimators = args.n_stu_trees)
                forest.fit(X_train, y_train)
                acc = forest.score(X_test[public_data_size:], y_test[public_data_size:])
                logger.info("all in test acc: %f" % acc)
            elif args.model == 'gbdt':
                public_data_size = int(len(y_test) * args.auxiliary_data_portion)
                gbdt = xgb.XGBClassifier(max_depth=args.max_tree_depth, n_estimators = args.n_stu_trees, learning_rate = args.lr, gamma = 1, reg_lambda = 1, tree_method='hist')
                gbdt.fit(X_train, y_train,
                         eval_set=[(X_train, y_train), (X_test[public_data_size:], y_test[public_data_size:])],
                         eval_metric='error',
                         verbose=True)
                evals_result = gbdt.evals_result()
                print("evals result:", evals_result)
                logger.info("evals result: " + ' '.join([str(elem) for elem in evals_result['validation_1']['error']]))
                logger.info("eval result 50 rounds: %f" % evals_result['validation_1']['error'][49])
                logger.info("eval last result: %f" % evals_result['validation_1']['error'][args.n_stu_trees-1])
            else:
                nets, _, _ = init_nets(args.net_config, args.dropout_p, 1, args)
                nets[0].to(device)
                trainacc, testacc = train_net(0, nets[0], train_dl_global, remain_test_dl, args.stu_epochs, args.lr,
                                              args.optimizer, device=device)
                logger.info("all in test acc: %f" % testacc)

        elif args.alg == 'fedenb':
            logger.info("Initializing nets")
            nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args, args.n_teacher_each_partition)

            logger.info("Training nets")
            nets_list = local_train_net(nets, args, net_dataidx_map, X_train, y_train, X_test, y_test, local_split=False, device=device)
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
            logger.info("Compute uniform ensemble accuracy")
            uens_train_acc, _ = compute_ensemble_accuracy(nets_list, train_dl_global, n_classes,  ensemble_method=args.ensemble_method, uniform_weights=True, device=device)
            uens_test_acc, _ = compute_ensemble_accuracy(nets_list, test_dl_global, n_classes, ensemble_method=args.ensemble_method, uniform_weights=True, device=device)

            logger.info("Uniform ensemble (Train acc): {}".format(uens_train_acc))
            logger.info("Uniform ensemble (Test acc): {}".format(uens_test_acc))

        elif args.alg == 'fedboost':
            if args.model != 'tree':
                print("not supported yet")
                exit(1)
            logger.info("Initializing trees")
            trees = init_trees(args.max_tree_depth, args.n_parties, args.n_teacher_each_partition, libsvm_datasets[args.dataset])

            logger.info("Training trees")
            trees_list = local_train_trees(trees, args, net_dataidx_map, X_train, y_train, X_test, y_test)
            ens_train_acc = compute_tree_ensemble_accuracy(trees_list, X_train, y_train)
            ens_test_acc = compute_tree_ensemble_accuracy(trees_list, X_test, y_test)
            logger.info("All trees ensemble train acc: %f" % ens_train_acc)
            logger.info("All trees ensemble test acc: %f" % ens_test_acc)
            fedboost(trees, args, net_dataidx_map, X_train, y_train, X_test, y_test, libsvm_datasets[args.dataset])

        elif args.alg =='pate':
            if args.model == 'tree' or args.model == 'gbdt':
                logger.info("Initializing trees")
                public_data_size = int(len(y_test) * args.auxiliary_data_portion)
                query_data_size = int(len(y_test) * args.query_portion)
                local_query_data_size = int(len(y_test) * args.local_query_portion)
                tea_trees = []
                n_parti_top2_differ_one = np.zeros(args.n_parties)
                n_instances_portion = np.zeros(args.n_parties)
                vote_counts_parties = []
                # filter_query=0

                trees = init_trees(args.max_tree_depth, 1, args.n_teacher_each_partition, libsvm_datasets[args.dataset], args)

                # logger.info("In party %d Train local trees" % party_id)
                central_train_trees_in_a_party(trees, args, X_train, y_train, X_test, y_test)
                if args.model == 'tree':
                    stu_forest = RandomForestClassifier(max_depth = args.max_tree_depth, n_estimators=args.n_stu_trees)
                elif args.model == 'gbdt':
                    stu_forest = xgb.XGBClassifier(max_depth=args.max_tree_depth, n_estimators = args.n_stu_trees, learning_rate=args.lr,
                                                      gamma=1, reg_lambda=1, tree_method='hist')
                filter_query = args.filter_query
                if args.dp_level == 2:
                    gamma = args.gamma
                    top2_counts_differ_one, vote_counts_in_a_party = train_a_student_tree(trees, X_test[:local_query_data_size],
                                                                                          y_test[:local_query_data_size], 2,
                                                                                          stu_forest, gamma, filter_query)
                else:
                    gamma = 0
                    top2_counts_differ_one, vote_counts_in_a_party = train_a_student_tree(trees, X_test[:local_query_data_size],
                                                                                              y_test[:local_query_data_size],
                                                                                              2, stu_forest,
                                                                                              gamma, filter_query)
                test_acc = stu_forest.score(X_test[public_data_size:], y_test[public_data_size:])
                logger.info("central pate test acc %f" % test_acc)

            else:
                nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args,
                                       args.n_teacher_each_partition)
                nets_list = central_train_net_on_a_party(nets, args, X_train, y_train, X_test,
                                                       y_test, remain_test_dl, local_split=args.is_local_split,
                                                       device=device)
                # nets_list_partition.append(nets_list)

                if args.train_local_student:
                    if not args.with_unlabeled:
                        print("need public unlabeled data!")
                        exit(1)

                    # logger.info("in party %d" % party_id)
                    stu_nets, _, _ = init_nets(args.net_config, args.dropout_p, 1, args, 1)
                    stu_net = stu_nets[0]
                    if args.dp_level == 2:
                        gamma = args.gamma
                        _, top2_counts_differ_one, vote_counts_in_a_party = train_a_student(nets_list,
                                                                                            local_query_dl,
                                                                                            public_ds,
                                                                                            remain_test_dl,
                                                                                            stu_net, n_classes,
                                                                                            args, gamma=gamma,
                                                                                            is_subset=1,
                                                                                            filter_query=0,
                                                                                            device=device)
                    else:
                        gamma = 0
                        _, top2_counts_differ_one, vote_counts_in_a_party = train_a_student(nets_list, local_query_dl,
                                                                                            public_ds,
                                                                                            remain_test_dl, stu_net,
                                                                                            n_classes, args, gamma=gamma,
                                                                                            is_subset=1,
                                                                                            filter_query=0,
                                                                                            device=device)


                    local_stu_test_acc = compute_accuracy(stu_net, remain_test_dl, device=device)
                    # local_stu_test_acc = compute_accuracy(stu_net, remain_test_dl, device=device)

                    logger.info("stu_test_acc: %f" % local_stu_test_acc)

                # tea_nets.append(stu_net)
                else:
                    for net in nets_list:
                        tea_nets.append(net)

        elif args.alg =='pate2':
            args.n_teacher_each_partition = 1

            if args.model == 'tree' or args.model == 'gbdt':
                logger.info("Initializing trees")
                public_data_size = int(len(y_test) * args.auxiliary_data_portion)
                query_data_size = int(len(y_test) * args.query_portion)
                local_query_data_size = int(len(y_test) * args.local_query_portion)
                tea_trees = []
                n_parti_top2_differ_one = np.zeros(args.n_parties)
                n_instances_portion = np.zeros(args.n_parties)
                vote_counts_parties = []
                # filter_query=0



                # logger.info("In party %d Train local trees" % party_id)
                tree_list = []
                for party_id in range(args.n_parties):
                    trees = init_trees(args.max_tree_depth, 1, 1,
                                       libsvm_datasets[args.dataset], args)
                    local_train_trees_in_a_party(trees, args, net_dataidx_map[party_id], X_train, y_train, X_test, y_test)
                    tree_list.append(trees[0])
                if args.model == 'tree':
                    stu_forest = RandomForestClassifier(max_depth = args.max_tree_depth, n_estimators=args.n_stu_trees)
                elif args.model == 'gbdt':
                    stu_forest = xgb.XGBClassifier(max_depth=args.max_tree_depth, n_estimators = args.n_stu_trees, learning_rate=args.lr,
                                                      gamma=1, reg_lambda=1, tree_method='hist')
                filter_query = args.filter_query
                if args.dp_level == 2:
                    gamma = args.gamma
                    top2_counts_differ_one, vote_counts_in_a_party = train_a_student_tree(tree_list, X_test[:local_query_data_size],
                                                                                          y_test[:local_query_data_size], 2,
                                                                                          stu_forest, gamma, filter_query)
                else:
                    gamma = 0
                    top2_counts_differ_one, vote_counts_in_a_party = train_a_student_tree(tree_list, X_test[:local_query_data_size],
                                                                                              y_test[:local_query_data_size],
                                                                                              2, stu_forest,
                                                                                              gamma, filter_query)
                test_acc = stu_forest.score(X_test[public_data_size:], y_test[public_data_size:])
                logger.info("central pate test acc %f" % test_acc)
            else:



                tea_model = []

                for party_id in range(args.n_parties):
                    nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args,
                                           args.n_teacher_each_partition)
                    nets_list = local_train_net_on_a_party(nets, args, net_dataidx_map, party_id, X_train, y_train, X_test,
                                                            y_test, remain_test_dl, local_split=0,
                                                            device=device)
                    tea_model.append(nets_list[0])
                # nets_list_partition.append(nets_list)

                if args.train_local_student:
                    if not args.with_unlabeled:
                        print("need public unlabeled data!")
                        exit(1)

                    logger.info("in party %d" % party_id)
                    stu_nets, _, _ = init_nets(args.net_config, args.dropout_p, 1, args, 1)
                    stu_net = stu_nets[0]
                    if args.dp_level == 2:
                        gamma = args.gamma
                        _, top2_counts_differ_one, vote_counts_in_a_party = train_a_student(tea_model,
                                                                                            local_query_dl,
                                                                                            public_ds,
                                                                                            remain_test_dl,
                                                                                            stu_net, n_classes,
                                                                                            args, gamma=gamma,
                                                                                            is_subset=1,
                                                                                            filter_query=0,
                                                                                            device=device)
                    else:
                        gamma = 0
                        _, top2_counts_differ_one, vote_counts_in_a_party = train_a_student(tea_model, local_query_dl,
                                                                                            public_ds,
                                                                                            remain_test_dl, stu_net,
                                                                                            n_classes, args, gamma=gamma,
                                                                                            is_subset=1,
                                                                                            filter_query=0,
                                                                                            device=device)



                    local_stu_test_acc = compute_accuracy(stu_net, remain_test_dl, device=device)
                    # local_stu_test_acc = compute_accuracy(stu_net, remain_test_dl, device=device)

                    logger.info("stu_test_acc: %f" % local_stu_test_acc)

                # tea_nets.append(stu_net)

        elif args.alg == 'fedkt' or args.alg == 'fedkt_fedavg' or args.alg=='fedkt_fedprox':
            if args.fedkt_seed is not None:
                np.random.seed(args.fedkt_seed)
                torch.manual_seed(args.fedkt_seed)
            if args.model == 'tree' or args.model == 'gbdt' or args.model == 'random_forest' or args.model == 'gbdt_ntree':
                logger.info("Initializing trees")
                # X_test = np.random.shuffle(X_test)
                # y_test = np.random.shuffle(y_test)
                public_data_size = int(len(y_test) * args.auxiliary_data_portion)
                query_data_size = int(len(y_test) * args.query_portion)
                local_query_data_size = int(len(y_test) * args.local_query_portion)
                tea_trees = []
                n_parti_top2_differ_one = np.zeros(args.n_parties)
                n_instances_portion = np.zeros(args.n_parties)
                vote_counts_parties = []
                # filter_query=0
                for party_id in range(args.n_parties):

                    n_instances_portion[party_id] = len(net_dataidx_map[party_id]) / X_train.shape[0]
                    for i in range(args.n_partition):
                        trees = init_trees(args.max_tree_depth, 1, args.n_teacher_each_partition, libsvm_datasets[args.dataset], args)

                        logger.info("In party %d Train local trees" % party_id)
                        local_train_trees_in_a_party(trees, args, net_dataidx_map[party_id], X_train, y_train, X_test, y_test)
                        if args.model == 'tree' or args.model == 'random_forest':
                            stu_forest = RandomForestClassifier(max_depth = args.max_tree_depth, n_estimators=args.n_stu_trees)
                        elif args.model == 'gbdt' or args.model == 'gbdt_ntree':
                            stu_forest = xgb.XGBClassifier(max_depth=args.max_tree_depth, n_estimators = args.n_stu_trees, learning_rate=args.lr,
                                                              gamma=1, reg_lambda=1, tree_method='hist')
                        filter_query = args.filter_query
                        if i < args.max_z:
                            filter_query=0
                        if args.dp_level == 2:
                            gamma = args.gamma
                            top2_counts_differ_one, vote_counts_in_a_party = train_a_student_tree(trees, X_test[:local_query_data_size],
                                                                                                  y_test[:local_query_data_size], 2,
                                                                                                  stu_forest, gamma, filter_query)
                        else:
                            gamma = 0
                            top2_counts_differ_one, vote_counts_in_a_party = train_a_student_tree(trees, X_test[:local_query_data_size],
                                                                                                      y_test[:local_query_data_size],
                                                                                                      2, stu_forest,
                                                                                                      gamma, filter_query)
                        # logger.info("vote counts in a local party:")
                        # logger.info('\n'.join('\t'.join('%d' %x for x in y) for y in vote_counts_in_a_party))
                        vote_counts_parties = np.append(vote_counts_parties,vote_counts_in_a_party)
                        print("top2_counts_differ_one: ", top2_counts_differ_one)
                        logger.info("top2_counts_differ_one: %d" % top2_counts_differ_one)
                        if top2_counts_differ_one != 0:
                            n_parti_top2_differ_one[party_id] += 1
                        tea_trees.append(stu_forest)
                # vote_counts_parties = np.reshape(vote_counts_parties, (args.n_parties,-1))
                if args.model == 'tree' or args.model == 'random_forest':
                    final_forest = RandomForestClassifier(max_depth=args.max_tree_depth, n_estimators=args.n_stu_trees)
                elif args.model == 'gbdt' or args.model == 'gbdt_ntree':
                    final_forest = xgb.XGBClassifier(max_depth=args.max_tree_depth, n_estimators = args.n_stu_trees, learning_rate=args.lr,
                                                   gamma=1, reg_lambda=1, tree_method='hist')
                if args.dp_level == 1:
                    gamma = args.gamma
                    _, vote_counts = train_a_student_tree(tea_trees, X_test[:query_data_size],
                                                                 y_test[:query_data_size], 2, final_forest, gamma, 0)
                else:
                    gamma = 0
                    _, vote_counts = train_a_student_tree(tea_trees, X_test[:query_data_size],
                                                                 y_test[:query_data_size], 2, final_forest, gamma, 0)
                test_acc = final_forest.score(X_test[public_data_size:], y_test[public_data_size:])
                logger.info("global test acc %f" % test_acc)

                if args.privacy_analysis_file_name is not None:
                    file_path = os.path.join(args.logdir, args.privacy_analysis_file_name + "-%d" % n_exp)
                else:
                    file_path = os.path.join(args.logdir, log_path + "-%d" % n_exp)
                file_path1 = file_path + '-dp0'
                file_path2 = file_path + '-dp1'
                np.savez(file_path1, n_instances_portion, n_parti_top2_differ_one, vote_counts)
                print("vote counts parties:", vote_counts_parties)
                np.savez(file_path2, n_instances_portion, n_parti_top2_differ_one, vote_counts_parties.reshape(-1,2))

                avg_acc = 0.0
                for party_id in range(args.n_parties):
                    logger.info("Local training")
                    local_forest = RandomForestClassifier(max_depth = args.max_tree_depth, n_estimators=args.n_stu_trees)
                    local_forest.fit(X_train[net_dataidx_map[party_id]], y_train[net_dataidx_map[party_id]])
                    local_acc = local_forest.score(X_test[public_data_size:], y_test[public_data_size:])
                    logger.info("In party %d local test acc: %f" % (party_id, local_acc))
                    avg_acc += local_acc
                avg_acc /= args.n_parties
                logger.info("avg local acc: %f" % avg_acc)
                global_model = final_forest
            else:
                logger.info("Initializing nets")
                if args.n_teacher_each_partition == 1:
                    args.is_local_split = 0
                    args.train_local_student = 0
                logger.info("Training nets")


                tea_nets = []
                n_parti_top2_differ_one = np.zeros(args.n_parties)
                n_instances_portion = np.zeros(args.n_parties)
                vote_counts_parties = []


                for party_id in range(args.n_parties):


                    #start training student models
                    for i in range(args.n_partition):

                        filter_query = args.filter_query
                        if i < args.max_z:
                            filter_query = 0

                        nets, _, _ = init_nets(args.net_config, args.dropout_p, 1, args,
                                               args.n_teacher_each_partition)
                        nets_list = local_train_net_on_a_party(nets, args, net_dataidx_map, party_id, X_train, y_train, X_test, y_test, remain_test_dl, local_split=args.is_local_split,
                                                    device=device)
                        # nets_list_partition.append(nets_list)

                        if args.train_local_student:
                            if not args.with_unlabeled:
                                print("need public unlabeled data!")
                                exit(1)

                            logger.info("in party %d" % party_id)
                            stu_nets, _, _ = init_nets(args.net_config, args.dropout_p, 1, args, 1)
                            stu_net = stu_nets[0]
                            if args.dp_level == 2:
                                gamma = args.gamma
                                _, top2_counts_differ_one, vote_counts_in_a_party = train_a_student(nets_list,
                                                                                                    local_query_dl,
                                                                                                    public_ds,
                                                                                                    remain_test_dl,
                                                                                                    stu_net, n_classes,
                                                                                                    args, gamma=gamma,
                                                                                                    is_subset=1,
                                                                                                    filter_query=filter_query,
                                                                                                    device=device)
                            else:
                                gamma = 0
                                _, top2_counts_differ_one, vote_counts_in_a_party = train_a_student(nets_list, local_query_dl, public_ds,
                                                remain_test_dl, stu_net, n_classes, args, gamma=gamma, is_subset=1, filter_query=filter_query,device=device)
                            # logger.info("vote counts in a local party:")
                            # logger.info('\n'.join('\t'.join('%d' % x for x in y) for y in vote_counts_in_a_party))
                            vote_counts_parties = np.append(vote_counts_parties, vote_counts_in_a_party)
                            # print("len(nets_list): ", len(nets_list))
                            # print("vote counts in a party: ", vote_counts_in_a_party)
                            # print("top2_counts_differ_one: ", top2_counts_differ_one)
                            # logger.info("top2_counts_differ_one: %d" % top2_counts_differ_one)
                            if top2_counts_differ_one != 0:
                                n_parti_top2_differ_one[party_id] += 1


                            local_stu_test_acc = compute_accuracy(stu_net, remain_test_dl, device=device)
                            # local_stu_test_acc = compute_accuracy(stu_net, remain_test_dl, device=device)

                            logger.info("local_stu_test_acc: %f" % local_stu_test_acc)

                            tea_nets.append(stu_net)
                        else:
                            for net in nets_list:
                                tea_nets.append(net)

                    n_instances_portion[party_id] = len(net_dataidx_map[party_id]) / len(train_ds_global)

                # print("portion sum:", np.sum(n_instances_portion))

                if args.with_unlabeled:
                    global_stu_nets, _, _=init_nets(args.net_config, args.dropout_p, 1, args, 1)
                    global_stu_net = global_stu_nets[0]
                    if args.dp_level == 1:
                        gamma = args.gamma
                        _, _, vote_counts = train_a_student(tea_nets, query_dl, public_ds, remain_test_dl, global_stu_net,
                                                            n_classes, args, gamma=args.gamma,
                                                            is_subset=1, is_final_student=True, filter_query=0,device=device)
                    else:
                        gamma = 0
                        _, _, vote_counts = train_a_student(tea_nets, query_dl, public_ds, remain_test_dl, global_stu_net,
                                                            n_classes, args, gamma=args.gamma,
                                                            is_subset=1, is_final_student=True, filter_query=0,device=device)
                    # can change to local data to train the student

                    global_stu_test_acc = compute_accuracy(global_stu_net, remain_test_dl, device=device)
                    logger.info("global_stu_test_acc: %f"% global_stu_test_acc)
                else:
                    print("not supported yet")


                if args.privacy_analysis_file_name is not None:
                    file_path = os.path.join(args.logdir, args.privacy_analysis_file_name + "-%d" % n_exp)
                else:
                    file_path = os.path.join(args.logdir, log_path + "-%d" % n_exp)
                file_path1 = file_path + '-dp0'
                file_path2 = file_path + '-dp1'
                np.savez(file_path1, n_instances_portion, n_parti_top2_differ_one, vote_counts)
                np.savez(file_path2, n_instances_portion, n_parti_top2_differ_one, vote_counts_parties.reshape(-1,10))
                global_model = global_stu_net

            if args.alg == 'fedkt_fedavg' or args.alg == 'fedkt_fedprox':
                logger.info("Initializing nets")
                args.n_teacher_each_partition = 1
                args.is_local_split = 0
                nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties,
                                                                    args)
                global_model.to("cpu")
                for round in range(args.comm_round):
                    logger.info("in comm round:" + str(round))

                    global_para = get_trainable_parameters(global_model)

                    if args.is_same_initial:
                        for net_id, net in nets.items():
                            put_trainable_parameters(net, global_para)
                    else:
                        if round != 0:
                            for net_id, net in nets.items():
                                put_trainable_parameters(net, global_para)

                    # print("global paras:", list(global_model.parameters()))
                    # print("nets0 paras after:", list(nets[0].parameters()))
                    if args.alg == 'fedkt_fedavg':
                        local_train_net(nets, args, net_dataidx_map, X_train, y_train, X_test, y_test,
                                        remain_test_dl=remain_test_dl, local_split=False, device=device)
                    elif args.alg == 'fedkt_fedprox':
                        local_train_net_fedprox(nets, global_model, args, net_dataidx_map, X_train, y_train, X_test,
                                                y_test, remain_test_dl=remain_test_dl, local_split=False, device=device)
                        global_model.to('cpu')
                    # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

                    # update global model
                    total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
                    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)]
                    weights = [get_trainable_parameters(nets[i].cpu()) for i in range(args.n_parties)]
                    # print("avg freqs:", fed_avg_freqs)
                    # print("weights:", weights)
                    average_weight = sum(weights[i] * fed_avg_freqs[i] for i in range(args.n_parties))
                    # print("average_weight:", average_weight)
                    put_trainable_parameters(global_model, average_weight)

                    # global model accuracy
                    # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

                    # print("len test_ds_global:", len(test_ds_global))

                    logger.info('global n_training: %d' % len(train_dl_global))
                    logger.info('global n_test: %d' % len(remain_test_dl))

                    train_acc = compute_accuracy(global_model, train_dl_global)
                    test_acc, conf_matrix = compute_accuracy(global_model, remain_test_dl, get_confusion_matrix=True)

                    logger.info('>> Global Model Train accuracy: %f' % train_acc)
                    logger.info('>> Global Model Test accuracy: %f' % test_acc)

