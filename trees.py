from sklearn import tree
from sklearn import preprocessing
from sklearn import metrics
import logging
import numpy as np
import torch
import copy
import xgboost as xgb
from experiments import prepare_uniform_weights, normalize_weights
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_trees(max_tree_depth, n_parties, n_local_models, task_type, args):
    n_total_trees = n_parties * n_local_models
    trees = {tree_i: None for tree_i in range(n_total_trees)}

    for tree_i in range(n_total_trees):
        if args.model == 'tree' or args.model == 'gbdt_tree':
            if task_type == "binary_cls":
                trees[tree_i] = tree.DecisionTreeClassifier(max_depth = max_tree_depth)
            elif task_type == "reg":
                # trees[tree_i] = tree.DecisionTreeRegressor(max_depth = max_tree_depth)
                print("not supported yet")
                exit(1)
        elif args.model == 'random_forest':
            trees[tree_i] = RandomForestClassifier(max_depth = args.max_tree_depth, n_estimators=args.n_stu_trees)

        elif args.model == 'gbdt':
            trees[tree_i] = xgb.XGBClassifier(max_depth=args.max_tree_depth, n_estimators = args.n_stu_trees, learning_rate=args.lr, gamma=1, reg_lambda=1, tree_method='hist')
        elif args.model == 'gbdt_ntree':
            trees[tree_i] = xgb.XGBClassifier(max_depth=args.max_tree_depth, n_estimators = args.n_stu_trees, learning_rate=args.lr, gamma=1,
                                              reg_lambda=1, tree_method='hist')
    return list(trees.values())


def compute_tree_ensemble_accuracy(trees, X_test, y_test):
    y_pred_prob = np.zeros(len(list(y_test)))
    # print("local trees size:", len(trees))
    weights_list = prepare_uniform_weights(2, len(trees))
    weights_norm = normalize_weights(weights_list)
    # print("len of weights norm: ", weights_norm.size())
    # print("weights norm:", weights_norm)
    out_weight = None
    for tree_id, tree in enumerate(trees):
        pred = tree.predict_proba(X_test)
        # pred (n_samples, n_classes)
        if out_weight is None:
            out_weight = weights_norm[tree_id] * torch.tensor(pred,dtype=torch.float)
        else:
            out_weight += weights_norm[tree_id] * torch.tensor(pred,dtype=torch.float)

    _, pred_label = torch.max(out_weight.data, 1)
    # print("pred label:", pred_label)
    # print("y test:", y_test)
    # print("out weight:", out_weight)
    # print("len of out weight:", len(out_weight))
    correct_num = 0
    # print(pred_label == torch.BoolTensor(y_test))
    correct_num += (pred_label == torch.LongTensor(y_test)).sum().item()

    # print("correct num:", correct_num)
    # for i, pred_i in enumerate(out_weight):
    #     pred_class = np.argmax(pred_i)
    #     if pred_class == y_test[i]:
    #         correct_num += 1
    total = len(list(y_test))
    acc = correct_num / total
    return acc


# def compute_forest_accuracy(model, X_test, y_test):
#     model.predict(X_test)


def local_train_trees(trees, args, net_dataidx_map, X_train, y_train, X_test, y_test):
    n_local_models = args.n_teacher_each_partition
    # print("y_test:", y_test)
    # print("x_train:",X_train)
    for party_id in range(args.n_parties):
        dataidxs = net_dataidx_map[party_id]
        # print("dataidxs: ", dataidxs)
        logger.info("In party %d. n_training: %d" % (party_id, len(dataidxs)))
        dataidx_arr = np.array(dataidxs)
        np.random.shuffle(dataidx_arr)
        # partition the local data to n_local_models parts
        dataidx_each_model = np.array_split(dataidx_arr, n_local_models)
        # print("dataidx_each_model: ", dataidx_each_model)
        for tree_id in range(n_local_models):
            dataid = dataidx_each_model[tree_id]
            # print("dataid:", dataid)
            # logger.info("Training tree %s. n_training: %d" % (str(tree_id), len(dataid)))
            tree_id_global = tree_id + party_id * n_local_models
            # clf = tree.DecisionTreeClassifier(max_depth=args.max_depth)
            trees[tree_id_global].fit(X_train[dataid], y_train[dataid])
            acc = trees[tree_id_global].score(X_test, y_test)
            # logger.info('>> One tree acc: %f' % acc)

        ens_acc = compute_tree_ensemble_accuracy(trees[party_id * n_local_models : (party_id + 1) * n_local_models], X_test, y_test)
        logger.info("Local ensemble acc: %f" % ens_acc)
    return trees


def local_train_trees_in_a_party(trees, args, dataidxs, X_train, y_train, X_test, y_test):
    n_local_models = args.n_teacher_each_partition
    # print("y_test:", y_test)
    # print("x_train:",X_train)
    # print("dataidxs: ", dataidxs)
    # logger.info("In party %d. n_training: %d" % (party_id, len(dataidxs)))
    dataidx_arr = np.array(dataidxs)
    np.random.shuffle(dataidx_arr)
    # partition the local data to n_local_models parts
    dataidx_each_model = np.array_split(dataidx_arr, n_local_models)
    # print("dataidx_each_model: ", dataidx_each_model)
    for tree_id in range(n_local_models):
        dataid = dataidx_each_model[tree_id]
        # print("dataid:", dataid)
        # logger.info("Training tree %s. n_training: %d" % (str(tree_id), len(dataid)))
        # clf = tree.DecisionTreeClassifier(max_depth=args.max_depth)
        trees[tree_id].fit(X_train[dataid], y_train[dataid])
        acc = trees[tree_id].score(X_test, y_test)
        # logger.info('>> One tree acc: %f' % acc)

    ens_acc = compute_tree_ensemble_accuracy(trees, X_test, y_test)
    logger.info("Local ensemble acc: %f" % ens_acc)
    return trees

def central_train_trees_in_a_party(trees, args, X_train, y_train, X_test, y_test):
    n_local_models = args.n_teacher_each_partition
    # print("y_test:", y_test)
    # print("x_train:",X_train)
    # print("dataidxs: ", dataidxs)
    # logger.info("In party %d. n_training: %d" % (party_id, len(dataidxs)))
    dataidx_arr = np.arange(len(y_train))
    np.random.shuffle(dataidx_arr)
    # partition the local data to n_local_models parts
    dataidx_each_model = np.array_split(dataidx_arr, n_local_models)
    # print("dataidx_each_model: ", dataidx_each_model)
    for tree_id in range(n_local_models):
        dataid = dataidx_each_model[tree_id]
        # print("dataid:", dataid)
        # logger.info("Training tree %s. n_training: %d" % (str(tree_id), len(dataid)))
        # clf = tree.DecisionTreeClassifier(max_depth=args.max_depth)
        trees[tree_id].fit(X_train[dataid], y_train[dataid])
        acc = trees[tree_id].score(X_test, y_test)
        # logger.info('>> One tree acc: %f' % acc)

    ens_acc = compute_tree_ensemble_accuracy(trees, X_test, y_test)
    logger.info("Local ensemble acc: %f" % ens_acc)
    return trees


def train_a_student_tree(trees, public_data, public_data_label, n_classes, stu_model, gamma, filter_query, threshold=None, n_partition=None, apply_consistency=False, is_final_student=False):
    vote_counts = np.zeros((len(public_data_label), n_classes))
    for tree_id, tree in enumerate(trees):
        y_pred = tree.predict(public_data)
        y_prob = tree.predict_proba(public_data)
        # print("y_pred:", y_pred)
        if is_final_student and apply_consistency:
            if tree_id % n_partition == 0:
                votes_base = y_pred
                votes_flag = np.ones(len(y_pred), dtype=int)
            else:
                for i,y in enumerate(y_pred):
                    if votes_flag[i]:
                        if int(y) != votes_base[i]:
                            votes_flag[i] = 0
                    if (tree_id % n_partition) == (n_partition - 1) and votes_flag[i]:
                        vote_counts[i][int(y)] += n_partition
        else:
            for i, y in enumerate(y_pred):
                if threshold is not None:
                    if y_prob[i] >= threshold:
                        vote_counts[i][int(y)] += 1
                else:
                    vote_counts[i][int(y)] += 1
    vote_counts_origin = copy.deepcopy(vote_counts).astype("int")




    if gamma != 0:
        for i in range(vote_counts.shape[0]):
            vote_counts[i] += np.random.laplace(loc=0.0, scale=float(1.0 / gamma), size=vote_counts.shape[1])
    final_pred = np.argmax(vote_counts, axis=1)
    logger.info("Labeling acc %f" % ((final_pred == public_data_label).sum()/len(public_data_label)))

    if filter_query:
        confident_query_idx=[]
        for idx, row in enumerate(vote_counts_origin):
            top2_counts = row[np.argsort(row)[-2:]]
            if top2_counts[1] - top2_counts[0] > 2:
            # if top2_counts[1] > args.n_teacher_each_partition * args.query_filter_threshold:
                confident_query_idx.append(idx)

        print("len confident query idx:", len(confident_query_idx))
        logger.info("len confident query idx: %d" % len(confident_query_idx))
        # local_query_ds = data.Subset(public_ds, confident_query_idx)

        public_data = public_data[confident_query_idx]
        final_pred = [final_pred[i] for i in confident_query_idx]
        # query_data_size = int(len(y_test) * args.query_portion)

    stu_model.fit(public_data, final_pred)

    top1_class_counts = np.zeros(500)
    top2_class_counts = np.zeros(500)
    top_diff_counts = np.zeros(500)
    top2_counts_differ_one = 0
    for row in vote_counts_origin:
        # print(row)
        top2_counts = row[np.argsort(row)[-2:]]
        if top2_counts[1] - top2_counts[0] <= 1:
            top2_counts_differ_one+=1
        # print(top2_counts[1] - top2_counts[0])
        top_diff_counts[top2_counts[1] - top2_counts[0]] += 1
        top1_class_counts[top2_counts[1]] += 1
        top2_class_counts[top2_counts[0]] += 1

    return top2_counts_differ_one, vote_counts_origin


# should compare with randomly choose
def fedboost(trees, args, net_dataidx_map, X_train, y_train, X_test, y_test, task_type):
    for party_id in range(args.n_parties):
        dataidxs = net_dataidx_map[party_id]
        X_train_local = X_train[dataidxs]
        y_train_local = y_train[dataidxs]
        current_pred = np.zeros((len(y_train_local), 2))
        ensemble_tree_ids = np.zeros(args.n_ensemble_models, dtype=int)
        isselected = np.zeros(len(trees), dtype=int)
        for final_tree_id in range(args.n_ensemble_models):
            temp_loss = float("inf")
            temp_tree_id = -1
            for tree_id, tree in enumerate(trees):
                if isselected[tree_id] == 1:
                    continue
                if task_type == "binary_cls":
                    temp_pred = current_pred + tree.predict_proba(X_train_local)
                    current_pred_norm = preprocessing.normalize(temp_pred, axis=1, norm='l1')
                    current_loss = metrics.log_loss(y_train_local, current_pred_norm)
                    if tree_id in range(party_id*args.n_local_models, (party_id+1)*args.n_local_models):
                        current_loss += args.lambda_boost
                    if current_loss < temp_loss:
                        temp_loss = current_loss
                        temp_tree_id = tree_id
                elif task_type == "reg":
                    print("not supported yet!")
                    exit(1)
            ensemble_tree_ids[final_tree_id] = temp_tree_id
            current_pred += args.lr * trees[temp_tree_id].predict_proba(X_train_local)
            isselected[temp_tree_id] = 1
        ens_acc = compute_tree_ensemble_accuracy([trees[i] for i in ensemble_tree_ids], X_test, y_test)
        logger.info("In party %d" % party_id)
        logger.info("Selected trees %s" % " ".join(str(e) for e in ensemble_tree_ids))
        logger.info("Boost acc: %f" % ens_acc)


