"""
Environments:   Python = 3.7
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
from sklearn.exceptions import ConvergenceWarning
from time import *
from sklearn.neural_network import MLPClassifier


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# load data
def creatData():
    sum_data = []
    sum_labels = []
    for i in range(1, 6):
        batch_path = './data/cifar-10-batches-py/data_batch_%d' % i
        batch_dict = unpickle(batch_path)
        train_batch = batch_dict[b'data'].astype('float')
        train_labels = np.array(batch_dict[b'labels'])
        sum_data.append(train_batch)
        sum_labels.append(train_labels)

    train_data = np.concatenate(sum_data)
    train_labels = np.concatenate(sum_labels)

    test_path = os.path.join('./data/cifar-10-batches-py', 'test_batch')
    test_dict = unpickle(test_path)
    test_data = test_dict[b'data'].astype('float')
    test_labels = np.array(test_dict[b'labels'])

    return train_data, train_labels, test_data, test_labels


def classification_MLP(data_train_input, labels_train_input, data_test_input, labels_test_input):
    mlp = MLPClassifier()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(data_train_input, labels_train_input)
    print("Input test set score: %f" % mlp.score(data_test_input, labels_test_input))

    test_score = mlp.score(data_test_input, labels_test_input)
    label_predict = mlp.predict(data_test_input)

    return test_score, label_predict


def show_numbers_in_eachCLass(te_feats, te_label, tr_feats, tr_label):
    index = []
    index2 = []
    for i in range(0, 10):
        index.append(np.nonzero(labels_test_small == i)[0])
        index2.append(np.nonzero(labels_train_small == i)[0])

        print("test size of class", i, " = ", len(index[i]))
        print("train size of class", i, " = ", len(index2[i]))


def cross_validation(trainval_feats, trainval_label):
    all_scores = []
    all_mean_scores = []

    # for k in features:
    for k in range(29, 100, 10):
        print("k = ", k)
        pcadata = PCA(k / 100)
        tr_feats_afterPCA = pcadata.fit_transform(trainval_feats)
        mlp = MLPClassifier()
        folds = 10
        scores = cross_val_score(mlp, tr_feats_afterPCA, trainval_label, cv=folds, scoring='accuracy')
        mean_score = np.mean(scores)

        all_scores.append(scores)
        all_mean_scores.append(mean_score)

        print("information kept: ", sum(pcadata.explained_variance_ratio_), " score = ", scores)
        print("information kept: ", sum(pcadata.explained_variance_ratio_), " mean score = ", mean_score)

    print("all scores: ", all_scores)
    print("all mean scores: ", all_mean_scores)


def eval_mymlp(te_feats, te_label, tr_feats, tr_label):
    all_f1 = []
    all_accuracy = []
    x_axis = []
    for k in range(29, 100, 10):
        print("k = ", k)
        pcadata = PCA(k / 100)
        tr_feats_afterPCA = pcadata.fit_transform(tr_feats)
        te_feats_afterPCA = pcadata.transform(te_feats)
        x_axis.append(sum(pcadata.explained_variance_ratio_))
        accuracy, label_pred = classification_MLP(tr_feats_afterPCA, tr_label, te_feats_afterPCA, te_label)
        # print("predict labels: ", label_pred)
        f1 = f1_score(te_label, label_pred, average=None)

        all_f1.append(f1)
        all_accuracy.append(accuracy)

    all_f1_trans = np.transpose([all_f1])
    # print("all f1: ", all_f1)
    # print("all f1 trans: ", all_f1_trans)
    plt.figure(1)
    for i in range(10):
        plt.plot(x_axis, all_f1_trans[i])
        plt.xlabel('information_kept')
        plt.ylabel('f1 scores')

    plt.legend(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
               loc='lower left')
    # plt.show()
    plt.savefig('./image/f1_information kept')

    # accuracy of the model
    plt.figure(2)
    plt.plot(x_axis, all_accuracy)
    plt.xlabel('feature dimension')
    plt.ylabel('accuracy')
    # plt.show()
    plt.savefig('./image/accuracy_information kept')


def train_model_hidden_layer_cv(trainval_feats, trainval_label):
    all_scores = []
    all_mean_scores = []
    all_layers = []

    for k in range(500, 3000, 500):
        print("hidden_layers = ", k)
        all_layers.append(k)

        mlp_hidden_layer_cv = MLPClassifier(hidden_layer_sizes=(int(k),), max_iter=50,
                                            alpha=1e-4, solver='sgd', verbose=1, tol=1e-4, random_state=1,
                                            learning_rate_init=1e-3)
        folds = 10
        scores = cross_val_score(mlp_hidden_layer_cv, trainval_feats, trainval_label, cv=folds, scoring='accuracy')
        mean_score = np.mean(scores)

        all_scores.append(scores)
        all_mean_scores.append(mean_score)

    # Find the largest mean score and return it
    max_index = all_mean_scores.index(max(all_mean_scores))
    print("best layer: ", all_layers[max_index])

    return all_layers[max_index]


def train_model_max_iter_cv(trainval_feats, trainval_label):
    all_scores = []
    all_mean_scores = []
    all_iters = []

    for k in range(50, 400, 50):
        print("max iter = ", k)
        all_iters.append(k)

        mlp_max_iter_cv = MLPClassifier(hidden_layer_sizes=(500,), max_iter=int(k),
                                        alpha=1e-4, solver='sgd', verbose=1, tol=1e-4, random_state=1,
                                        learning_rate_init=1e-3)
        folds = 10
        scores = cross_val_score(mlp_max_iter_cv, trainval_feats, trainval_label, cv=folds, scoring='accuracy')
        mean_score = np.mean(scores)

        all_scores.append(scores)
        all_mean_scores.append(mean_score)

    # Find the largest mean score and return it
    max_index = all_mean_scores.index(max(all_mean_scores))
    print("best max iter: ", all_iters[max_index])

    return all_iters[max_index]


def train_model_lr_cv(trainval_feats, trainval_label):
    all_scores = []
    all_mean_scores = []
    all_lr = []
    # 1e-3 = 0.001
    for k in range(1, 4):
        print("learning rate = ", k / 1000)
        all_lr.append(k / 1000)

        mlp_lr_cv = MLPClassifier(hidden_layer_sizes=(500,), max_iter=50,
                                  alpha=1e-4, solver='sgd', verbose=1, tol=1e-4, random_state=1,
                                  learning_rate_init=(k / 1000))
        folds = 10
        scores = cross_val_score(mlp_lr_cv, trainval_feats, trainval_label, cv=folds, scoring='accuracy')
        mean_score = np.mean(scores)

        all_scores.append(scores)
        all_mean_scores.append(mean_score)

    # Find the largest mean score and return it
    max_index = all_mean_scores.index(max(all_mean_scores))
    print("best learning rate: ", all_lr[max_index])

    return all_lr[max_index]


def train_model_hidden_layer(te_feats, te_label, tr_feats, tr_label):
    all_scores = []
    all_layers = []
    all_f1 = []

    for k in range(500, 3000, 500):
        print("hidden_layers = ", k)
        all_layers.append(k)

        mlp_hidden_layer = MLPClassifier(hidden_layer_sizes=(int(k),), max_iter=50,
                                         alpha=1e-4, solver='sgd', verbose=1, tol=1e-4, random_state=1,
                                         learning_rate_init=1e-3)

        mlp_hidden_layer.fit(tr_feats, tr_label)
        label_pred = mlp_hidden_layer.predict(te_feats)

        f1 = f1_score(te_label, label_pred, average=None)
        all_f1.append(f1)

        score = mlp_hidden_layer.score(te_feats, te_label)
        all_scores.append(score)

    print(all_f1)
    print(all_scores)

    all_f1_trans = np.transpose([all_f1])
    # print("all f1: ", all_f1)
    # print("all f1 trans: ", all_f1_trans)

    plt.figure(3)
    for i in range(10):
        plt.plot(all_layers, all_f1_trans[i])
        plt.xlabel('hidden layers')
        plt.ylabel('f1 scores')

    plt.legend(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
               loc='lower left')
    # plt.show()
    plt.savefig('./image/f1_hidden layers')

    # accuracy of the model
    plt.figure(4)
    plt.plot(all_layers, all_scores)
    plt.xlabel('hidden layers')
    plt.ylabel('accuracy')
    # plt.show()
    plt.savefig('./image/accuracy_hidden layers')


def train_model_max_iter(te_feats, te_label, tr_feats, tr_label):
    all_scores = []
    all_f1 = []
    all_iters = []

    for k in range(50, 400, 50):
        print("max iter = ", k)
        all_iters.append(k)

        mlp_max_iter = MLPClassifier(hidden_layer_sizes=(500,), max_iter=int(k),
                                     alpha=1e-4, solver='sgd', verbose=1, tol=1e-4, random_state=1,
                                     learning_rate_init=1e-3)
        mlp_max_iter.fit(tr_feats, tr_label)
        label_pred = mlp_max_iter.predict(te_feats)

        f1 = f1_score(te_label, label_pred, average=None)
        all_f1.append(f1)

        score = mlp_max_iter.score(te_feats, te_label)
        all_scores.append(score)

    print(all_f1)
    print(all_scores)

    all_f1_trans = np.transpose([all_f1])
    # print("all f1: ", all_f1)
    # print("all f1 trans: ", all_f1_trans)

    plt.figure(5)
    for i in range(10):
        plt.plot(all_iters, all_f1_trans[i])
        plt.xlabel('max iter')
        plt.ylabel('f1 scores')

    plt.legend(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
               loc='lower left')
    # plt.show()
    plt.savefig('./image/f1_max iter')

    # accuracy of the model
    plt.figure(6)
    plt.plot(all_iters, all_scores)
    plt.xlabel('max iter')
    plt.ylabel('accuracy')
    # plt.show()
    plt.savefig('./image/accuracy_max iter')


def train_model_lr(te_feats, te_label, tr_feats, tr_label):
    all_scores = []
    all_f1 = []
    all_lr = []
    # 1e-3 = 0.001
    for k in range(1, 4):
        print("learning rate = ", k / 1000)
        all_lr.append(k / 1000)

        mlp_lr = MLPClassifier(hidden_layer_sizes=(500,), max_iter=50,
                               alpha=1e-4, solver='sgd', verbose=1, tol=1e-4, random_state=1,
                               learning_rate_init=(k / 1000))
        mlp_lr.fit(tr_feats, tr_label)
        label_pred = mlp_lr.predict(te_feats)

        f1 = f1_score(te_label, label_pred, average=None)
        all_f1.append(f1)

        score = mlp_lr.score(te_feats, te_label)
        all_scores.append(score)

    print(all_f1)
    print(all_scores)

    all_f1_trans = np.transpose([all_f1])
    # print("all f1: ", all_f1)
    # print("all f1 trans: ", all_f1_trans)

    plt.figure(7)
    for i in range(10):
        plt.plot(all_lr, all_f1_trans[i])
        plt.xlabel('learning rate')
        plt.ylabel('f1 scores')

    plt.legend(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
               loc='lower left')
    # plt.show()
    plt.savefig('./image/f1_learning rate')

    # accuracy of the model
    plt.figure(8)
    plt.plot(all_lr, all_scores)
    plt.xlabel('learning rate')
    plt.ylabel('accuracy')
    # plt.show()
    plt.savefig('./image/accuracy_learning rate')





if __name__ == '__main__':
    data_train, labels_train, data_test, labels_test = creatData()

    # Switch to float for more precision
    data_train = data_train / 255.0
    data_test = data_test / 255.0

    data_train_small = data_train[0:5000]
    labels_train_small = labels_train[0:5000]

    # Use 1000 images as a test set
    data_test_small = data_test[0:1000]
    labels_test_small = labels_test[0:1000]
    print('data load finish')

    # Print the number of each class in train set and test set
    show_numbers_in_eachCLass(data_test_small, labels_test_small, data_train_small, labels_train_small)

    #################### 2.1: 10 fold cross validation
    cross_validation(data_train_small, labels_train_small)

    ##################### 2.2: f1 & accuracy
    eval_mymlp(data_test_small, labels_test_small, data_train_small, labels_train_small)

    ##################### 2.3: Train MLP with different hyper-parameters: max iter, learning rate, hidden layer
    # initial: max iter = 50, hidden layer = 500 learning rate = 1e-3 (0.001)
    best_layer_cv = train_model_hidden_layer_cv(data_train_small, labels_train_small)
    best_iter_cv = train_model_max_iter_cv(data_train_small, labels_train_small)
    best_lr_cv = train_model_lr_cv(data_train_small, labels_train_small)

    ##################### 2.4: Train MLP with different hyper-parameters: max iter, learning rate, hidden layer
    train_model_hidden_layer(data_test_small, labels_test_small, data_train_small, labels_train_small)
    train_model_max_iter(data_test_small, labels_test_small, data_train_small, labels_train_small)
    train_model_lr(data_test_small, labels_test_small, data_train_small, labels_train_small)

    ##################### 4: performance
    begin_time = time()
    # Use optimal parameters
    # train size: 50000   test size: 10000
    mlp_task4 = MLPClassifier(hidden_layer_sizes=(1500,), max_iter=100,
                              alpha=1e-4, solver='sgd', verbose=1, tol=1e-4, random_state=1,
                              learning_rate_init=1e-3)

    mlp_task4.fit(data_train, labels_train)

    # calculate the accuracy on the test set
    predicted = mlp_task4.predict(data_test)

    score = accuracy_score(labels_test, predicted)
    print('test set accuracy: %d %%' % (100 * score))

    f1 = f1_score(labels_test, predicted, average=None)
    p_test = precision_score(labels_test, predicted, average='macro')
    r_test = recall_score(labels_test, predicted, average='macro')

    print("test set - f1 scores: ", f1)
    print("test set - precision score: ", p_test)
    print("test set - recall score: ", r_test)

    end_time = time()
    run_time = end_time - begin_time
    print('MLP time：', run_time)

    # calculate the accuracy on the train set
    predicted_train = mlp_task4.predict(data_train)

    score = accuracy_score(labels_train, predicted_train)
    print('train set accuracy: %d %%' % (100 * score))

    f1 = f1_score(labels_train, predicted_train, average=None)
    p_test = precision_score(labels_train, predicted_train, average='macro')
    r_test = recall_score(labels_train, predicted_train, average='macro')

    print("train set - f1 scores: ", f1)
    print("train set - precision score: ", p_test)
    print("train set - recall score: ", r_test)


