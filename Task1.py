"""
Environments:   Python = 3.7
"""
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA


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


if __name__ == '__main__':
    data_train, labels_train, data_test, labels_test = creatData()
    data_train = data_train / 255
    data_test = data_test / 255

    # Apply PCA to reduce the original input features
    # into new feature vectors with different amount of information kept, e.g. 30%, 50%, 70%, 100%
    pca100 = PCA(n_components=int(32*32*3))
    pca100.fit_transform(data_train)
    pca100.transform(data_test)
    print('Information kept: ', sum(pca100.explained_variance_ratio_) * 100, '%')
    print('Noise variance: ', pca100.noise_variance_)

    pca70 = PCA(n_components=14)
    pca70.fit_transform(data_train)
    pca70.transform(data_test)
    print('Information kept: ', sum(pca70.explained_variance_ratio_) * 100, '%')
    print('Noise variance: ', pca70.noise_variance_)

    pca50 = PCA(n_components=4)
    pca50.fit_transform(data_train)
    pca50.transform(data_test)
    print('Information kept: ', sum(pca50.explained_variance_ratio_) * 100, '%')
    print('Noise variance: ', pca50.noise_variance_)

    pca30 = PCA(n_components=1)
    pca30.fit_transform(data_train)
    pca30.transform(data_test)
    print('Information kept: ', sum(pca30.explained_variance_ratio_) * 100, '%')
    print('Noise variance: ', pca30.noise_variance_)


