"""
Environments:   Python = 3.7
"""
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import *


# define a CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # original image: 32*32*3
        # New size of the image = (32 - 3 + 2*1)/1 + 1 = 32
        # Convolution layer 1 (32*32*32)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # Convolution layer 2 (32*32*64)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # Convolution layer 3 (32*32*128)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # Maximum pooling layer
        # filter size = 2, stride = 2 (16*16*128)
        self.pool = nn.MaxPool2d(2, 2)

        # LINEAR LAYER 1 (500)
        self.fc1 = nn.Linear(128 * 4 * 4, 500)
        # LINEAR LAYER 1 (10)
        self.fc2 = nn.Linear(500, 10)
        # dropout (p=0.3): Prevent overfitting
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flatten image input
        # print(x.shape)
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def train_model():
    global data
    # Use the cross entropy loss function
    criterion = nn.CrossEntropyLoss()
    # Using stochastic gradient descent, the learning rate is 0.01
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # times of the model training
    n_epochs = 20

    # track change in validation loss
    valid_loss_min = np.Inf
    for epoch in range(1, n_epochs + 1):

        # keep tracks of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # model training
        model.train()
        for data, target in train_loader:
            # move tensors to gpu if cuda is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass:compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass:compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step(parameters updata)
            optimizer.step()
            # updata training loss
            train_loss += loss.item() * data.size(0)

        # model validation
        model.eval()
        for data, target in valid_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        # calculate the mean loss
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        print('Epoch:{} \tTraining loss:{} \tValidation loss:{}'.format(
            epoch, train_loss, valid_loss
        ))

        # save the model if the validation set loss is reduced
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({} --> {}). Saving model ...'.format(
                valid_loss_min, valid_loss
            ))

            # save model
            torch.save(model.state_dict(), 'model_50000.pt')
            # record and update the min loss
            valid_loss_min = valid_loss


def test_model():
    global data
    # load model and test model
    model.load_state_dict(torch.load('model_50000.pt', map_location=torch.device('cpu')))

    # track test loss
    model.eval()

    correct_test = 0
    total_test = 0
    label_test = []
    label_pred_test = []
    predicted_list_test = []
    with torch.no_grad():
        for (data, target) in test_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(data)
            # get the predicted labels
            _, predicted = torch.max(outputs.data, 1)
            predicted_list_test.append(predicted)
            # count total test images
            total_test += target.size(0)
            # count the images with the correct prediction
            correct_test += (predicted == target).sum().item()

            # labels
            label_test.append(np.squeeze(target.numpy()) if not train_on_gpu else np.squeeze(
                target.cpu().numpy()))
            # predicted labels
            label_pred_test.append(np.squeeze(predicted.numpy()) if not train_on_gpu else np.squeeze(
                predicted.cpu().numpy()))

    # print("predicted_list_test: ", predicted_list_test)
    print('test set accuracy: %d %%' % (100 * correct_test / total_test))

    # transform tensors to a vector
    predicted_label_vector = []
    label_test_vector = []
    for i in range(len(label_pred_test)):
        for j in range(4):
            predicted_label_vector.append(label_pred_test[i][j])
            label_test_vector.append(label_test[i][j])

    # print("predicted number in test set", predicted_label)
    # print("labels in test set", label_test)

    # find f1_scores, precision_score and recall_score on test_set
    f1_test = f1_score(label_test_vector, predicted_label_vector, average=None)
    p_test = precision_score(label_test_vector, predicted_label_vector, average='macro')
    r_test = recall_score(label_test_vector, predicted_label_vector, average='macro')
    print("test set - f1 scores: ", f1_test)
    print("test set - precision score: ", p_test)
    print("test set - recall score: ", r_test)

    end_time = time()
    run_time = end_time - begin_time
    print('CNN time：', run_time)

    # calculate the accuracy on the train set
    correct_train = 0
    total_train = 0
    label_train = []
    label_pred_train = []
    predicted_list_train = []
    with torch.no_grad():
        for (data, target) in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(data)
            # get the predicted labels
            _, predicted_train = torch.max(outputs.data, 1)
            predicted_list_train.append(predicted_train)
            # count total test images
            total_train += target.size(0)
            # count the images with the correct prediction
            correct_train += (predicted_train == target).sum().item()

            # labels
            label_train.append(np.squeeze(target.numpy()) if not train_on_gpu else np.squeeze(
                target.cpu().numpy()))
            # predicted labels
            label_pred_train.append(np.squeeze(predicted_train.numpy()) if not train_on_gpu else np.squeeze(
                predicted_train.cpu().numpy()))

    # print("predicted_list_train: ", predicted_list_train)
    print('train set accuracy: %d %%' % (100 * correct_train / total_train))

    # transform tensors to a vector
    predicted_label_train_vector = []
    label_train_vector = []
    for i in range(len(label_pred_train)):
        for j in range(4):
            predicted_label_train_vector.append(label_pred_train[i][j])
            label_train_vector.append(label_train[i][j])

    # print("predicted_label_train: ", predicted_label_train)
    # print("label_train: ", label_train)

    # find f1_scores, precision_score and recall_score on train_set
    f1_train = f1_score(label_train_vector, predicted_label_train_vector, average=None)
    p_train = precision_score(label_train_vector, predicted_label_train_vector, average='macro')
    r_train = recall_score(label_train_vector, predicted_label_train_vector, average='macro')
    print("train set - f1 scores: ", f1_train)
    print("train set - precision score: ", p_train)
    print("train set - recall score: ", r_train)


if __name__ == '__main__':
    # Check whether the GPU is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA IS NOT AVAILABLE!')
    else:
        print('CUDA IS AVAILABLE!')

    # load data
    num_workers = 0
    # Load 16 images per batch
    batch_size = 16
    # percentage of training set to use as validation
    valid_size = 0.2

    # Convert the data to torch.FloatTensor and standardize it
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Select data for training set and test set
    train_data = datasets.CIFAR10(
        'data', train=True,
        download=True, transform=transform
    )
    test_data = datasets.CIFAR10(
        'data', train=True, download=True, transform=transform
    )

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders(combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers)

    # 10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # create a CNN
    model = Net()
    print(model)

    if train_on_gpu:
        model.cuda()

    begin_time = time()

    train_model()
    test_model()
