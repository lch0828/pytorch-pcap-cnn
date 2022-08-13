from dataloader import PCAP
import torch.utils.data as data
import torch
import sys
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
#from trainer import fit
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from trainer import fit_model
import seaborn as sns
from IPython.display import display
from PIL import Image
from cnn import CNN

cuda = torch.cuda.is_available()


def load_dataset(path, train_split = 0.8):
    train_dataset = PCAP(path)
    train_set_size = int(len(train_dataset) * train_split)
    valid_set_size = len(train_dataset) - train_set_size
    train_dataset, test_dataset = data.random_split(train_dataset, [train_set_size, valid_set_size])
    return train_dataset, test_dataset

def train_nn(train_loader, test_loader, epoch = 50, learning_rate = 1e-3):
    n_epochs = epoch
    lr = learning_rate
    loss_func = nn.CrossEntropyLoss()
    model=CNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if cuda:
        model.cuda()

    training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(model, loss_func, optimizer, n_epochs, train_loader, test_loader)
    torch.save(model.state_dict(),'1DCNN.pth')
    return training_loss, training_accuracy, validation_loss, validation_accuracy
    
def result_chart(training_accuracy, validation_accuracy, n_epochs = 50):
    fig, ax = plt.subplots(figsize=(12,8))

    plt.xlim(-0.5,50)
    plt.ylim(75,100)
    major_ticks_top_x=np.linspace(0,50,6)
    minor_ticks_top_x=np.linspace(0,50,51)
    major_ticks_top_y=np.linspace(75,100,6)
    minor_ticks_top_y=np.linspace(75,100,26)

    plt.plot(range(n_epochs), training_accuracy, 'b-', label='Train')
    plt.plot(range(n_epochs), validation_accuracy, 'g-', label='Test')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Accuracy (%)', fontsize=15)
    plt.legend(loc=4, fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_xticks(major_ticks_top_x)
    ax.set_yticks(major_ticks_top_y)
    ax.set_xticks(minor_ticks_top_x,minor=True)
    ax.set_yticks(minor_ticks_top_y,minor=True)
    plt.grid(which="major",alpha=0.6)
    plt.grid(which="minor",alpha=0.3)
    plt.show()

def confusion_matrix(model, test_loader):
    print('Confusion matrix: ')
    cm = np.zeros((10, 10), dtype=np.float)
    pos = np.zeros((10, 2), dtype=np.float)
    neg = np.zeros((10, 2), dtype=np.float)
    model.eval()

    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        y_hat = torch.max(outputs.data, 1)[1]
        y = labels.long()
        for i in range(len(y)):
            cm[y[i], y_hat[i]] += 1
            if y[i] != y_hat[i]:
                pos[y_hat[i]][0]+=1
                neg[y[i]][0]+=1
            else:
                pos[y[i]][1] += 1

    labels = ['Chat', 'Email', 'File Transfer', 'Streaming', 'VoIP', 'VPN: Chat', 'VPN: File Transfer', 'VPN: Email', 'VPN: Streaming','VPN: VoIP']
    classnum = len(labels)
    sumrc=0
    sumpr=0
    sumf1=0
    for i in range(classnum):
        Rc = pos[i][1]/(pos[i][1]+pos[i][0])
        Pr = pos[i][1]/(pos[i][1]+neg[i][0])
        sumrc+=Rc
        sumpr+=Pr
        sumf1+=(2*Rc*Pr)/(Rc+Pr)
        print(f'{labels[i]}: Rc: {Rc}, Pr: {Pr}, F1: {(2*Rc*Pr)/(Rc+Pr)}')
    print(f'AveRC = {sumrc/classnum}, AvePr = {sumpr/classnum}, Avef1 = {sumf1/classnum}')
    normalised_cm = cm / cm.sum(axis=1, keepdims=True)
    normalised_cm = np.nan_to_num(normalised_cm)
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(
        data=normalised_cm, cmap='Blues', 
        xticklabels=labels, yticklabels=labels,
        annot=True, ax=ax, fmt='.1%', vmin=0, vmax=1
    )
    sns.set(font_scale=1.3)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, .2, .4, .6, .8, 1])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.set_xlabel('Predict labels', fontsize=20)
    ax.set_ylabel('True labels', fontsize=20)
    fig.show()

def plot_dataset(dataset):
    tmp=[]
    height = 0
    count = 0
    index = 0
    while index < (15000*10):
        tmp.append(dataset[index][0].cpu().detach().numpy())
        height += 1
        index += 1
        if(height == 50):
            tmp = np.array(tmp)
            tmp = np.squeeze(tmp)
            tmp = (tmp*255).astype(np.uint8)
            img = Image.fromarray(tmp, mode='L')
            print(dataset[index][1]) #label
            display(img)
            tmp = []
            height = 0
            count += 1
            index = 15000*count
