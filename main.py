import torch
from cnn import CNN
from util import load_dataset, result_chart, confusion_matrix, train_nn
cuda = torch.cuda.is_available()


if __name__ == '__main__':
    train_dataset, test_dataset = load_dataset('./pre') #path to ./pre folder

    epoch = 50
    batch = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False, **kwargs)
    training_loss, training_accuracy, validation_loss, validation_accuracy = train_nn(train_loader, test_loader, epoch = 50)
    result_chart(training_accuracy, validation_accuracy, epoch)

    #load model
    model=CNN()
    model.load_state_dict(torch.load('1DCNN.pth'))
    if cuda:
        model.cuda()
    model.eval()
    confusion_matrix(model, test_loader)

    #plot_dataset(train_dataset)