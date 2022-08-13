import torch
from cnn import CNN
from util import load_dataset, result_chart, confusion_matrix, train_nn
cuda = torch.cuda.is_available()


if __name__ == '__main__':
    train_dataset, test_dataset = load_dataset('./pre')

    batch = 128
    kwargs = {'num_workers':1, 'pin_memory': True} if cuda else {}
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False, **kwargs)

    # load model
    model = CNN()
    model.load_state_dict(torch.load('1DCNN.pth'))
    if cuda:
        model.cuda()
    model.eval()
    confusion_matrix(model, test_loader)

