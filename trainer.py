import torch
import numpy as np

def fit_model(model, loss_func, optimizer, num_epochs, train_loader, test_loader):
    # Traning the Model
    #history-like list for store loss & acc value
    model.train()
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []

    for epoch in range(num_epochs):
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            train_loss = loss_func(outputs, labels)
            train_loss.backward()
            optimizer.step()
            predicted = torch.max(outputs.data, 1)[1]
            total_train += len(labels)
            correct_train += (predicted == labels).float().sum()

        #store val_acc / epoch
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)

        # store loss / epoch
        training_loss.append(train_loss.data)

        #evaluate model & store loss & acc / epoch
        correct_test = 0
        total_test = 0

        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            val_loss = loss_func(outputs, labels)
            predicted = torch.max(outputs.data, 1)[1]
            total_test += len(labels)
            correct_test += (predicted == labels).float().sum()

        val_accuracy = 100 * correct_test / float(total_test)
        validation_accuracy.append(val_accuracy)

        validation_loss.append(val_loss.data)
        print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}% Val_Loss: {} Val_accuracy: {:.6f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
    
    return training_loss, training_accuracy, validation_loss, validation_accuracy