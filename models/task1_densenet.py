import torchvision.models as models
from torch import nn
from config import *
import time
import pdb

def denseNet121_pretrained():
    # Pre-trained on ImageNET
    model = models.densenet121(pretrained=True)
    
    # DenseNet121 has only one linear layer in the clasiffier
    # Change final linear layer to 2 outpus
    model.classifier = nn.Linear(1024, T1_CLASES)
    return model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    since = time.time()

    for epoch in range(num_epochs):
        init_epoch_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(DEVICE)
            labels = labels[0].to(DEVICE)   # Get only the gender information
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))
        print(f"Time per epoch: {time.time() - init_epoch_time}")
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model


def eval_model(model, testloader, criterion):
    since = time.time()
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in testloader:
        inputs = inputs.to(DEVICE)
        labels = labels[0].to(DEVICE)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(testloader.dataset)
    epoch_acc = running_corrects.double() / len(testloader.dataset)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', epoch_loss, epoch_acc))
    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))