import torchvision.models as models
from torch import nn
from config import *
import time
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay
import seaborn as sns
import pdb
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def task2_denseNet121_pretrained():
    # Pre-trained on ImageNET
    model = models.densenet121(pretrained=True)
    
    # DenseNet121 has only one linear layer in the clasiffier
    model.classifier = nn.Linear(1024, T2_CLASES)
    return model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    since = time.time()

    for epoch in range(num_epochs):
        init_epoch_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode
        running_loss = 0.0

        # Iterate over data.
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(DEVICE)
            labels = labels[1].to(DEVICE).float()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            with torch.set_grad_enabled(True):
                preds = model(inputs)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        print('{} Loss: {:.4f}'.format('train', epoch_loss))
        print(f"Time per epoch: {time.time() - init_epoch_time}")
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model


def eval_model(model, testloader):
    criterion = torch.nn.MSELoss()   # El Mear error es la raiz cuadrada del MSE
    since = time.time()
    model.eval()   # Set model to evaluate mode
    
    # statistics
    running_MSE = 0.0

    for inputs, labels in testloader:
        inputs = inputs.to(DEVICE)
        labels = labels[1].to(DEVICE).float()

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # statistics
        running_MSE += loss.item() * inputs.size(0)
    
    time_elapsed = time.time() - since
    MSE = running_MSE / len(testloader.dataset)
    RMSE = np.sqrt(MSE)
    

    print('{} RMSE: {:.4f}'.format('test', RMSE))
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def visualice(model, testloader):
    model.eval() 
    i = 0
    for inputs, labels in testloader:
        inputs = inputs.to(DEVICE)
        labels = labels[1].cpu().numpy()
        preds = model(inputs)
        preds = preds.cpu().detach().numpy().astype(int)

        print(labels[0])
        print(preds[0])

        # 1 EXAMPLE
        preds = preds[0]
        labels = labels[0]

        lm = preds.reshape((-1,2))
        #lm = labels.reshape((-1,2))   # Usar este para ver las autenticas labels

        img = inputs[0]
        img = img.permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.scatter(lm[0,0], lm[0,1], marker="<")
        plt.scatter(lm[1,0], lm[1,1], marker=">")
        plt.scatter(lm[2,0], lm[2,1], marker="^")
        plt.scatter(lm[3,0], lm[3,1], marker="3")
        plt.scatter(lm[4,0], lm[4,1], marker="4")
        plt.show()
        plt.clf()
        if i==10:
            break
        i+=1


