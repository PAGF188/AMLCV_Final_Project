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
        running_corrects = 0

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
    
    # statistics
    running_loss = 0.0
    running_corrects = 0
    total_values = np.array([])
    total_preds = np.array([])
    total_labels = np.array([])

    for inputs, labels in testloader:
        total_labels = np.concatenate([total_labels, labels[0]])
        inputs = inputs.to(DEVICE)
        labels = labels[0].to(DEVICE)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            values, preds = torch.max(outputs, 1)
            total_values = np.concatenate([total_values, values.cpu()])
            total_preds = np.concatenate([total_preds, preds.cpu()])
        
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    time_elapsed = time.time() - since

    epoch_loss = running_loss / len(testloader.dataset)
    epoch_acc = running_corrects.double() / len(testloader.dataset)
    mc = confusion_matrix(total_labels, total_preds)
    fpr, tpr, _ = roc_curve(total_labels, total_values)
    
    # TO SHOW CONFUSION MATRIX AS PLOT
    g = sns.heatmap(mc/np.sum(mc), cmap="Reds", annot=True, fmt = '.2%', square=1,   linewidth=2.)
    g.set_xticklabels(['0 (w)', '1 (m)'])
    g.set_yticklabels(['0 (w)', '1 (m)'])
    figure = g.get_figure()    
    figure.savefig(os.path.join(MODEL_SAVE_DIR, T1_FOLDER) + 'cm.png', dpi=400)

    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, pos_label=0).plot()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, T1_FOLDER) + 'roc_0_w.png', dpi=400)

    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, pos_label=1).plot()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, T1_FOLDER) + 'roc_1_m.png', dpi=400)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format('test', epoch_loss, epoch_acc))
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


