"""
Single-task 1 : Gender classification. 
Train and evaluate a baseline network that, given a face image, predicts the gender of the person.

Author: Pablo Garcia Fernnadez
"""

from config import *
from utils.celebamini import CelebAMini
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.visualization import plot_dataset_examples
from models.task1_densenet import *

def single_task1():
    # 1) CREATE DATALOADERS
    print("1) Creating DATALOADERS")
    basic_transforms =  transforms.Compose([
        transforms.Resize(T1_SIZE), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = CelebAMini(os.getcwd(), transform=basic_transforms, split='train')
    test_set = CelebAMini(os.getcwd(), transform=basic_transforms, split='test')
    #plot_dataset_examples(train_set)
    #plot_dataset_examples(test_set)

    train_loader = DataLoader(train_set, batch_size=T1_BATCH, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=T1_BATCH, shuffle=True)
    dataloaders_dict = {}
    dataloaders_dict['train'] = train_loader
    dataloaders_dict['test'] = test_loader

    # 2) CREATE MODEL
    print("1) Creating DENSENET MODEL")
    model = denseNet121_pretrained()
    model = model.to(DEVICE)
    print(model)

    if os.path.isfile(os.path.join(MODEL_SAVE_DIR, T1_NAME)):
        print(f"MODEL is already trained!! ")
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, T1_NAME)))
    else:
        print(f"Training MODEL...")
        optimizer = torch.optim.Adam(model.parameters(), lr=T1_LR)
        criterion = torch.nn.CrossEntropyLoss()
        model = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=T1_EPOCHS)
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, T1_NAME))

    # 3) TEST RESULTS
    eval_model(model, dataloaders_dict['test'], torch.nn.CrossEntropyLoss())




if __name__ == "__main__":
    print("SINGLE TASK1: GENDER CLASSIFICATION!!! \n")
    single_task1()