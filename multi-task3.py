"""
Multi-task 3 : Clasification and eyes, mouse and nose landmark location. 
Train and evaluate a baseline network that, given a face image, predicts
both the gender and the eyes and nose location.

Author: Pablo Garcia Fernnadez
"""

from config import *
from utils.celebamini import CelebAMini
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.visualization import plot_dataset_examples
from models.task3_densenet import *
import pdb

def multi_task3():
    # 1) CREATE DATALOADERS
    print("1) Creating DATALOADERS")
    basic_transforms =  transforms.Compose([
        transforms.Resize(T2_SIZE), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scale_factor = np.array(T2_SIZE) / np.array(INPUT_SIZE_IMAGE)  

    train_set = CelebAMini(os.getcwd(), transform=basic_transforms, split='train', s_f = scale_factor)
    test_set = CelebAMini(os.getcwd(), transform=basic_transforms, split='test', s_f = scale_factor)

    train_loader = DataLoader(train_set, batch_size=T2_BATCH, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=T2_BATCH, shuffle=True)
    dataloaders_dict = {}
    dataloaders_dict['train'] = train_loader
    dataloaders_dict['test'] = test_loader

    # 2) CREATE MODEL
    print("1) Creating DENSENET MODEL")
    model = MULTITASK()
    model = model.to(DEVICE)
    print(model)

    if os.path.isfile(os.path.join(MODEL_SAVE_DIR, T3_NAME)):
        print(f"MODEL is already trained!! ")
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, T3_NAME)))
    else:
        print(f"Training MODEL...")
        optimizer = torch.optim.Adam(model.parameters(), lr=T3_LR)
        criterion_clas = torch.nn.CrossEntropyLoss()
        criterion_reg = torch.nn.MSELoss()
        model = train_model(model, dataloaders_dict, criterion_clas, criterion_reg, optimizer, num_epochs=T3_EPOCHS)
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, T3_NAME))

    # 3) TEST RESULTS
    eval_model(model, dataloaders_dict['test'])
    #visualice(model, dataloaders_dict['test'])



if __name__ == "__main__":
    print("MULTI TASK3: CLASSIFICATION AND REGRESSION !!! \n")
    multi_task3()