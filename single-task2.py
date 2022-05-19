"""
Single-task 2 : Eyes, mouse and nose landmark location. 
Train and evaluate a baseline network that, given a face image, predicts  the (x,y) coordinates, 
in the image, of the left eye, the right eye, mouse left, mouse right and the nose.

Author: Pablo Garcia Fernnadez
"""

from config import *
from utils.celebamini import CelebAMini
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.visualization import plot_dataset_examples
from models.task2_densenet import *

def single_task2():
    # 1) CREATE DATALOADERS
    print("1) Creating DATALOADERS")
    basic_transforms =  transforms.Compose([
        transforms.Resize(T2_SIZE), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = CelebAMini(os.getcwd(), transform=basic_transforms, split='train')
    test_set = CelebAMini(os.getcwd(), transform=basic_transforms, split='test')
    #plot_dataset_examples(train_set)
    #plot_dataset_examples(test_set)

    train_loader = DataLoader(train_set, batch_size=T2_BATCH, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=T2_BATCH, shuffle=True)
    dataloaders_dict = {}
    dataloaders_dict['train'] = train_loader
    dataloaders_dict['test'] = test_loader

    # 2) CREATE MODEL
    print("1) Creating DENSENET MODEL")
    model = task2_denseNet121_pretrained()
    model = model.to(DEVICE)
    print(model)

    if os.path.isfile(os.path.join(MODEL_SAVE_DIR, T2_NAME)):
        print(f"MODEL is already trained!! ")
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, T2_NAME)))
    else:
        print(f"Training MODEL...")
        optimizer = torch.optim.Adam(model.parameters(), lr=T2_LR)
        criterion = torch.nn.MSELoss()
        model = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=T2_EPOCHS)
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, T2_NAME))

    # 3) TEST RESULTS
    #eval_model(model, dataloaders_dict['test'], torch.nn.MSELoss())
    visualice(model, dataloaders_dict['test'])



if __name__ == "__main__":
    print("SINGLE TASK2: LANDMARK LOCATION!!! \n")
    single_task2()