import os
from random import randint
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
MODEL_SAVE_DIR = ROOT_DIR + 'saves/'
LOG_DIR = ROOT_DIR + 'log/'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########################
# DATA PATH CONFIGURATION
#########################
BASE_FOLDER = 'celeba-mini'
CSV_FILENAME = 'celeba-mini.csv'
IMG_FOLDER = 'images'
TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.3

#####################################
# SINGLE TASK1 - GENDER CLASIFICATION 
#####################################
T1_NAME = "single_task1.pt"
T1_SIZE = (224, 224)
T1_CLASES = 2
T1_BATCH = 32
T1_EPOCHS = 3
T1_LR = 1e-4