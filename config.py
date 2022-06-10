import os
from random import randint
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
MODEL_SAVE_DIR = ROOT_DIR + 'saves/'
LOG_DIR = ROOT_DIR + 'log/'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE_IMAGE = (178, 218)

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
T1_FOLDER = "task1"
T1_NAME = os.path.join(T1_FOLDER, "single_task1.pt")
T1_SIZE = (224, 224)
T1_CLASES = 2
T1_BATCH = 32
T1_EPOCHS = 3
T1_LR = 1e-4

#####################################
# SINGLE TASK2 - LANDMARKS 
#####################################
T2_FOLDER = "task2"
T2_NAME = os.path.join(T2_FOLDER, "single_task2.pt")
T2_SIZE = (224, 224)
T2_CLASES = 5*2
T2_BATCH = 32
T2_EPOCHS = 50
T2_LR = 1e-3

#####################################
# MULTI TASK3 
#####################################
T3_FOLDER = "task3"
T3_NAME = os.path.join(T3_FOLDER, "multi_task3.pt")
T3_SIZE = (224, 224)
T3_BATCH = 32
T3_EPOCHS = 50
T3_LR = 1e-3