"""
DATASET CLASS
Based on the teacher's code, modify to include the training-testing division.
"""

from torchvision.datasets.vision import VisionDataset
import PIL
import os
import torch
import pandas
import pdb
from config import *
from sklearn.model_selection import train_test_split
import pdb

class CelebAMini(VisionDataset):
    """
    CelebA-mini is a subsample of the CelebA dataset 
    (<http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>)
    
    Args: 
        root (string): Root directory where the celeba-mini directory is
        transform (callable, optional): Input image transforms
        target_transform (callable, optional): Transform functions for the targets
        
    Dataset:
        2-tuple with:
         - sample: The input image (PIL.Image)
         - target: 2-tuple with:
           - gender: 0 for female, 1 for male
           - landmarks: 10x1 Tensor with:
             [0]: left eye x coordinate
             [1]: left eye y coordinate
             [2]: right eye x coordinate
             [3]: right eye y coordinate
             [4]: nose x coordinate
             [5]: nose y coordinate
             [6]: left mouth x coordinate
             [7]: left mouth y coordinate
             [8]: right mouth x coordinate
             [9]: right mouth y coordinate
    """

    base_folder = BASE_FOLDER
    csv_filename= CSV_FILENAME
    img_folder = IMG_FOLDER
    
    def __init__(self, root, 
                 transform = None,
                 target_transform = None, split="train", s_f=1) :
        super(CelebAMini,self).__init__(root, transform=transform, 
                                        target_transform=target_transform)
        
        #check the paths
        csv_path = os.path.join(self.root, self.base_folder, self.csv_filename)
        self.img_path = os.path.join(self.root, self.base_folder, self.img_folder)
        if not (os.path.isfile(csv_path) and os.path.isdir(self.img_path)):
            raise RuntimeError(f"Dataset not found in '{self.root}'.")
        
        #load the csv data
        csv_data = pandas.read_csv(csv_path, sep=',\s+', header=0, engine='python', index_col=0)

        # DIVIDIR EN TRAIN Y TEST
        y=csv_data['gender']
        # Random state para obtener siempre la misma particion
        train, test = train_test_split(csv_data, test_size=TEST_SPLIT, stratify=y, random_state=88)

        if split=="train":
            self.filename = train.index.values
            self.gender = torch.as_tensor(train.iloc[:,0].values)
            self.landmarks = torch.as_tensor(train.iloc[:, 1:].values)
        else:
            self.filename = test.index.values
            self.gender = torch.as_tensor(test.iloc[:,0].values)
            self.landmarks = torch.as_tensor(test.iloc[:, 1:].values)

        #verify images exist
        for f in self.filename:
            this_path = os.path.join(self.img_path, f)
            if not os.path.isfile(this_path):
                raise RuntimeError(f"Dataset corruption: Image '{this_path}' not found.")
        # SCALE FACTOR
        self.s_f = s_f

    def __len__(self):
        return len(self.filename)
    
    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.img_path, self.filename[index]))
        
        if self.transform is not None:
            X = self.transform(X)
        
        T = (self.gender[index], self.landmarks[index,:])
        if self.target_transform is not None:
            T_reg = self.transform(T_reg)

        # Adaptar coordenadas de regresion al resize de la imagen:
        T_reg = T[1].reshape((-1,2))
        T_reg = (T_reg * self.s_f).long()
        T_reg = T_reg.reshape(-1)
        T = (T[0], T_reg)

        return X,T
                           
    
   
   
   
if __name__ == "__main__":
    # Unit tests
    
    test = CelebAMini(os.getcwd())
    
    for x,t in test:
        print(x)
        print(t)
        
    import matplotlib.pyplot as plt
    import random
    
    rindex = random.randint(0,len(test))
    x, t = test[rindex]
    
    print(test)
    print(test.filename[rindex])
    print(x)
    print(t[0])
    print(t[1])
    
    lm = t[1].reshape((-1,2))
    plt.imshow(x)
    plt.title("Male" if t[0].item() else "Female")
    plt.scatter(lm[0,0], lm[0,1], marker="<")
    plt.scatter(lm[1,0], lm[1,1], marker=">")
    plt.scatter(lm[2,0], lm[2,1], marker="^")
    plt.scatter(lm[3,0], lm[3,1], marker="3")
    plt.scatter(lm[4,0], lm[4,1], marker="4")
    plt.show()
    
