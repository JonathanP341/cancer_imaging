import torch
from torch.utils.data import Dataset
import nibabel as nib
import os
import numpy as np
import torchio as tio

#Getting the data into a dataset
class BraTSDataSet(Dataset):
    def __init__(self, cases, transform):
        self.cases = cases
        self.transform = transform
        #self.channels = ["flair", "t1", "t1ce", "t2"]
        self.channels = ["flair", "t1ce"] #Switching to just 2 so that we have 2 channels instead of 4 to make it much faster
        #self.label_dict = {0: "No Tumor", 1: "Necrotic & Non-enchancing Tumor", 2:"Edema", 4: "Enhancing Tumor"}
        self.label_dict = {0: "No Tumor", 1: "Tumor"}
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case_path = self.cases[idx] #Getting the relative path of the case specified
        brats_num = case_path.split("\\")

        #Now we have to seperate it into image and label or X and y
        
        #Getting X
        inputs = []
        for type in self.channels:
            inputs.append(nib.load(os.path.join(case_path, brats_num[-1] + "_" + type + ".nii.gz")).get_fdata()) #Getting all of the types we want
        x = np.stack(inputs, axis=0)

        #Getting y
        y = nib.load(os.path.join(case_path, brats_num[-1] + "_seg.nii.gz")).get_fdata()

        #Coverting the label in y from 4 to 3 because we can just assume 3 is 4 and its continuous
        #y[y == 4] = 3

        #Using binary classification for now to make it easier for dice loss
        y[y >= 1] = 1 # Works because this is a numpy array

        if self.transform:
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            subject = tio.Subject(
                image = tio.ScalarImage(tensor=x), #This is (Channels, D, H, W)
                label = tio.LabelMap(tensor=y.unsqueeze(0)) #Makes out seg mask  (1, D, H, W)
            )
            transformed = self.transform(subject)
            x = transformed.image.data #Keeps it the same size
            y = transformed.label.data.squeeze(0) #Removing that first dimension at the beginning
            return x.float(), y.long()
        else:
            return torch.from_numpy(x).float(), torch.from_numpy(y).long()