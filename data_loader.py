import os
from BraTSDataSet import BraTSDataSet
from torch.utils.data import DataLoader


def create_dataloaders(data_path: str, transform, batch_size: int):
    #Getting all of the files 
    cases = []
    for d in os.listdir(data_path): #Looping through all of the directories in the folder and appending them to the 
        if d.startswith("BraTS2021"):
            cases.append(os.path.join(data_path, d)) #Putting the full relative path in the list

    #Creating the training set
    train_dataset = BraTSDataSet(cases=cases[:int(len(cases)*0.15)], transform=transform) #Contains 1000 cases at first, now 187
    #Creating a validation set
    valid_dataset = BraTSDataSet(cases=cases[int(len(cases)*0.15):int(len(cases)*0.2)], transform=transform) #Contains ~63
    #Creating the testing set
    test_dataset = BraTSDataSet(cases=cases[int(len(cases)*0.4):int(len(cases)*0.5)], transform=transform) #Contains 251 cases, now 125

    #Converting the sets into dataloaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
    
    return train_dataloader, valid_dataloader, test_dataloader, train_dataset.channels, train_dataset.label_dict