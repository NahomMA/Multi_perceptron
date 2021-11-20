import os
import torch
from torch.utils.data import Dataset, DataLoader, dataloader
import numpy as np
from config import cfg

# TODO: Define your data path (the directory containing the 4 np array files)
DATA_PATH = 'C:\\Users\\Nahom M Birhan\\Desktop\\Fall_2021\\Int_to_ML_for_ENG\\hw6\\hw6\dataset\\Q1'
class FMNIST(Dataset):
    def __init__(self, set_name):
        
        super(FMNIST, self).__init__()
        # TODO: Retrieve all the images and the labels, and store them
        self.set_name=set_name
        if set_name=='train':
           
            self.train_images=np.load(DATA_PATH+'\\train_images.npy')/255
            self.train_lables=np.load(DATA_PATH+'\\train_labels.npy')
        if set_name=='test':
          
            self.test_images=np.load(DATA_PATH+'\\test_images.npy')/255
            self.test_lables=np.load(DATA_PATH+'\\test_labels.npy')
        
        # as class variables. Maintaing any other class variables that 
        # you might need for the other class methods. Note that the 
        # methods depends on the set (train or test) and thus maintaining
        # that is essential.
      
        
    
    def __len__(self):
        # TODO: Complete this
        if self.set_name=='train':
            return len(self.train_lables)
    
        if self.set_name=='test':
            return len(self.test_lables)
        
        
      
    def __getitem__(self, index):
        # TODO: Complete this
        if self.set_name=='train':
            return torch.from_numpy(self.train_images[index]).reshape(28*28), np.long(self.train_lables[index])


        if self.set_name=='test':
            return torch.from_numpy(self.test_images[index]).reshape(28*28), np.long(self.test_lables[index])

def get_data_loader(set_name):
    # TODO: Create the dataset class tailored to the set (train or test)
    # provided as argument. Use it to create a dataloader. Use the appropriate
    # hyper-parameters from cfg
    # Parameters
    dataset=FMNIST(set_name)
    dataloader=DataLoader(dataset=dataset, batch_size=cfg['batch_size'],num_workers=0 )     
    
    return dataloader
    
    
