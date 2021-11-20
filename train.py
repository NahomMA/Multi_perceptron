import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from data import get_data_loader
from network import Network
from config import cfg

try:
    from termcolor import cprint
except ImportError:
    cprint = None

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

def get_lr(optimizer):
    #TODO: Returns the current Learning Rate being used by
    # the optimize
    return [lr['lr'] for  lr in optimizer.param_groups][0]

'''
Use the average meter to keep track of average of the loss or 
the test accuracy! Just call the update function, providing the
quantities being added, and the counts being added
'''
class AvgMeter():
    def __init__(self):
        self.qty = 0
        self.cnt = 0
    
    def update(self, increment, count):
        self.qty += increment
        self.cnt += count
    
    def get_avg(self):
        if self.cnt == 0:
            return 0
        else: 
            return self.qty/self.cnt


def run(net, epoch, loader, optimizer, criterion, logger, scheduler, train=True):
    
        # TODO: Performs a pass over data in the provided loader
     # TODO: Initalize the different Avg Meters for tracking loss and accuracy (if test)
   
    avg_train=AvgMeter() 
    test_loss=AvgMeter()
    test_acc=AvgMeter()
    # TODO: Iterate over the loader and find the loss. Calculate the loss and based on which
    
    if train:
        
        for batch_idx, (data, target) in enumerate(loader): 
            
            optimizer.zero_grad()
            logit=net(data)
            loss_train=criterion(logit,target)
           
            #TODO: Log the training/testing loss using tensorboard. 
          
            avg_train.update(loss_train,cfg['batch_size'])
            loss_train.backward()
            optimizer.step()
        writer.add_scalar("train loss", loss_train, epoch)   
        return avg_train.get_avg(),()
        
        
          # set is being provided update you model. Also keep track of the accuracy if we are running
          # on the test set.
    if not train:
  
        for batch_idx, (data, target) in enumerate(loader):
            net.eval()
            out_put=net(data)
            loss_test=criterion(out_put,target)
            test_loss.update(loss_test,cfg['batch_size'])
            # TODO: Log the training/testing loss using tensorboard. 
           
            # _, predicted=out_put.max(1)
            predicted=out_put.argmax(1)
            correct = predicted.eq(target).sum().item()
          
            # print("Corrects", correct)
            test_acc.update(correct, cfg['batch_size'])
            # TODO: return the average loss, and the accuracy (if test set)
        writer.add_scalar("test loss", loss_test, epoch)  
        return test_loss.get_avg(), test_acc.get_avg()
        
        


def train(net, train_loader, test_loader, logger):    
    # TODO: Define the SGD optimizer here. Use hyper-parameters from cfg
    optimizer = optim.SGD(net.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
    # TODO: Define the criterion (Objective Function) that you will be using
    
    
    criterion = torch.nn.CrossEntropyLoss( )
    # TODO: Define the ReduceLROnPlateau scheduler for annealing the learning rate
    scheduler =optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=cfg['patience'],factor=cfg['lr_decay'],verbose=True)


    for i in range(cfg['epochs']):
        # Run the network on the entire train dataset. Return the average train loss
        # Note that we don't have to calculate the accuracy on the train set.
        loss, _ = run(net, i, train_loader, optimizer, criterion, logger, scheduler)

        # TODO: Get the current learning rate by calling get_lr() and log it to tensorboard
        writer.add_scalar("learning rate",get_lr(optimizer),i )
        
        # Logs the training loss on the screen, while training
        if i % cfg['log_every'] == 0:
            log_text = "Epoch: [%d/%d], Training Loss:%2f" % (i, cfg['epochs'], loss)
            log_print(log_text, color='green', attrs=['bold'])

        # Evaluate our model and add visualizations on tensorboard
        if i % cfg['val_every'] == 0:
            # TODO: HINT - you might need to perform some step before and after running the network
            # on the test set
            # Run the network on the test set, and get the loss and accuracy on the test set 
            loss, acc = run(net, i, test_loader, optimizer, criterion, logger, scheduler, train=False)
            log_text = "Epoch: %d, Test Accuracy:%2f" % (i, acc*100.0)
            log_print(log_text, color='red', attrs=['bold'])

            # TODO: Perform a step on the scheduler, while using the Accuracy on the test set
            scheduler.step(acc)
            
            # TODO: Use tensorboard to log the Test Accuracy and also to perform visualization of the 
            writer.add_scalar("test loss",loss,i )
             
            writer.add_scalar("test accuracy",acc,i )
            # 2 weights of the first layer of the network!
       

if __name__ == '__main__':
    # TODO: Create a network object
    net = Network()
    # net=net(constant_weight=0)
    

    # TODO: Create a tensorboard object for logging
    writer = SummaryWriter()

    # TODO: Create train data loader
    train_loader = get_data_loader('train')
# =============================================================================
#     for batch_idx, (data, target) in enumerate(train_loader):
#         print('train shape', data.shape)
#         print(len(train_loader))
#         
#         break
#     
# =============================================================================
    

    # TODO: Create test data loader
    test_loader = get_data_loader('test')
# =============================================================================
#     for batch_idx, (data, target) in enumerate(test_loader):
#         print(batch_idx)
#         print('test shape', data.shape)
#         print(len(test_loader))
#         
#         break
# 
# =============================================================================
    # Run the training!
    train(net, train_loader, test_loader, writer)
    writer.flush()