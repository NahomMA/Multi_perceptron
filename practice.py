import torch
from config import cfg
import numpy as np
from data import get_data_loader
import network
y=torch.tensor(4.0,requires_grad=True)
print(y.dtype)
z=y**2
z.backward()
print(y.grad)
print(cfg['batch_size'])
DATA_PATH = 'C:\\Users\\Nahom M Birhan\\Desktop\\Fall_2021\\Int_to_ML_for_ENG\\hw6\\hw6\dataset\\Q1'
train_images=np.load(DATA_PATH+ '\\train_images.npy')/255
# print(train_images.shape)
train=get_data_loader('train')


for batch_idx, (data, target) in enumerate(train):
    print('train=',data.shape)
    print('target=',target.shape)
    break
def test(param):
    if param==x:
        print(True)
        print('x',param)
        
    else:
        print(False)
        print('Not x',param)

x=10
y=3
test(x)

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

inst=AvgMeter()
n=AvgMeter()
for i in range (3):
    inst.update(i,3)
    n.update(i,6)
print('avg',inst.get_avg())
print(inst.cnt)
print(n.qty)
net=network.Network()
net.eval()
net.save('test_model')

print(net)
print('saved moddl\n')
print(net.load('test_model'))
for batch_idx, (data, target) in enumerate(train):
    net.forward(data)
    break

from numpy.linalg import inv2
a = np.array(np.identity(4))
np.linalg()