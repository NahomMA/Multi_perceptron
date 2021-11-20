import torch
import torch.nn as nn

def xavier_init(param):
    # NOTE: Not for Vanilla Classifier
    # TODO: Complete this to initialize the weights
    nn.init.xavier_normal_(param, gain=1.0)
    

def zero_init(param):
    # NOTE: Not for Vanilla Classifier
    # TODO: Complete this to initialize the weights
    nn.init.zeros_(param)
    

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO: Define the model architecture here
        
        self.L1=nn.Linear(784,100)
        self.L2=nn.Linear(100,10)
        
        # self.model = nn.Sequential(
        # nn.Linear(784,100),
        
        # nn.ReLU(inplace=True),
        # nn.Linear(100,10),
        # nn.Softmax(dim=1)
        # )
        #raise NotImplementedError
        # NOTE: Not for Vanilla Classsifier
        # TODO: Initalize weights by calling the
        # init_weights method
        self.init_weights()
    def init_weights(self):
        # NOTE: Not for Vanilla Classsifier
        # TODO: Initalize weights by calling by using the
        # appropriate initialization function
    
# =============================================================================
#         xavier_init(self.L2.weight)
#         xavier_init(self.L1.weight)
#         zero_init(self.L1.bias)
#         zero_init(self.L2.bias)
# =============================================================================
        
        # zero_init(self.L1.weight)
        # zero_init(self.L1.bias)
        # zero_init(self.L2.weight)
        # zero_init(self.L2.bias)
        pass
        
    def forward(self, x):
        # TODO: Define the forward function of your model
        x=x.float()
        L1_out=self.L1(x)
        # print('before',L1_out)
        m = nn.ReLU()
        L1_out = m(L1_out)
        # print('after',L1_out)
        out=self.L2(L1_out)
        # m2 = nn.Softmax()
        # out = m2(out)
        
        
        
        #activated = self.activation
        #l2_out = self.L2(activated)
        # raise NotImplementedError
        
        
        return out
    
    def save(self, ckpt_path):
        # TODO: Save the checkpoint of the model
      
       torch.save(self.state_dict(), ckpt_path)
        

    def load(self, ckpt_path):
        # TODO: Load the checkpoint of the model
        # model = torch.load(ckpt_path)
        model = Network()
        model.load_state_dict(torch.load(ckpt_path))

        return model