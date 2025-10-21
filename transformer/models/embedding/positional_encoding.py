import torch 
from torch import nn 

class PositionalEncoding(nn.Module): 

    def __init__(self, d_model, max_len = 5e3, device): 
        
        super(PositionalEncoding,self).__init__()

        # Create matrix encoding once 
        encoding = torch.zeros(max_len,d_model,device = device)
        encoding.require_grad = False # Encoding don't train 

        pos = torch.arange(0,max_len,device=device).float().unsqueeze(dim=1)

        _2i = toch.arange(0,d_model,step=2,device=device).float()
        denominator = torch.pow(1e4,_2i / d_model)

        encoding[:,0::2] = torch.sin(pos / denominator)
        encoding[:,1::2] = torch.cos(pos / denominator)

        self.encoding = encoding.unsqueeze(0)

    def forward(self,x): 

        batch_size,seq_len, d_model = x.shape

        return x + self.encoding[:,:seq_len,:]



