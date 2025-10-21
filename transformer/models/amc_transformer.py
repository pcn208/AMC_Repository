import torch
from torch import nn

from encoder import Encoder 

class AMCTransformer(nn.Module): 

    def __init__(self,num_patches,patch_size,num_classes,d_model,n_head,n_layers,ffn_hidden,drop_prob,device): 
        super().__init__()

        self.encoder = Encoder(num_patches = num_patches,
                               patch_size = patch_size,
                               d_model=d_model,
                               n_head=n_head,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.mlp_head = nn.Linear(d_model,num_classes)

    def forward(self,src): 

        enc_output = self.encoder(src) -> (batch_size,num_pathes + 1, d_model)

        cls_output = enc_output[:,0]

        output = self.mlp_head(cls_output)

        retur output 
