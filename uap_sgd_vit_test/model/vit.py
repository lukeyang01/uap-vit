import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import torch.nn.functional as F
from train_common import patchify, get_positional_embeddings
torch.manual_seed(42)
import pdb
import random
import math
random.seed(42)
from train_common import *
# Original credit to:
# Author:   Brian Pulfer
# URL:      https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# Created:  2024-07-06


class TransformerEncoder(nn.Module):
    def __init__(self,hidden_d,n_heads,mlp_ratio=4):
        super(TransformerEncoder, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(hidden_d)

        self.multi_head_attention = MultiHeadAttention(hidden_d,n_heads)
        
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d,mlp_ratio*hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio*hidden_d,hidden_d))
    def forward(self, x):
        # TODO: Define the foward pass of the Transformer Encoder block as illistrated in 
        #       Figure 4 of the spec.
        # NOTE: Don't forget about the residual connections!
        x = x + self.multi_head_attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,num_features,num_heads):
        super().__init__()
        
        self.num_features = num_features
        self.num_heads = num_heads
        #Dimension of each atention head's 
        query_size = int(num_features/num_heads)


        #Note: nn.ModuleLists(list) taskes a python list of layers as its parameters
        #The object at the i'th index of the list passed to nn.ModuleList 
        #should corresopnd to the i'th attention head's K,Q, or V respective learned linear mapping
        q_modList_input = [nn.Linear(query_size,query_size) for _ in range(num_heads)]
        self.Q_mappers = nn.ModuleList(q_modList_input)

        k_modList_input = [nn.Linear(query_size,query_size) for _ in range(num_heads)]
        self.K_mappers = nn.ModuleList(k_modList_input)

        v_modList_input = [nn.Linear(query_size,query_size) for _ in range(num_heads)]
        self.V_mappers = nn.ModuleList(v_modList_input)


        self.query_size = query_size
        self.scale_factor = math.sqrt(num_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        result = []
        #Remember, we turned each image into a sequence of 16 dimensional "tokens" for our model
        #Loop through the batch of patch embedding sequences
        for sequence in x:
            #each element in seq_result should be a single attention head's
            # attention values
            seq_result = []
            for head in range(self.num_heads):
                W_k = self.K_mappers[head]
                W_q = self.Q_mappers[head]
                W_v = self.V_mappers[head]

                #Extract the portion of the embedding for the given head
                #If we have n attention heads and an embedding of size e,
                #the query size will be := e/n
                seq = sequence[:, head * self.query_size: (head + 1) * self.query_size]

                #Get the given head's k,q,and v representations
                k = W_k(seq)
                q = W_q(seq)
                v = W_v(seq)

                #Perform scaled dot product self attention, refer to formula
                attention = self.softmax(q @ k.T / (self.query_size ** 0.5))
                attention = attention @ v

                #Log the current attention head's attention values
                seq_result.append(attention)

            #For the current sequence (patched image) being processed,
            #combine each attention head's attention values columnwise
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class ViT(nn.Module):
    def __init__(self,
                 num_patches, 
                 num_blocks,
                 num_hidden,
                 num_heads,
                 num_classes = 2,
                 chw_shape = (3,64,64)):
        super(ViT, self).__init__()

        # Attributes
        self.chw = chw_shape
        self.num_patches = num_patches

        #Tip: What would the size of a single patch be given the width/height 
        # of an image and the number of patches? While the final patch size should be 2D,
        # it may be easier to consider each dimesnion separately as a starting point.
        self.patch_size = (self.chw[1] / num_patches, self.chw[2] / num_patches)
        self.embedding_d = num_hidden
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        # 1) Patch Tokenizer
            # input_d should hold the number of pixels in a single patch, 
            # dont forget a patch is created with pixels across all img chanels
        self.input_d = int(self.chw[0] * self.patch_size[0] * self.patch_size[1])

        # create a linear layer to embed each patch token
        self.patch_to_token = nn.Linear(self.input_d, self.embedding_d)

        # 2) Learnable classifiation token
        # Use nn.Parameter to create a learnable classification token of shape (1,self.embedding_d)
        self.cls_token = nn.Parameter(torch.rand(1, self.embedding_d))
        
        # 3) Positional embedding
        self.pos_embed = nn.Parameter(get_positional_embeddings(self.num_patches ** 2 + 1, self.embedding_d).clone().detach())
        self.pos_embed.requires_grad = False

        # 4) Transformer encoder blocks
        # Add the number of transformer blocks specified by num_blocks
        transformer_block_list = [TransformerEncoder(num_hidden, num_heads) for _ in range(num_blocks)]
        self.transformer_blocks = nn.ModuleList(transformer_block_list)

        # 5) Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_d, num_classes),
            nn.Softmax(dim=-1))
        
    def forward(self, X):
        B, C, H, W = X.shape

        #patch images
        patches = patchify(X,self.num_patches)

        # TODO: Get linear projection of each patch
        embeded_patches = self.patch_to_token(patches)

        #add classification (sometimes called 'cls') token to the tokenized_patches
        all_tokens = torch.stack([torch.vstack((self.cls_token, embeded_patches[i])) for i in range(len(embeded_patches))])
        
        # Adding positional embedding
        pos_embed = self.pos_embed.repeat(B, 1, 1)
        all_tokens = all_tokens + pos_embed

        # TODO: run the positionaly embeded tokens
        #       through all transformer blocks stored in self.transformer_blocks
        for block in self.transformer_blocks:
            all_tokens = block(all_tokens)

        # Extract the classification token and put through mlp
        all_tokens = all_tokens[:, 0]
        all_tokens = self.mlp(all_tokens)

        return all_tokens