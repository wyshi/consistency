from PersuasionInteract import PersuasiveBot

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,5"


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import regex as re
import random
import itertools
import tqdm
import time

from torch.utils.tensorboard import SummaryWriter
from apex import amp
from gpt_model import GPT2SimpleLM
from pytorch_pretrained_bert import GPT2Tokenizer, OpenAIAdam
from allennlp.nn.beam_search import BeamSearch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torchfly.modules.losses import SequenceFocalLoss, SequenceCrossEntropyLoss


# In[3]:


class PersuadeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        self.turn_ending = tokenizer.encode("\n\n\n")
        self.dialog_ending = [tokenizer.encoder["[EOS]"]]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dial_tokens = [tokenizer.encode(item) + self.turn_ending for item in self.data[index]]
        role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
        dial_tokens[-1] = dial_tokens[-1][:-2] + self.dialog_ending
        return role_ids, dial_tokens
        

class Collate_Function:
    """This function handles batch collate.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.EOS = self.tokenizer.encoder["[EOS]"]

    def __call__(self, unpacked_data):
        return unpacked_data


# In[4]:


def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
    """
    # batch support!
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
        logits = torch.where(logits < min_values, 
                             torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
                             logits)
    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    
    return logits


# In[5]:


tokenizer = torch.load("DataProcess/special3_gpt2_tokenizer.pkl")

class GPT2SmallConfig:
    vocab_size = 50257 + len(tokenizer.__special_tokens__)
    n_special = len(tokenizer.__special_tokens__)
    n_positions = 1024
    n_ctx = 1024
    n_embd = 768
    n_layer = 12
    n_head = 12
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02
    gradient_checkpointing = False
    
class GPT2MediumConfig:
    vocab_size = 50257 + len(tokenizer.__special_tokens__)
    n_special = len(tokenizer.__special_tokens__)
    n_positions = 1024
    n_ctx = 1024
    n_embd = 1024
    n_layer = 24
    n_head = 16
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02
    gradient_checkpointing = True


# In[6]:


tokenizer.encode("[name_slot]")


# In[7]:


tokenizer.decode([43384])


# ### load the data

# In[8]:


train_data = torch.load("DataProcess/train_dialogs.pkl")
val_data = torch.load("DataProcess/val_dialogs.pkl")

train_dataset = PersuadeDataset(train_data, tokenizer)
val_dataset = PersuadeDataset(val_data, tokenizer)

batch_size = 1
collate_func = Collate_Function(tokenizer)

train_dataloader = DataLoader(dataset=train_dataset, 
                              shuffle=True, 
                              batch_size=batch_size, 
                              collate_fn=collate_func)
val_dataloader = DataLoader(dataset=val_dataset, 
                            shuffle=False, 
                            batch_size=batch_size, 
                            collate_fn=collate_func)


# In[9]:


model_A = GPT2SimpleLM(GPT2SmallConfig)
model_B = GPT2SimpleLM(GPT2SmallConfig)
model_A_states, model_B_states = torch.load("Checkpoint/best.th", map_location="cuda:1")#torch.load("CheckpointMedium/model_state_epoch_3.th")

# model_A = GPT2SimpleLM(GPT2MediumConfig)
# model_B = GPT2SimpleLM(GPT2MediumConfig)
# model_A_states, model_B_states = torch.load("Checkpoint/best.th")#torch.load("persuasion_medium_3.th")

model_A.load_state_dict(model_A_states)
model_B.load_state_dict(model_B_states)


# In[10]:


device = torch.device("cuda")
model_A = model_A.to(device)
model_B = model_B.to(device)


# ## Test

# In[30]:


model_A.eval()
model_B.eval()

prev_input = tokenizer.encode("A:")
prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)
# past_position_ids = torch.LongTensor([[0, 1]]).to(device)

temperature = 0.8
past = None
flag = True

sep = tokenizer.encode("\n\n\n")

while flag:
    "Sampling based method"
    sent = []
    with torch.no_grad():
        for i in range(200):
            logits, past = model_A(prev_input, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_filtering(logits, top_k=200, top_p=0.9)
            # prev_input = logits.argmax(-1).unsqueeze(1)
            probs = F.softmax(logits, -1)
            prev_input = torch.multinomial(probs, num_samples=1)
            prev_word = prev_input.item()

            if prev_word == 628:
                break
            elif prev_word == tokenizer.encoder["[EOS]"]:
                flag = False
                break
            else:
                sent.append(prev_word)
            
            # past_position_ids = past_position_ids[:, -1:] + 1

    if not flag:
        break

    print("A:" + tokenizer.decode(sent))
    
    # finish tail
    prev_input = torch.LongTensor(sep).unsqueeze(0).to(device)
    _, past = model_A(prev_input, past=past)
    
    # input and update B's utterance
#     user = input("B:")
    
#     if user == "quit":
#         break
        
#     user = tokenizer.encode("B:" + user)
#     prev_input = user + sep
#     prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)
    
#     _, past = model_B(prev_input, past=past)
    
    prev_input = tokenizer.encode("B:")
    prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)
    
    sent = []
    with torch.no_grad():
        for i in range(200):
            logits, past = model_B(prev_input, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_filtering(logits, top_k=200, top_p=0.9)
            # prev_input = logits.argmax(-1).unsqueeze(1)
            probs = F.softmax(logits, -1)
            prev_input = torch.multinomial(probs, num_samples=1)
            prev_word = prev_input.item()

            if prev_word == 628:
                break
            elif prev_word == tokenizer.encoder["[EOS]"]:
                flag = False
                break
            else:
                sent.append(prev_word)
    
    print("B:" + tokenizer.decode(sent))
    
    # finish tail
    prev_input = torch.LongTensor(sep).unsqueeze(0).to(device)
    _, past = model_B(prev_input, past=past)
    
    # start A's utterance
    suffix = tokenizer.encode("A:")
    prev_input = torch.LongTensor(suffix).unsqueeze(0).to(device)


# In[12]:


# past[0].shape


# In[ ]:





# In[ ]:




