#!/usr/bin/env python -W ignore::DeprecationWarning
# coding: utf-8

# In[1]:


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
from allennlp.training.checkpointer import Checkpointer
# from gpt_model import GPT2SimpleLM
# from pytorch_pretrained_bert import GPT2Tokenizer, OpenAIAdam, GPT2Model
# from torchfly.criterions import SequenceFocalLoss, SequenceCrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW#, WarmupLinearSchedule
from torchfly.modules.losses import SequenceFocalLoss, SequenceCrossEntropyLoss
from UnlikelihoodLoss import SequenceUnlikelihoodLoss
# In[2]:


torch.backends.cudnn.benchmark = True
torch.manual_seed(123)
np.random.seed(123)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


# In[3]:


# class PersuadeDataset(Dataset):
#     def __init__(self, data, tokenizer):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.tokenizer.max_len = 1500
#         self.turn_ending = tokenizer.encode("\n\n\n")
#         self.dialog_ending = [tokenizer.encoder["[EOS]"]]
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, index):
#         dial_tokens = [tokenizer.encode(item) + self.turn_ending for item in self.data[index]]
#         role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
#         dial_tokens[-1] = dial_tokens[-1][:-2] + self.dialog_ending
#         return role_ids, dial_tokens
        

# class Collate_Function:
#     """This function handles batch collate.
#     """
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#         self.EOS = self.tokenizer.encoder["[EOS]"]
        
#     def __call__(self, unpacked_data):
#         return unpacked_data


# # In[4]:

class PersuadeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        # tokenizer weird behavior
        self.turn_ending = [628, 198]
        # tokenizer.encode("\n\n\n")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dial_tokens = [self.tokenizer.encode(item) + self.turn_ending for item in self.data[index]]
        role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
        return role_ids, dial_tokens
        
    def collate(self, unpacked_data):
        return unpacked_data

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# class GPT2SmallConfig:
#     vocab_size = 50257 + len(tokenizer.__special_tokens__)
#     n_special = len(tokenizer.__special_tokens__)
#     n_positions = 1024
#     n_ctx = 1024
#     n_embd = 768
#     n_layer = 12
#     n_head = 12
#     resid_pdrop = 0.1
#     embd_pdrop = 0.1
#     attn_pdrop = 0.1
#     layer_norm_epsilon = 1e-5
#     initializer_range = 0.02
#     gradient_checkpointing = False
    
# class GPT2MediumConfig:
#     vocab_size = 50257 + len(tokenizer.__special_tokens__)
#     n_special = len(tokenizer.__special_tokens__)
#     n_positions = 1024
#     n_ctx = 1024
#     n_embd = 1024
#     n_layer = 24
#     n_head = 16
#     resid_pdrop = 0.1
#     embd_pdrop = 0.1
#     attn_pdrop = 0.1
#     layer_norm_epsilon = 1e-5
#     initializer_range = 0.02
#     gradient_checkpointing = True


# In[5]:



model_A = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model_B = GPT2LMHeadModel.from_pretrained("gpt2-medium")
import config as cfg
# model_A_states, model_B_states = torch.load(cfg.old_medium_model_dir, map_location="cuda:5")#torch.load("CheckpointMedium/model_state_epoch_3.th")
# model_A_states['transformer.wte.weight'] = model_A_states['transformer.wte.weight'][:50257,:]
# model_A_states['lm_head.weight'] = model_A_states['lm_head.decoder.weight'][:50257,:]
# model_B_states['transformer.wte.weight'] = model_B_states['transformer.wte.weight'][:50257,:]
# model_B_states['lm_head.weight'] = model_B_states['lm_head.decoder.weight'][:50257,:]
model_A_states, model_B_states = torch.load("models/persuasion-gpt2-medium.pth", map_location="cuda:5")#torch.load("CheckpointMedium/model_state_epoch_3.th")
# print("load success")

model_A.load_state_dict(model_A_states, strict=False)
model_B.load_state_dict(model_B_states, strict=False)
# model_A.load_state_dict(torch.load("/home/qingyang/Desktop/GPT2_Modification/special3_gpt2_small.pth"))
# model_B.load_state_dict(torch.load("/home/qingyang/Desktop/GPT2_Modification/special3_gpt2_small.pth"))

# model_A = GPT2SimpleLM(GPT2MediumConfig)
# model_B = GPT2SimpleLM(GPT2MediumConfig)
# model_A.load_state_dict(torch.load("/home/qingyang/Desktop/GPT2_Modification/special3_gpt2_medium.pth"))
# model_B.load_state_dict(torch.load("/home/qingyang/Desktop/GPT2_Modification/special3_gpt2_medium.pth"))


# ### load the data

# In[6]:


train_data = torch.load("DataProcess/train_dialogs.pkl")
val_data = torch.load("DataProcess/val_dialogs.pkl")

train_dataset = PersuadeDataset(train_data, tokenizer)
val_dataset = PersuadeDataset(val_data, tokenizer)

batch_size = 1

train_dataloader = DataLoader(dataset=train_dataset, 
                              shuffle=True, 
                              batch_size=batch_size, 
                              collate_fn=train_dataset.collate)
val_dataloader = DataLoader(dataset=val_dataset, 
                            shuffle=False, 
                            batch_size=batch_size, 
                            collate_fn=train_dataset.collate)


# ## Define the model

# In[7]:


device = torch.device("cuda:5")
torch.cuda.set_device(device)
model_A = model_A.to(device)
model_B = model_B.to(device)


# In[8]:


# define the losses
criterion = SequenceFocalLoss(gamma=1.0, beta=0.0)
eval_criterion = SequenceCrossEntropyLoss()
# unlikelihood_criterion = SequenceUnlikelihoodLoss(padding_idx=tokenizer.encoder["[PAD]"])

# In[9]:


def train_one_iter(batch, update_count, fp16=False):
    role_ids, dialog_tokens = batch
    dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]
    
    past = None
    all_logits = []
    # A_logits = []
    # B_logits = []
    # A_target = []
    # B_target = []
#     user = tokenizer.encode("B:" + user)
#     sep = tokenizer.encode("\n\n\n") 
#     suffix = tokenizer.encode("A:")
#     prev_input = sep + user + sep + suffix
    
#     prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(device)
#     past_length = past_position_ids.item()
    
#     past_position_ids = np.arange(past_length, past_length+2).tolist() + \
#                          np.arange(len(user) + 2).tolist() + \
#                          np.arange(2).tolist()
    
#     past_position_ids = torch.LongTensor(past_position_ids).unsqueeze(0).to(device)
    
    for turn_num, dial_turn_inputs in enumerate(dial_inputs):
        if role_ids[turn_num] == 0:
            # breakpoint()
            logits, past = model_A(dial_turn_inputs, past=past)
            all_logits.append(logits)
        else:
            # breakpoint()
            logits, past = model_B(dial_turn_inputs, past=past)
            all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=1) # torch.Size([1, 505, 50260]), 505 = sum of tokens from 21 sentences
    
    
    
    # target
    all_logits = all_logits[:, :-1].contiguous() # torch.Size([1, 504, 50260])
    target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()# torch.Size([1, 504])
    target_mask = torch.ones_like(target).float()# torch.Size([1, 504])
    
    if False:
        loss = criterion(all_logits, target, target_mask, label_smoothing=0.02, reduce=True) # torch.Size([])
    else:
        loss = unlikelihood_criterion(all_logits, target)
    loss /= num_gradients_accumulation
    
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
        
    record_loss = loss.item() * num_gradients_accumulation
    # print("record_loss: {}".format(record_loss))
    perplexity = np.exp(record_loss)
    
    return record_loss, perplexity

import tqdm
def validate(dataloader, ep=0):
    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader)

        total_ppl = []

        for batch in pbar:
            
            if sum([len(item) for item in batch[0][1]]) > 1024:
                continue
            
            role_ids, dialog_tokens = batch[0]
            dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]

            past = None
            all_logits = []
            # A_logits = []
            # B_logits = []
            # A_target = []
            # B_target = []

            for turn_num, dial_turn_inputs in enumerate(dial_inputs):
                if role_ids[turn_num] == 0:
                    logits, past = model_A(dial_turn_inputs, past=past)
                    all_logits.append(logits)
                else:
                    logits, past = model_B(dial_turn_inputs, past=past)
                    all_logits.append(logits)

            all_logits = torch.cat(all_logits, dim=1)
            
            # target
            all_logits = all_logits[:, :-1].contiguous()
            target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
            target_mask = torch.ones_like(target).float()
            
            loss = eval_criterion(all_logits, target, target_mask, label_smoothing=-1, reduce="sentence")      

            ppl = torch.exp(loss)
            total_ppl.extend(ppl.tolist())

        print(f"Epoch {ep} Validation Perplexity: {np.mean(total_ppl)} Variance: {np.var(total_ppl)}")
        
        return np.mean(total_ppl)


# ### Training

# In[10]:

if False:
    checkpointer = Checkpointer(serialization_dir="Checkpoint", 
                                keep_serialized_model_every_num_seconds=3600*2, 
                                num_serialized_models_to_keep=5)


    # In[11]:


    # optimizer
    num_epochs = 10
    num_gradients_accumulation = 1
    num_train_optimization_steps = num_train_optimization_steps = len(train_dataset) * num_epochs // batch_size // num_gradients_accumulation

    param_optimizer = list(model_A.named_parameters()) + list(model_B.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]


    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                        lr=2e-5,
                        warmup=0.1,
                        max_grad_norm=1.0,
                        weight_decay=0.01,
                        t_total=num_train_optimization_steps)


    # In[12]:


    # support fp16
    # [model_A, model_B], optimizer = amp.initialize([model_A, model_B], optimizer, opt_level="O1")


# In[13]:
if False:
    from tqdm import tqdm as tqdm_bar
    update_count = 0
    progress_bar = tqdm.tqdm_notebook
    start = time.time()
    old_ppl = -float('Inf')

    for ep in tqdm_bar(range(num_epochs)):

        "Training"
        pbar = progress_bar(train_dataloader)
        model_A.train()
        model_B.train()
        
        for batch in pbar:
            batch = batch[0]
            # without relative position
            if sum([len(item) for item in batch[1]]) > 1024:
                continue
                
            record_loss, perplexity = train_one_iter(batch, update_count, fp16=False)
            
            update_count += 1

            if update_count % num_gradients_accumulation == num_gradients_accumulation - 1:
                # update for gradient accumulation
                optimizer.step()
                optimizer.zero_grad()
                
                # speed measure
                end = time.time()
                speed = batch_size * num_gradients_accumulation / (end - start)
                start = end
                
                # show progress
                pbar.set_postfix(loss=record_loss, perplexity=perplexity, speed=speed)

        "Evaluation"
        model_A.eval()
        model_B.eval()
        ppl = validate(val_dataloader)
        
        is_best_so_far = ppl > old_ppl
        old_ppl = ppl
        checkpointer.save_checkpoint(ep, [model_A.state_dict(), model_B.state_dict()], {"None": None}, is_best_so_far)

else:
    validate(val_dataloader)
# In[ ]:





# In[ ]:




