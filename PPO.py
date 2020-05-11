#!/usr/bin/env python
# coding: utf-8
# logging is important
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

from time import time
import logging
from os import listdir
log_dir = max([int(f[3]) for f in listdir(".") if f.startswith("ppo") and f.endswith(".log")]) + 1
log_dir = f"ppo{log_dir}.log"
logging.basicConfig(filename='ppo8.log', level=logging.INFO)
# logging.basicConfig(filename='hello2.log', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import pdb
# pdb.set_trace()
from tqdm import tqdm

# In[37]:
import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import KLDivLoss

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


def cleanup():
    dist.destroy_process_group()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)    


from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
# from gpt_model import GPT2SimpleLM, GPT2MultipleChoiceHead
from GPTModel1 import GPT2LMHeadModel_modified
from pytorch_pretrained_bert import OpenAIAdam
from PersuasionInteract import PersuasiveBot, sent_tokenize_modified
from nltk.tokenize import sent_tokenize
import config as cfg
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import time
import logging
import numba
from torch.utils.data import DataLoader, Dataset
import pdb

#from transformers import WarmupLinearSchedule
from apex.optimizers import FusedLAMB, FusedAdam
from transformers import AdamW, get_linear_schedule_with_warmup
# from torchfly.transformers import UnifiedTokenizer, GPT2SimpleLM
from torchfly.modules.losses import SequenceFocalLoss, SequenceCrossEntropyLoss
# from torchfly.decode import top_filtering


# In[38]:



class PpoParams:
    ppo_epoch = 2
    num_dialogs_to_sample = 1 # number of dialog in one batch
    mini_batchsize = 32#8
    batchsize = mini_batchsize*2#128 
    self_play_prob = 0.0

def make_batch_sequences(sampled_sequences, type_func=torch.LongTensor, padding_value=1):
    # transform into LongTensor
    sampled_sequences = [type_func(item) 
                        for item in sampled_sequences 
                        if not isinstance(item,  type_func)]
    try:
        batch_sequences = nn.utils.rnn.pad_sequence(sampled_sequences, 
                                                    batch_first=True, 
                                                    padding_value=padding_value)
        return batch_sequences

    except:
        pdb.set_trace()

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
        dial_tokens = [self.tokenizer.encode(item[:2]) + self.tokenizer.encode(item[2:]) + self.turn_ending for item in self.data[index]]
        dial_sents = [item[2:] for item in self.data[index]]#self.data[index]#[item[2:] for item in self.data[index]]
        role_ids = [0 if item.startswith("A:") else 1 for item in self.data[index]]
        return role_ids, dial_tokens, dial_sents        
    def collate(self, unpacked_data):
        return unpacked_data

def load_model(cfg, device1, device2, split_into1, split_into2, dropout, device_list1, device_list2, 
               model_A_dir=None, use_old_model_B=False):
    if cfg.model_size == "small":
        lm_config1 = GPT2Config().from_pretrained('gpt2')
        lm_config1.output_hidden_states = True
        lm_config1.resid_pdrop =dropout
        lm_config1.embd_pdrop  =dropout
        lm_config1.attn_pdrop  =dropout
        lm_config1.summary_first_dropout=dropout

        lm_config2 = GPT2Config().from_pretrained('gpt2')
        lm_config2.output_hidden_states = True
        lm_config2.resid_pdrop =dropout
        lm_config2.embd_pdrop  =dropout
        lm_config2.attn_pdrop  =dropout
        lm_config2.summary_first_dropout=dropout
        # model_A = GPT2LMHeadModel_modified(config=lm_config1, device=device1, split_into=split_into)
        # model_B = GPT2LMHeadModel_modified(config=lm_config2, device=device2, split_into=split_into)
        # model_A = GPT2LMHeadModel_modified.from_pretrained("gpt2", output_hidden_states=True,
        #                                                     resid_pdrop =dropout,
        #                                                     embd_pdrop  =dropout,
        #                                                     attn_pdrop  =dropout,
        #                                                     summary_first_dropout=dropout)
        model_A = GPT2LMHeadModel_modified(config=lm_config1)
        model_A.set_variables(device=device1, split_into=split_into1, device_list=device_list1)
        # model_B = GPT2LMHeadModel_modified.from_pretrained("gpt2", output_hidden_states=True,
        #                                                     resid_pdrop =dropout,
        #                                                     embd_pdrop  =dropout,
        #                                                     attn_pdrop  =dropout,
        #                                                     summary_first_dropout=dropout        )
        model_B = GPT2LMHeadModel_modified(config=lm_config2)
        model_B.set_variables(device=device2, split_into=split_into2, device_list=device_list2)
    elif cfg.model_size == "medium":
        lm_config1 = GPT2Config().from_pretrained('gpt2-medium')
        lm_config1.output_hidden_states = True
        lm_config1.resid_pdrop =dropout
        lm_config1.embd_pdrop  =dropout
        lm_config1.attn_pdrop  =dropout
        lm_config1.summary_first_dropout=dropout

        lm_config2 = GPT2Config().from_pretrained('gpt2-medium')
        lm_config2.output_hidden_states = True
        lm_config2.resid_pdrop =dropout
        lm_config2.embd_pdrop  =dropout
        lm_config2.attn_pdrop  =dropout
        lm_config2.summary_first_dropout=dropout
        # model_A = GPT2LMHeadModel_modified(config=lm_config1, device=device1, split_into=split_into)
        # model_B = GPT2LMHeadModel_modified(config=lm_config2, device=device2, split_into=split_into)
        # model_A = GPT2LMHeadModel_modified.from_pretrained("gpt2-medium", output_hidden_states=True,
        #                                                     resid_pdrop =dropout,
        #                                                     embd_pdrop  =dropout,
        #                                                     attn_pdrop  =dropout,
        #                                                     summary_first_dropout=dropout        )
        model_A = GPT2LMHeadModel_modified(config=lm_config1)
        model_A.set_variables(device=device1, split_into=split_into1, device_list=device_list1)
        # model_B = GPT2LMHeadModel_modified.from_pretrained("gpt2-medium", output_hidden_states=True,
        #                                                     resid_pdrop =dropout,
        #                                                     embd_pdrop  =dropout,
        #                                                     attn_pdrop  =dropout,
        #                                                     summary_first_dropout=dropout)
        model_B = GPT2LMHeadModel_modified(config=lm_config2)
        model_B.set_variables(device=device2, split_into=split_into2, device_list=device_list2)

   # pdb.set_trace()
    # load the model
    if cfg.model_size == "small":
        if cfg.use_old_model:
            model_A_states, model_B_states = torch.load(cfg.old_small_model_dir, map_location=device1)
            model_A_states['transformer.wte.weight'] = model_A_states['transformer.wte.weight'][:50257,:]
            model_A_states['lm_head.weight'] = model_A_states['lm_head.decoder.weight'][:50257,:]
            model_B_states['transformer.wte.weight'] = model_B_states['transformer.wte.weight'][:50257,:]
            model_B_states['lm_head.weight'] = model_B_states['lm_head.decoder.weight'][:50257,:]
            print("loaded old small model")
        else:
            model_A_states, model_B_states = torch.load(cfg.new_small_model_dir, map_location=device1)
            print("loaded new small model")
    elif cfg.model_size == "medium":
        if cfg.use_old_model:
            model_A_states, model_B_states = torch.load(cfg.old_medium_model_dir, map_location=device1)
            model_A_states['transformer.wte.weight'] = model_A_states['transformer.wte.weight'][:50257,:]
            model_A_states['lm_head.weight'] = model_A_states['lm_head.decoder.weight'][:50257,:]
            model_B_states['transformer.wte.weight'] = model_B_states['transformer.wte.weight'][:50257,:]
            model_B_states['lm_head.weight'] = model_B_states['lm_head.decoder.weight'][:50257,:]
            print("loaded old medium model")
        else:
            model_A_states, model_B_states = torch.load(cfg.new_medium_model_dir, map_location=device1)
            print("loaded new medium model")
    if model_A_dir is not None and model_A_dir != cfg.old_medium_model_dir:
        if use_old_model_B:
            model_A_states, _ = torch.load(model_A_dir)
            print("loaded RL-NEW model!!! model_A only")
            strict_A, strict_B = True, False
        else:
            model_A_states, model_B_states = torch.load(model_A_dir)
            print("loaded RL-NEW model!!! model_A and model_B")
            strict_A, strict_B = True, True
    else:
        strict_A = False
        strict_B = False
    model_A.load_state_dict(model_A_states, strict=strict_A)
    model_B.load_state_dict(model_B_states, strict=strict_B)

    del model_A_states, model_B_states
    torch.cuda.empty_cache()
    # to device
    model_A.to(device1)
    model_B.to(device2)
 
    return model_A, model_B

def load_GPT2(cfg, device1, split_into, device_list, dropout):
    if cfg.model_size == "small":
        lm_config1 = GPT2Config().from_pretrained('gpt2')
        lm_config1.output_hidden_states = True
        lm_config1.resid_pdrop =dropout
        lm_config1.embd_pdrop  =dropout
        lm_config1.attn_pdrop  =dropout
        lm_config1.summary_first_dropout=dropout

        # model_A = GPT2LMHeadModel_modified.from_pretrained("gpt2", output_hidden_states=True, 
        #                                                     resid_pdrop =dropout,
        #                                                     embd_pdrop  =dropout,
        #                                                     attn_pdrop  =dropout,
        #                                                     summary_first_dropout=dropout)
        model_A = GPT2LMHeadModel_modified(config=lm_config1)
        model_A.set_variables(device=device1, split_into=split_into, device_list=device_list)
    elif cfg.model_size == "medium":
        lm_config1 = GPT2Config().from_pretrained('gpt2-medium')
        lm_config1.output_hidden_states = True
        lm_config1.resid_pdrop =dropout
        lm_config1.embd_pdrop  =dropout
        lm_config1.attn_pdrop  =dropout
        lm_config1.summary_first_dropout=dropout
        # model_A = GPT2LMHeadModel_modified.from_pretrained("gpt2-medium", output_hidden_states=True,
        #                                                     resid_pdrop =dropout,
        #                                                     embd_pdrop  =dropout,
        #                                                     attn_pdrop  =dropout,
        #                                                     summary_first_dropout=dropout)
        model_A = GPT2LMHeadModel_modified(config=lm_config1)
        model_A.set_variables(device=device1, split_into=split_into, device_list=device_list)

    if cfg.model_size == "small":
        if cfg.use_old_model:
            model_A_states, model_B_states = torch.load(cfg.old_small_model_dir, map_location=device1)
            del model_B_states
            torch.cuda.empty_cache()
            model_A_states['transformer.wte.weight'] = model_A_states['transformer.wte.weight'][:50257,:]
            model_A_states['lm_head.weight'] = model_A_states['lm_head.decoder.weight'][:50257,:]
            print("loaded old small model")
        else:
            model_A_states, model_B_states = torch.load(cfg.new_small_model_dir, map_location=device1)
            print("loaded new small model")

        # model_A_states = torch.load("Checkpoint/original_GPT2_small.pth")
    elif cfg.model_size == "medium":
        if cfg.use_old_model:
            model_A_states, model_B_states = torch.load(cfg.old_medium_model_dir, map_location=device1)
            del model_B_states
            torch.cuda.empty_cache()
            model_A_states['transformer.wte.weight'] = model_A_states['transformer.wte.weight'][:50257,:]
            model_A_states['lm_head.weight'] = model_A_states['lm_head.decoder.weight'][:50257,:]
            print("loaded old medium model")
        else:
            model_A_states, model_B_states = torch.load(cfg.new_medium_model_dir, map_location=device1)
            print("loaded new medium model")
        # model_A_states = torch.load("Checkpoint/original_GPT2_medium.pth")

    model_A.load_state_dict(model_A_states, strict=False)
    del model_A_states
    torch.cuda.empty_cache()
    # to device
    model_A.to(device1)

    for param in model_A.parameters():
        param.requires_grad = False

    return model_A



# In[42]:


def distribute_collect(sequences, logprobs, pred_token, pred_logprob, working_indices, termination_token=2):
    """To support effecient batch text generation
        The algorithm automatically filters out generated samples.
    """
    terminate_list = []
    keep_list = []
    
    # loop over all results    
    for count, worker_index in enumerate(working_indices):
        sequences[worker_index].append(pred_token[count])
        logprobs[worker_index].append(pred_logprob[count])
        
        # filtering algorithm
        if pred_token[count] == termination_token:
            terminate_list.append(worker_index)
        else:
            keep_list.append(count)

    # remove from working_sent_indices
    if len(terminate_list) > 0:
        for worker_index in terminate_list:
            working_indices.remove(worker_index)

    # we would need both
    return keep_list, terminate_list


# In[43]:


import collections

class ReplayBuffer:
    """Simple Replay Buffer
    """
    # TODO Priority Replay Buffer 
    def __init__(self, maxlen=512):
        self.maxlen = maxlen
        self.buffer = collections.deque(maxlen=maxlen)
        
    def add(self, samples):
        self.buffer.extend(samples)

    def mean(self, calculate_original=True):
        if calculate_original:
            original_rewards = [item[-2] for item in self.buffer]
            return np.mean(original_rewards)
        else:
            normalized_rewards = [item[-3] for item in self.buffer]
            return np.mean(normalized_rewards)

    def std(self, calculate_original=True):
        original_rewards = [item[-2] for item in self.buffer]
        return np.std(original_rewards)

    def __getitem__(self, index):
        return self.buffer[index]

    def __len__(self):
        return len(self.buffer)
    
    def __repr__(self):
        return repr(self.buffer)

class CustomRewardFunc:
    """Give reward for the entire sequence
    """
    def __init__(self):
        self.ground_truth_reward = 10
        self.success_candidates_reward = 2
        self.backup_candidates_reward = 0.5
        self.failed_candidates_reward = -2
        self.long_candidate_penalty = -3
        self.len_denominator = float('Inf')
        self.len_cut = 50


    def __call__(self, sequences, have_enough_candidates, with_ground_truth):
        if with_ground_truth:
            ground_truth, sents_success, sents_failed = sequences
        else:
            sents_success, sents_failed = sequences

        rewards = []
        if have_enough_candidates:
            for sent in sents_success:
                if len(sent.split()) < self.len_cut:
                    rewards.append(self.success_candidates_reward + len(sent.split())/self.len_denominator)
                else:
                    rewards.append(self.long_candidate_penalty)
            for sent in sents_failed:
                if len(sent.split()) < self.len_cut:
                    rewards.append(self.failed_candidates_reward - len(sent.split())/self.len_denominator)
                else:
                    rewards.append(self.long_candidate_penalty)
            # rewards = [self.success_candidates_reward]*len(sents_success) + \
            #         [self.failed_candidates_reward]*len(sents_failed)                
        else:
            for sent in sents_success:
                if len(sent.split()) < self.len_cut:
                    rewards.append(self.backup_candidates_reward + len(sent.split())/self.len_denominator)
                else:
                    rewards.append(self.long_candidate_penalty)
            for sent in sents_failed:
                if len(sent.split()) < self.len_cut:
                    rewards.append(self.failed_candidates_reward - len(sent.split())/self.len_denominator)
                else:
                    rewards.append(self.long_candidate_penalty)

            # rewards = [self.backup_candidates_reward]*len(sents_success) + \
            #         [self.failed_candidates_reward]*len(sents_failed)
        

        if with_ground_truth:
            rewards = [self.ground_truth_reward + len(sent.split())/self.len_denominator] + rewards

        

        return rewards

        # for seq in sequences:
        #     seq_reward = seq.count(self.reward_token)
        #     rewards.append(seq_reward)



# import pickle as pkl
# with open("ppo_replay_buffer.pkl", "rb") as fh:
#     a = pkl.load(fh)
# import pdb
# pdb.set_trace()

class Actor(PersuasiveBot):
    """Text Generation
    """
    def __init__(self, model_A, model_B, tokenizer, device1, device2, dialog_i):
        super().__init__(model_A=model_A, model_B=model_B, tokenizer=tokenizer, device1=device1, device2=device2)
        
        train_data = torch.load("DataProcess/train_dialogs.pkl")
        val_data = torch.load("DataProcess/val_dialogs.pkl")
        self.train_dataset = PersuadeDataset(train_data, tokenizer)
        self.val_dataset = PersuadeDataset(val_data, tokenizer)
        self.dialog_i = dialog_i
        # batch_size = 1

        # self.train_dataloader = DataLoader(dataset=train_dataset, 
        #                             shuffle=True, 
        #                             batch_size=batch_size, 
        #                             collate_fn=train_dataset.collate)
        # self.val_dataloader = DataLoader(dataset=val_dataset, 
        #                             shuffle=False, 
        #                             batch_size=batch_size, 
        #                             collate_fn=train_dataset.collate)


        self.contexts = []
        self.reward_func = CustomRewardFunc()
        self.reload()

    def reload(self):
        super().reload()
        self.contexts = []

    def sample_generations(self, sample_size=1, mode=None):
        """
        the sample unit is one dialog, sample_size=1 means at a time we sample one dialog
        """
        self.model_A.eval()
        self.model_B.eval()
        
        final_contexts, final_sents, final_rewards, final_context_ids = [], [], [], []
        with torch.no_grad():
            for _ in range(sample_size):
                if mode is None:
                    if True: #self.dialog_i > 0:#True: #self.dialog_i > 0:
                        mode = np.random.choice([cfg.self_play_mode, cfg.supervised_mode], replace=False, 
                                                p=[PpoParams.self_play_prob, 1-PpoParams.self_play_prob])
                    else:
                        mode = cfg.self_play_mode
                if mode == cfg.self_play_mode:
                    pdb.set_trace()
                logger.info(f"in mode: {mode}")
                if mode == cfg.supervised_mode:
                    batch = self.train_dataset[np.random.choice(len(self.train_dataset))]
                    role_ids, dial_tokens, dial_sents = batch
                    dial_inputs = []
                    for item in dial_tokens:
                        if item[0] == 32:
                            dial_inputs.append(torch.LongTensor(item).unsqueeze(0).to(self.device1))
                        else:
                            dial_inputs.append(torch.LongTensor(item).unsqueeze(0).to(self.device2))

                    print(f"len: {len(role_ids)}")
                    NUM_SUCCESS_SENTS = 0
                    NUM_TURNS = 0
                    for role_id, dial_turn_inputs, dial_sent in zip(role_ids, dial_inputs, dial_sents):
                        print(f"turn #: {self.turn_i}\n\n\n")
                        # pdb.set_trace()
                        # if self.turn_i > 9:
                        #     break
                        # if dial_turn_inputs[0]
                        if role_id == 0:
                            if self.past is None:
                                user_text = ""
                            response, [sents_success, sents_failed], have_enough_candidates, usr_input_text = self.chat(input_text=user_text, mode=mode)
                            ground_truth = dial_sent
                            # logging
                            NUM_SUCCESS_SENTS += len(sents_success)
                            NUM_TURNS += 1
                            try:
                                assert not ground_truth.startswith("A:")
                            except:
                                pdb.set_trace()
                            cur_rewards = self.reward_func([ground_truth, sents_success, sents_failed], have_enough_candidates, with_ground_truth=True)

                            # print(f"truth: {ground_truth}")
                            # print(f"sent_success: \n{sents_success}")
                            # print(f"sent_failed: \n{sents_failed}")
                            # update
                            ground_truth_sents = sent_tokenize_modified(ground_truth)                    
                            sent_acts, _ = self.global_profile.regex_label(self.model_clf,
                                                                ground_truth_sents, 
                                                                which_task="A")
                            self.global_profile.update(sents=ground_truth_sents, sent_labels=sent_acts, who=self.domain.SYS) #self.last_sys_labels = self.sys_profile.update(sys_texts=sents, sys_labels=sent_acts)
                            
                            # pdb.set_trace()
                            try:
                                assert self.tokenizer.decode(dial_turn_inputs[0][:2].tolist()) == "A:"
                            except:
                                pdb.set_trace()
                            if self.past is not None and self.model_A.device != self.past[0].device:
                                past = [p.to(self.model_A.device) for p in self.past]
                                self.past = past
                            _, self.past, hidrden_states = self.model_A(dial_turn_inputs, past=self.past)
                            self.model_clf.set_past(sent=ground_truth, 
                                                    which_task="A")

                            # put in replay buffer
                            for sent, reward in zip([ground_truth] + sents_success + sents_failed, cur_rewards):
                                final_contexts.append(deepcopy(self.contexts))
                                final_sents.append("A:"+sent)
                                final_rewards.append(reward)
                                final_context_ids.append(f"{self.dialog_i}-{self.turn_i}-supervised")
                                # self.replay_buffer.add([deepcopy(self.contexts), "A:"+sent, reward])

                            # update contexts
                            logging.info(f"sys: {ground_truth}")
                            logging.info(f"success candidates: {sents_success}")
                            logging.info(f"failed candidates: {sents_failed}")
                            logging.info(f"----------------------")
                            self.contexts.append("A:"+ground_truth)

                        else:
                            # breakpoint()
                            user_text = dial_sent
                            try:
                                assert not user_text.startswith("B:")
                            except:
                                pdb.set_trace()
                            self.contexts.append("B:"+user_text)
                            logging.info(f"----------------------")
                            print(f"user: {user_text}")
                            logging.info(f"user: {user_text}")
                            # logits, past = model_B(dial_turn_inputs, past=past)
                            # all_logits.append(logits)

                    print(f"avg success sent: {NUM_SUCCESS_SENTS/NUM_TURNS}")
                    logger.info(f"avg success sent: {NUM_SUCCESS_SENTS/NUM_TURNS}")
                    # finish tail
                    if role_id == 1: # the last sent is user
                        # throw away the last user sentence
                        pass

                else:
                    # is_end = False
                    while True:
                        print(f"turn #: {self.turn_i}\n\n\n")
                        # pdb.set_trace()
                        if self.past is None:
                            user_text = ""
                        else:
                            user_text, user_texts_labels = self.generate_user_utt_self_play()                
                        # system-side
                            if "closing" in user_texts_labels or self.turn_i >=10: #\
                                # "bye" in user_text.lower() or "have a great day" in user_text.lower() \
                                # or "have a great night" in user_text.lower() \
                                # or "have a good day" in user_text.lower() \
                                # or "have a good night" in user_text.lower() \
                                # or "have a nice day" in user_text.lower() \
                                # or "have a nice night" in user_text.lower() \
                                # self.turn_i >= 10:
                                break

                            self.contexts.append("B:"+user_text)
                            print(f"user: {user_text}")
                            logging.info(f"user: {user_text}")
                        sys_sent, [sents_success, sents_failed], have_enough_candidates = self.sys_respond_and_update(mode=mode)
                        cur_rewards = self.reward_func([sents_success, sents_failed], have_enough_candidates, with_ground_truth=False)

                        # put in replay buffer
                        for sent, reward in zip(sents_success + sents_failed, cur_rewards):
                            final_contexts.append(deepcopy(self.contexts))
                            final_sents.append("A:"+sent)
                            final_rewards.append(reward)
                            final_context_ids.append(f"{self.dialog_i}-{self.turn_i}-selfplay")
                            # self.replay_buffer.add([deepcopy(self.contexts), "A:"+sent, reward])

                        # update contexts
                        self.contexts.append("A:"+sys_sent)
                        print(f"sys: {sys_sent}")
                        logging.info(f"sys: {sys_sent}")

                        turn_responses = ["usr: "+user_text,
                                        "sys: "+sys_sent]

                        self.logs['global_profiles'].append(self.global_profile.get_profiles())
                        self.logs['responses'].append(turn_responses)

                self.dialog_i += 1
                self.reload()
            assert len(final_contexts) == len(final_sents) == len(final_rewards) == len(final_context_ids)
            return final_contexts, final_sents, final_rewards, final_context_ids

class Trainer:
    """Reinforcement Learning Trainer
    """
    def __init__(self, actor, model_A, model_B, device1, device2, GPT2_model, use_approx_kl):
        self.use_approx_kl = use_approx_kl
        self.sample_size = 1 # num of dialog to sample at a time
        self.maxlen = 256
        if REPLAY_BUFFER_DIR is None:
            self.replay_buffer = ReplayBuffer(self.maxlen)
        else:
            self.replay_buffer = torch.load(REPLAY_BUFFER_DIR)
            # pdb.set_trace()
        self.actor = actor
        self.tokenizer = tokenizer
        self.turn_ending = [628, 198]
        self.model_A = model_A
        self.model_B = model_B
        self.device1 = device1
        self.device2 = device2
        assert self.model_A.device is self.device1
        assert self.model_B.device is self.device2

        self.GPT2 = GPT2_model

        self.kl_loss = KLDivLoss(reduction="sum")
        self.trained_steps = 0
        # self.reward_func = CustomRewardFunc(tokenizer)
        
    def collect_generations(self, total_size=64, normalize_reward=True):
        assert self.model_A.device is self.device1
        assert self.model_B.device is self.device2

        logger.info("Collecting Samples")
        # define storage
        all_contexts, all_sents, all_encoded_sents, all_rewards, all_context_ids = [], [], [], [], []
        
        while total_size > 0:
            assert self.model_A.device is self.device1
            assert self.model_B.device is self.device2

            real_sample_size = min(self.sample_size, total_size)
            
            # sample sequences
            contexts, sents, rewards, context_ids = self.actor.sample_generations(sample_size=real_sample_size)
            # actor.replay_buffer[0]
            # [[], 'A:Good morning, how are you this Sunday morning?', 2]
            
            import pdb
            # pdb.set_trace()
            all_contexts.extend(contexts)
            all_sents.extend(sents)
            encoded_sents = []
            for s in sents:
                if s.startswith("A:") or s.startswith("B:"):
                    encoded_sents.append(self.tokenizer.encode(s[:2]) + self.tokenizer.encode(s[2:]) + self.turn_ending)
                else:
                    pdb.set_trace()
            all_encoded_sents.extend(encoded_sents)
            all_rewards.extend(rewards)
            all_context_ids.extend(context_ids)

            # decrease
            total_size -= self.sample_size
            logger.info(f"{total_size} samples remaining")
        
        all_rewards = np.array(all_rewards)

        if normalize_reward:
            if len(self.replay_buffer) > 0:
                final_rewards = (all_rewards - self.replay_buffer.mean()) / (self.replay_buffer.std() + 1e-5)
            else:  
                final_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-5)
            # final_rewards = (all_rewards - replay_buffer.mean()) / (replay_buffer.std() + 1e-5)
        else:
            final_rewards = all_rewards
        
        # pdb.set_trace()
        assert len(all_contexts) == len(all_sents) == len(all_encoded_sents) == len(final_rewards) == len(all_context_ids) == len(all_rewards)
        self.replay_buffer.add(zip(all_contexts, all_sents, all_encoded_sents, final_rewards, all_rewards, all_context_ids))
        
        # logging
        logger.info(f"replay buffer mean: {self.replay_buffer.mean(calculate_original=False)}, {self.replay_buffer.std(calculate_original=False)}")
        logger.info(f"replay buffer mean: {self.replay_buffer.mean(calculate_original=False)}, {self.replay_buffer.std(calculate_original=False)}")
        logger.info("Collecting Samples finished!")
        
        return all_rewards

    # def kl_divergence(self, logits1, logits2):
    #     pdb.set_trace()
    #     probs = F.softmax(logits1, 2)
    #     start = time.time()
    #     kl = torch.where(probs == 0, torch.LongTensor(0).to(probs.device), probs * (F.log_softmax(logits1, 2) - F.log_softmax(logits2, 2))).mean()
    #     end1 = time.time()
    #     kl = (F.softmax(logits1, 2) * (F.log_softmax(logits1, 2) - F.log_softmax(logits2, 2))).mean()
    #     end2 = time.time()
    #     print(f"{end1-start}")
    #     print(f"{end2-end1}")
    #     return kl

    def calculate_old_logprobs(self, buffer_contexts, buffer_context_ids, buffer_sents, buffer_encoded_sents):
        assert self.model_A.device is self.device1
        assert self.model_B.device is self.device2

        start = time.time()
        buffer_old_logprobs = [None] * len(buffer_sents)
        buffer_old_logprobs_gpt2 = [None] * len(buffer_sents)
        buffer_old_logits = [None] * len(buffer_sents)
        buffer_old_logits_gpt2 = [None] * len(buffer_sents)
        indices = np.arange(len(buffer_sents))
        context_map = {}

        with torch.no_grad():
            for i, context_id in enumerate(buffer_context_ids):
                if context_id in context_map:
                    context_map[context_id]['sent_ids'].append(i)
                else:
                    context_map[context_id] = {#'past': None,
                                                'contexts': buffer_contexts[i],
                                                'sent_ids': [i]}
                    # contexts = buffer_contexts[i]

            for context_id in context_map:
                batch_sents = [buffer_encoded_sents[j] for j in context_map[context_id]['sent_ids']]
                batch_sents = make_batch_sequences(batch_sents, padding_value=PAD_TOKEN)
                mask = batch_sents.ne(PAD_TOKEN).float()
                # to device
                batch_sents = batch_sents.to(self.device1)
                mask = mask.to(self.device1)

                # calculate past
                past = self.make_past(context_map[context_id]['contexts']) 
                # pdb.set_trace()               
                if past is not None:
                    past = [p.repeat(1, batch_sents.shape[0], 1, 1, 1).to(self.model_A.device) for p in past]
                # if past is not None and self.model_A.device != past[0].device:
                #     past = [p.to(self.model_A.device) for p in past]
                
                logits, past, hidden_states = self.model_A(batch_sents, past=past)
                logits_gpt2, _, _ = self.GPT2(batch_sents, past=None)


                # prepare the loss func inputs
                logits = logits[:, :-1].contiguous()
                logits_gpt2 = logits_gpt2[:, :-1].contiguous()
                logits_gpt2 = logits_gpt2.to(logits.device)
                target = batch_sents[:, 1:].contiguous()
                mask = mask[:, 1:].contiguous()

                sequences_logprobs = - criterion(logits, target, mask)
                old_logprobs = sequences_logprobs.sum(1)
                
                #gpt2 logprobs
                sequences_logprobs_gpt2 = - criterion(logits_gpt2, target, mask)
                old_logprobs_gpt2 = sequences_logprobs_gpt2.sum(1)

                # store
                # pdb.set_trace()
                old_logprobs = old_logprobs.tolist()
                old_logprobs_gpt2 = old_logprobs_gpt2.tolist()
                if not self.use_approx_kl:
                    # sequences_logprobs = sequences_logprobs.tolist()
                    # sequences_logprobs_gpt2 = sequences_logprobs_gpt2.tolist()
                    logits = logits.cpu()
                    logits_gpt2 = logits_gpt2.cpu() # save gpu space
                # pdb.set_trace()
                for i, j in enumerate(context_map[context_id]['sent_ids']):
                    buffer_old_logprobs[j] = old_logprobs[i]
                    buffer_old_logprobs_gpt2[j] = old_logprobs_gpt2[i]
                    if not self.use_approx_kl:
                        buffer_old_logits[j] = logits[i]
                        buffer_old_logits_gpt2[j] = logits_gpt2[i]
        end = time.time()
        speed = (end - start) / len(buffer_sents)

        print(f"calculate_old_logprobs: {speed} per turn")
        logger.info(f"calculate_old_logprobs: {speed} per turn")
        # pdb.set_trace()
        return buffer_old_logprobs, buffer_old_logprobs_gpt2, buffer_old_logits, buffer_old_logits_gpt2


    def calculate_old_logprobs_GPT2(self, buffer_sents, buffer_sequences):

        buffer_old_logprobs = []
        assert len(buffer_sents) == len(buffer_sequences)
        indices = np.arange(len(buffer_sequences))

        with torch.no_grad():
            for i in range((len(buffer_sequences) // PpoParams.batchsize) + 1):
                batch_indices = indices[i * PpoParams.batchsize : (i + 1) * PpoParams.batchsize]
                batch_sequences = [buffer_sequences[j] for j in batch_indices]
                if len(batch_sequences) == 0:
                    assert (len(buffer_sequences) % PpoParams.batchsize) == 0
                    break
                
                # pdb.set_trace()
                # make batch
                batch_sequences = make_batch_sequences(batch_sequences, padding_value=PAD_TOKEN)
                mask = batch_sequences.ne(PAD_TOKEN).float()

                # to device
                device = self.GPT2.device
                batch_sequences = batch_sequences.to(device)
                mask = mask.to(device)

                logits, past, _ = self.GPT2(batch_sequences)

                # prepare the loss func inputs
                logits = logits[:, :-1].contiguous()
                target = batch_sequences[:, 1:].contiguous()
                mask = mask[:, 1:].contiguous()

                sequences_logprobs = - criterion(logits, target, mask)
                old_logprobs = sequences_logprobs.sum(1)

                # store
                buffer_old_logprobs.extend(old_logprobs.tolist())
        # pdb.set_trace()
        assert len(buffer_old_logprobs) == len(buffer_sents)
        return buffer_old_logprobs

    def make_past(self, contexts):
        past = None
        for context in contexts:
            try:
                assert context.startswith("A:") or context.startswith("B:")
            except:
                pdb.set_trace()
            encoded_context = self.tokenizer.encode(context[:2]) + self.tokenizer.encode(context[2:]) + self.turn_ending
            if context.startswith("A:"):
                encoded_context = torch.LongTensor(encoded_context).unsqueeze(0).to(self.device1)
                if past is not None and self.model_A.device != past[0].device:
                    past = [p.to(self.model_A.device) for p in past]
                logits, past, _ = self.model_A(encoded_context, past=past)
            elif context.startswith("B:"):
                encoded_context = torch.LongTensor(encoded_context).unsqueeze(0).to(self.device2)
                if past is not None and self.model_B.device != past[0].device:
                    past = [p.to(self.model_B.device) for p in past]
                logits, past, _ = self.model_B(encoded_context, past=past)
            else:
                raise ValueError(f"context: {context}")
        
        return past

    def train_steps(self, total_steps):
        for total_step in tqdm(range(total_steps)):
            self.trained_steps += 1
            start = time.time()
            
            if not DEBUG:
                all_rewards = trainer.collect_generations(total_size=PpoParams.num_dialogs_to_sample)
            else:
                all_rewards = [r[-2] for r in self.replay_buffer]

            # self.model_A.train()
            # self.model_B.train()
            buffer_contexts, buffer_sents, buffer_encoded_sents, buffer_rewards, buffer_no_normalized_rewards, buffer_context_ids = zip(*self.replay_buffer)
            buffer_old_logprobs, buffer_old_logprobs_gpt2, buffer_old_logits, buffer_old_logits_gpt2 = self.calculate_old_logprobs(buffer_contexts, buffer_context_ids, buffer_sents, buffer_encoded_sents)
            # these are lists of floats
            # buffer_old_logprobs_gpt2 = self.calculate_old_logprobs_GPT2(buffer_sents, buffer_encoded_sents)
            # pdb.set_trace()
            
            
            for ppo_epoch in tqdm(range(PpoParams.ppo_epoch)):
                indices = np.arange(len(buffer_rewards))
                np.random.shuffle(indices)
                
                optimizer.zero_grad()
                for i in range(PpoParams.batchsize // PpoParams.mini_batchsize):
                    sampled_indices = indices[i * PpoParams.mini_batchsize : (i + 1) * PpoParams.mini_batchsize]
                    if len(sampled_indices) == 0:
                        break
                    sampled_contexts = [buffer_contexts[j] for j in sampled_indices]
                    sampled_sents = [buffer_sents[j] for j in sampled_indices]
                    sampled_encoded_sents = [buffer_encoded_sents[j] for j in sampled_indices]
                    sampled_old_logprobs = [buffer_old_logprobs[j] for j in sampled_indices]
                    sampled_old_logprobs_gpt2 = [buffer_old_logprobs_gpt2[j] for j in sampled_indices]
                    sampled_rewards = [buffer_rewards[j] for j in sampled_indices]
                    sampled_old_logits = [buffer_old_logits[j] for j in sampled_indices]
                    sampled_old_logits_gpt2 = [buffer_old_logits_gpt2[j] for j in sampled_indices]
                    # make batches
                    # logits_list = []
                    # target_list = []
                    # mask_list = []
                    # torch.cuda.empty_cache()
                    # pdb.set_trace()

                    # sequence_logprob_list = []
                    try:
                        batch_encoded_sents = make_batch_sequences(sampled_encoded_sents, padding_value=PAD_TOKEN)
                    except:
                        pdb.set_trace()
                    
                    optimizer.zero_grad()
                    # logging purpose:
                    approx_kl_list, accurate_kl_list, approx_kl_gpt2_list, accurate_kl_gpt2_list, policy_loss_list = [], [], [], [], []
                    for i, (contexts, encoded_sent, reward, old_logprob, old_logprob_gpt2, old_logits, old_logits_gpt2) in enumerate(zip(sampled_contexts, sampled_encoded_sents, sampled_rewards, sampled_old_logprobs, sampled_old_logprobs_gpt2, sampled_old_logits, sampled_old_logits_gpt2)):
                        past = self.make_past(contexts)
                        # encoded_sent = batch_encoded_sents[i, :].unsqueeze(0)
                        encoded_sent = torch.LongTensor(encoded_sent).unsqueeze(0)
                        # pdb.set_trace()
                        # encoded_sent = make_batch_sequences([encoded_sent], padding_value=PAD_TOKEN)
                        # old_logprob = torch.FloatTensor([old_logprob])
                        # reward = torch.FloatTensor([reward])
                        mask = encoded_sent.ne(PAD_TOKEN).float()

                        # to device
                        encoded_sent = encoded_sent.to(self.device1)
                        # old_logprob = old_logprob.to(device)
                        # reward = reward.to(device)
                        mask = mask.to(self.device1)
                        if past is not None and self.model_A.device != past[0].device:
                            past = [p.to(self.model_A.device) for p in past]

                        logits, past, hidden_states = self.model_A(encoded_sent, past=past)

                        # prepare the loss func inputs
                        logits = logits[:, :-1].contiguous()
                        target = encoded_sent[:, 1:].contiguous()
                        mask = mask[:, 1:].contiguous()

                        sequences_logprob = - criterion(logits, target, mask)
                        # sequence_logprob_list.append(sequences_logprob)
                        # if np.isnan(logits.item()):
                        #     pdb.set_trace()

                        loss, cur_approx_kl, cur_accurate_kl, cur_approx_kl_gpt2, cur_accurate_kl_gpt2, cur_policy_loss = self.calculate_loss(sequences_logprob, old_logprob, old_logprob_gpt2, reward, 
                                                   logits, old_logits, old_logits_gpt2, normalize_over_length=True)
                        # pdb.set_trace()
                        loss.backward()     
                        # pdb.set_trace()                   
                        # logging
                        approx_kl_list.append(cur_approx_kl)
                        accurate_kl_list.append(cur_accurate_kl)
                        approx_kl_gpt2_list.append(cur_approx_kl_gpt2)
                        accurate_kl_gpt2_list.append(cur_accurate_kl_gpt2)
                        policy_loss_list.append(cur_policy_loss)
                        # loss = policy_loss + (sequences_logprob, kl)
                        # loss.backward()
                        # optimizer
                        # loss_list.append(loss)
                        # logits_list.append(logits)
                        # target_list.append(target)
                        # mask_list.append(mask)
                    # del past
                    # torch.cuda.empty_cache()
                    torch.nn.utils.clip_grad_norm_(self.model_A.parameters(), 0.5)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    """    
                    #(minibatch_size, 1)
                    old_logprobs = torch.FloatTensor(sampled_old_logprobs).to(self.device1)
                    old_logprobs_gpt2 = torch.FloatTensor(sampled_old_logprobs_gpt2).to(self.device1)
                    sampled_rewards = torch.FloatTensor(sampled_rewards).to(self.device1)

                    # calc advantages
                    advantages = sampled_rewards
                    advantages = advantages.clamp(-2, 2)
                    # clip(advantages, 2, -2)

                    # pdb.set_trace()
                    # (minibatch, logprob)
                    sequences_logprobs = torch.cat(sequence_logprob_list)#- criterion(logits, target, mask)
                    # del sequence_logprob_list

                    if USE_ENTROPY:
                        # here we need to calculate for the each token in the sequence
                        entropy = - (sequences_logprobs.exp() * sequences_logprobs).sum(1)
                        # print(f"entropy: {entropy}")
                        logger.info(f"entropy: {entropy}")
                        # entropy = entropy.clamp_min(min_entropy)
                    
                    logprobs = sequences_logprobs.sum(1) #@@ normalize by length

                    # shape: (batch)
                    ratio = (logprobs - old_logprobs).exp()
                    # shape: (batch)
                    policy_loss1 = - advantages * ratio
                    # shape: (batch)
                    policy_loss2 = - advantages * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
                    # shape: (batch)
                    policy_loss = torch.max(policy_loss1, policy_loss2)

                    # recording variables. no gradient!
                    with torch.no_grad():
                        clipfrac = ((ratio - 1.0).abs() > clip_range).float().mean()
                        approx_kl = (logprobs - old_logprobs).pow(2).mean()
                        # kl = sequences_logprob * (logprobs - old_logprobs).mean()#.pow(2).mean()
                        # kl(p, q) = p*log(p/q) = p(logp-log q)
                        # np.where(p != 0, p * np.log(p / q), 0)

                    # calculate KL with original gpt2
                    if not USE_ENTROPY:
                        approx_kl_gpt2 = (logprobs - old_logprobs_gpt2).pow(2)
                        # np.where(p != 0, p * np.log(p / q), 0)
                    
                    # print the final clipfrac and approxl
                    print(f"Approx KL {approx_kl.item()}, Clip Frac {clipfrac.item()}")
                    logger.info(f"Approx KL {approx_kl.item()}, Clip Frac {clipfrac.item()}")
                    if np.isnan(approx_kl.item()):
                        pdb.set_trace()

                    # get the final loss
                    # loss = policy_loss.mean() - entropy_coef * entropy.mean()
                    loss = policy_loss.mean() + kl_gpt2_coef * approx_kl_gpt2.mean()
                    logger.info(f"policy_loss {policy_loss.mean().item()}")
                    if policy_loss.mean().item() > 1e20:
                        pdb.set_trace()
                    if USE_ENTROPY:
                        logger.info(f"entropy {entropy.mean().item()}")
                    else:
                        logger.info(f"approx_kl_gpt2 {approx_kl_gpt2.mean().item()}")
                    logger.info(f"loss {loss.item()}")

                    # update the model
                    optimizer.zero_grad()
                    loss.backward()
                    # must clip the gradient for stability
                    # pdb.set_trace()
                    torch.nn.utils.clip_grad_norm_(self.model_A.parameters(), 0.5)
                    # pdb.set_trace() #assert not torch.isnan(grad).any()
                    optimizer.step()
                    
                    # scheduler.step()
                    """

            del past
            torch.cuda.empty_cache() 
            mean_reward = np.mean([r for r in all_rewards if r != self.actor.reward_func.ground_truth_reward])
            print("Mean Reward", mean_reward)
            logger.info(f"Mean Reward: {mean_reward}")
            end = time.time()
            speed = (end - start) / PpoParams.batchsize
            print("Speed", speed)
            logger.info(f"Speed: {speed}")

            print(f"max memory A: {torch.cuda.max_memory_allocated(model_A.device)}")
            logger.info(f"max memory A: {torch.cuda.max_memory_allocated(model_A.device)}")
            print(f"max memory B: {torch.cuda.max_memory_allocated(model_B.device)}")
            logger.info(f"max memory B: {torch.cuda.max_memory_allocated(model_B.device)}")
            if self.use_approx_kl:
                model_name = f"Checkpoint/{self.trained_steps+PREVIOUS_STEPS}_steps_{mean_reward}_reward_model_A_kl_{round(np.mean(approx_kl_gpt2_list), 2)}.pth"
            else:
                model_name = f"Checkpoint/{self.trained_steps+PREVIOUS_STEPS}_steps_{mean_reward}_reward_model_A_kl_{round(np.mean(accurate_kl_gpt2_list), 2)}.pth"
            torch.save((self.model_A.state_dict(), self.model_B.state_dict()), model_name)
            torch.save(self.replay_buffer, f"Checkpoint/replay_buffer.pth")

    def calculate_loss(self, sequences_logprob, old_logprob, old_logprob_gpt2, reward,
                       logits, old_logits, old_logits_gpt2, normalize_over_length):
        #(1, 1)
        # pdb.set_trace()
        outputs = []
        old_logprob = torch.FloatTensor([old_logprob]).to(self.device1)
        old_logprob_gpt2 = torch.FloatTensor([old_logprob_gpt2]).to(self.device1)
        reward = torch.FloatTensor([reward]).to(self.device1)
        if not self.use_approx_kl:
            old_logits = old_logits[:logits.shape[1]].unsqueeze(0).to(logits.device)
            old_logits_gpt2 = old_logits_gpt2[:logits.shape[1]].unsqueeze(0).to(logits.device)

        # calc advantages
        advantage = reward
        advantage = advantage.clamp(-2, 2)
        # clip(advantages, 2, -2)

        # # pdb.set_trace()
        # # (minibatch, logprob)
        # sequences_logprobs = torch.cat(sequence_logprob_list)#- criterion(logits, target, mask)
        # # del sequence_logprob_list

        if USE_ENTROPY:
            # here we need to calculate for the each token in the sequence
            entropy = - (sequences_logprob.exp() * sequences_logprob).sum(1)
            # print(f"entropy: {entropy}")
            logger.info(f"entropy: {entropy}")
            # entropy = entropy.clamp_min(min_entropy)
        
        logprob = sequences_logprob.sum(1) #@@ normalize by length
        if normalize_over_length:
            # pdb.set_trace()
            sequence_length = sequences_logprob.shape[1]
            logprob = logprob/sequence_length
            old_logprob = old_logprob/sequence_length
            old_logprob_gpt2 = old_logprob/sequence_length

        # shape: (batch)
        ratio = (logprob - old_logprob).exp()
        print(f"ratio 1: {ratio.item()}")
        logger.info(f"ratio 1: {ratio.item()}")
        # with torch.no_grad():
        #     ratio2 = (((logprob - old_logprob)*sequence_length).exp()/sequence_length).item()
        #     print(f"ratio 2: {ratio2}")
        # shape: (batch)
        policy_loss1 = - advantage * ratio
        # shape: (batch)
        policy_loss2 = - advantage * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
        # shape: (batch)
        policy_loss = torch.max(policy_loss1, policy_loss2)

        # recording variables. no gradient!
        with torch.no_grad():
            clipfrac = ((ratio - 1.0).abs() > clip_range).float().mean()
            approx_kl = (logprob - old_logprob).pow(2).mean()
            if not self.use_approx_kl:
                accurate_kl = self.kl_loss(F.log_softmax(old_logits, 2), F.softmax(logits, 2))
                # accurate_kl = self.kl_divergence(logits, old_logits)
            # kl = sequences_logprob * (logprobs - old_logprobs).mean()#.pow(2).mean()
            # kl(p, q) = p*log(p/q) = p(logp-log q)
            # np.where(p != 0, p * np.log(p / q), 0)

        # calculate KL with original gpt2
        if not USE_ENTROPY:
            
            if not self.use_approx_kl:
                with torch.no_grad():
                    approx_kl_gpt2 = (logprob - old_logprob_gpt2).pow(2)
                # accurate_kl_gpt2 = self.kl_divergence(logits, old_logits_gpt2)
                accurate_kl_gpt2 = self.kl_loss(F.log_softmax(old_logits_gpt2, 2), F.softmax(logits, 2))
                # accurate_kl_gpt2 = (F.softmax(logits) * (F.log_softmax(logits) - F.log_softmax(old_logits_gpt2))).mean()
            else:
                approx_kl_gpt2 = (logprob - old_logprob_gpt2).pow(2)
            # np.where(p != 0, p * np.log(p / q), 0)
        
        # print the final clipfrac and approxl
        print(f"Approx KL {approx_kl.item()}, Clip Frac {clipfrac.item()}")
        logger.info(f"Approx KL {approx_kl.item()}, Clip Frac {clipfrac.item()}")
        print(f"approx_kl_gpt2 {approx_kl_gpt2.mean().item()}")
        logger.info(f"approx_kl_gpt2 {approx_kl_gpt2.mean().item()}")
        if not self.use_approx_kl:
            print(f"accurate KL {accurate_kl.item()}")
            logger.info(f"accurate KL {accurate_kl.item()}")
            print(f"accurate KL GPT2 {accurate_kl_gpt2.item()}")
            logger.info(f"accurate KL GPT2 {accurate_kl_gpt2.item()}")
        if np.isnan(approx_kl.item()):
            pdb.set_trace()

        # get the final loss
        # loss = policy_loss.mean() - entropy_coef * entropy.mean()
        if self.use_approx_kl:
            loss = policy_loss.mean() + kl_gpt2_coef * approx_kl_gpt2.mean()
        else:
            loss = policy_loss.mean() + kl_gpt2_coef * accurate_kl_gpt2.mean()
        logger.info(f"policy_loss {policy_loss.mean().item()}")
        if policy_loss.mean().item() > 1e20:
            pdb.set_trace()
        if USE_ENTROPY:
            logger.info(f"entropy {entropy.mean().item()}")
        else:
            logger.info(f"approx_kl_gpt2 {approx_kl_gpt2.mean().item()}")
        logger.info(f"loss {loss.item()}")

        if not self.use_approx_kl:
            return loss, approx_kl.item(), accurate_kl.item(), approx_kl_gpt2.mean().item(), accurate_kl_gpt2.item(), policy_loss.mean().item()
        else:
            return loss, approx_kl.item(), None, approx_kl_gpt2.mean().item(), None, policy_loss.mean().item()

if __name__ == "__main__":
    NEW_MODEL_A_DIR = None#"Checkpoint/7_steps_2.3_reward_model_A_kl_47.13.pth"#None#"Checkpoint/9_steps_1.3272727272727274_reward_model_A.pth"#None#"Checkpoint/7_steps_1.984710743801653_reward_model_A.pth"#None#"Checkpoint/30_steps_2.0_reward_model_A.pth"#None#"Checkpoint/9_steps_-0.03278688524590164_reward_model_A.pth"#None#"Checkpoint/1_steps_1.12_reward_model_A.pth"#None#"Checkpoint/20_steps_0.049586776859504134_reward_model_A.pth"
    REPLAY_BUFFER_DIR = None#"Checkpoint/replay_buffer_in_exception.pth"#None#"Checkpoint/replay_buffer_in_exception.pth"#None#"Checkpoint/replay_buffer.pth"#None#"Checkpoint/replay_buffer.pth"#None#"Checkpoint/replay_buffer.pth"#None#"Checkpoint/replay_buffer.pth"#"Checkpoint/replay_buffer.pth"#None#"Checkpoint/replay_buffer.pth"
    DEBUG = False
    USE_APPROX_KL = False
    PREVIOUS_STEPS = 0
    TOTAL_STEPS = 50

    if REPLAY_BUFFER_DIR is not None:
        replay_buffer_temp = torch.load(REPLAY_BUFFER_DIR)
        PREV_DIALOGS = max([int(r[-1].split('-')[0]) for r in replay_buffer_temp])+1
    else:
        PREV_DIALOGS = 0
    print(f"{PREV_DIALOGS}")
    # pdb.set_trace()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")#torch.load(tokenizer_dir)
    DEVICE1 = torch.device(cfg.model_A_device)
    DEVICE1_list = cfg.model_A_device_list
    SPLIT_INTO1= cfg.split_into_A

    DEVICE2 = torch.device(cfg.model_B_device)
    DEVICE2_list = cfg.model_B_device_list
    SPLIT_INTO2= cfg.split_into_B
    
    DEVICE3 = cfg.model_C_device
    DEVICE3_list = cfg.model_C_device_list
    # DEVICE3 = torch.device(cfg.model_GPT2_device)
    SPLIT_INTO3 = cfg.split_into_C

    model_A, model_B = load_model(cfg=cfg, device1=DEVICE1, device2=DEVICE2, split_into1=SPLIT_INTO1,  split_into2=SPLIT_INTO2,
                                dropout=0,
                                model_A_dir=NEW_MODEL_A_DIR,
                                device_list1=DEVICE1_list,
                                device_list2=DEVICE2_list)
    # pdb.set_trace()
    GPT2_model = load_GPT2(cfg=cfg, device1=DEVICE3, split_into=SPLIT_INTO3, 
                            dropout=0, device_list=DEVICE3_list)
    # pdb.set_trace()
    PAD_TOKEN = tokenizer.encoder["<|endoftext|>"]
    USE_ENTROPY = False
    clip_range = 0.2
    entropy_coef = 1e-2
    kl_gpt2_coef = 1e-2
    min_entropy = 10.0 # depends on the task
    criterion = SequenceCrossEntropyLoss()


    actor = Actor(model_A=model_A, model_B=model_B, tokenizer=tokenizer, 
                  device1=DEVICE1, device2=DEVICE2, dialog_i=PREV_DIALOGS)

    # pdb.set_trace()
    # optimizer
    num_epochs = 10
    num_gradients_accumulation = 1
    num_train_optimization_steps = 1000

    param_optimizer = list(model_A.named_parameters()) + list(model_B.named_parameters())
    no_decay = ['ln', 'bias', 'LayerNorm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]


    num_train_optimization_steps = PpoParams.ppo_epoch * (PpoParams.batchsize // PpoParams.mini_batchsize) * TOTAL_STEPS

    from pytorch_pretrained_bert import OpenAIAdam    
    # optimizer = OpenAIAdam(optimizer_grouped_parameters,
    #                     lr=2e-5,
    #                     warmup=0.1,
    #                     max_grad_norm=1.0,
    #                     weight_decay=0.01,
    #                     t_total=num_train_optimization_steps)
    # from fairseq.optim.adafactor import Adafactor
    # optimizer = Adafactor(optimizer_grouped_parameters,
    #                       lr=2e-5,
    #                       clip_threshold=1,
    #                       )
    # from transformers import get_linear_schedule_with_warmup
    optimizer = AdamW(optimizer_grouped_parameters,
                    lr=3e-5,
                    eps=1e-06)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                     num_warmup_steps=100,
                                     num_training_steps=num_train_optimization_steps)
    # optimizer = FusedAdam(optimizer_grouped_parameters, 
    #                     lr=1e-6,
    #                     eps=1e-06,
    #                     bias_correction=False)







    IN_TRAINING = True

    trainer = Trainer(actor=actor, model_A=model_A, model_B=model_B, device1=DEVICE1, device2=DEVICE2,
                      GPT2_model=GPT2_model, use_approx_kl=USE_APPROX_KL)

    if IN_TRAINING:
        try:
            trainer.train_steps(total_steps=TOTAL_STEPS)
            # print(torch.cuda.memory_summary(device=model_A.device, abbreviated=False))
            # print(torch.cuda.memory_summary(device=model_B.device, abbreviated=False))

        except:
            torch.save((trainer.model_A.state_dict(), trainer.model_B.state_dict()), f"Checkpoint/in_exception_{trainer.trained_steps}_steps_reward_model_A.pth")
            torch.save(trainer.replay_buffer, f"Checkpoint/replay_buffer_in_exception.pth")
            print(torch.cuda.memory_summary(device=model_A.device, abbreviated=False))
            print(torch.cuda.memory_summary(device=model_B.device, abbreviated=False))
    else:
        # assert NEW_MODEL_A_DIR
        usr_text = ""
        while True:
            # interactive test
            sys_sent, [sents_success, sents_failed], have_enough_candidates, usr_input_text = actor.chat(usr_text)
            print(f"sys: {sys_sent}\n")
            # print(f"success:\n {sents_success}")
            # print(f"failed:\n {sents_failed}")

            usr_text = input("usr: ")
    # pdb.set_trace()





