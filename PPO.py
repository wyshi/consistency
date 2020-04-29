#!/usr/bin/env python
# coding: utf-8
# logging is important
import logging
logging.basicConfig(filename='ppo1.log', level=logging.INFO)
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
from PersuasionInteract import PersuasiveBot
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
from transformers import AdamW
# from torchfly.transformers import UnifiedTokenizer, GPT2SimpleLM
from torchfly.modules.losses import SequenceFocalLoss, SequenceCrossEntropyLoss
# from torchfly.decode import top_filtering


# In[38]:



class PpoParams:
    ppo_epoch = 2
    num_dialogs_to_sample = 1 # number of dialog in one batch
    batchsize = 128 
    mini_batchsize = 8
    self_play_prob = 0.2

def make_batch_sequences(sampled_sequences, type_func=torch.LongTensor, padding_value=1):
    # transform into LongTensor
    sampled_sequences = [type_func(item) 
                        for item in sampled_sequences 
                        if not isinstance(item,  type_func)]
    
    batch_sequences = nn.utils.rnn.pad_sequence(sampled_sequences, 
                                                batch_first=True, 
                                                padding_value=padding_value)
    return batch_sequences


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

def load_model(cfg, device1, device2, split_into, model_A_dir=None):
    if cfg.model_size == "small":
        # lm_config1 = GPT2Config().from_pretrained('gpt2')
        # lm_config2 = GPT2Config().from_pretrained('gpt2')
        # lm_config1.output_hidden_states = True
        # lm_config2.output_hidden_states = True
        # model_A = GPT2LMHeadModel_modified(config=lm_config1, device=device1, split_into=split_into)
        # model_B = GPT2LMHeadModel_modified(config=lm_config2, device=device2, split_into=split_into)
        model_A = GPT2LMHeadModel_modified.from_pretrained("gpt2", output_hidden_states=True)
        model_A.set_variables(device=device1, split_into=split_into)
        model_B = GPT2LMHeadModel_modified.from_pretrained("gpt2", output_hidden_states=True)
        model_B.set_variables(device=device2, split_into=split_into)
    elif cfg.model_size == "medium":
        # lm_config1 = GPT2Config().from_pretrained('gpt2-medium')
        # lm_config2 = GPT2Config().from_pretrained('gpt2-medium')
        # lm_config1.output_hidden_states = True
        # lm_config2.output_hidden_states = True
        # model_A = GPT2LMHeadModel_modified(config=lm_config1, device=device1, split_into=split_into)
        # model_B = GPT2LMHeadModel_modified(config=lm_config2, device=device2, split_into=split_into)
        model_A = GPT2LMHeadModel_modified.from_pretrained("gpt2-medium", output_hidden_states=True)
        model_A.set_variables(device=device1, split_into=split_into)
        model_B = GPT2LMHeadModel_modified.from_pretrained("gpt2-medium", output_hidden_states=True)
        model_B.set_variables(device=device2, split_into=split_into)

   # pdb.set_trace()
    # load the model
    if cfg.model_size == "small":
        model_A_states, model_B_states = torch.load(cfg.new_small_model_dir)
    elif cfg.model_size == "medium":
        if cfg.use_old_model:
            model_A_states, model_B_states = torch.load(cfg.old_medium_model_dir)
            model_A_states['transformer.wte.weight'] = model_A_states['transformer.wte.weight'][:50257,:]
            model_A_states['lm_head.weight'] = model_A_states['lm_head.decoder.weight'][:50257,:]
            model_B_states['transformer.wte.weight'] = model_B_states['transformer.wte.weight'][:50257,:]
            model_B_states['lm_head.weight'] = model_B_states['lm_head.decoder.weight'][:50257,:]
            print("loaded old medium model")
        else:
            model_A_states, model_B_states = torch.load(cfg.new_medium_model_dir)

    if model_A_dir is not None:
        model_A_states = torch.load(model_A_dir)
        print("loaded RL-NEW model!!!")
    if cfg.use_old_model:
        strict = False
    else:
        strict = True
    model_A.load_state_dict(model_A_states, strict=strict)
    model_B.load_state_dict(model_B_states, strict=strict)

    # to device
    model_A.to(device1)
    model_B.to(device2)
 
    return model_A, model_B



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
        self.ground_truth_reward = 2
        self.success_candidates_reward = 2
        self.backup_candidates_reward = 1
        self.failed_candidates_reward = -2


    def __call__(self, sequences, have_enough_candidates, with_ground_truth):
        if with_ground_truth:
            ground_truth, sents_success, sents_failed = sequences
        else:
            sents_success, sents_failed = sequences

        if have_enough_candidates:
            rewards = [self.success_candidates_reward]*len(sents_success) + \
                    [self.failed_candidates_reward]*len(sents_failed)                
        else:
            rewards = [self.backup_candidates_reward]*len(sents_success) + \
                    [self.failed_candidates_reward]*len(sents_failed)
        
        if with_ground_truth:
            rewards = [self.ground_truth_reward] + rewards

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
    def __init__(self, model_A, model_B, tokenizer, device1, device2):
        super().__init__(model_A=model_A, model_B=model_B, tokenizer=tokenizer, device1=device1, device2=device2)
        
        train_data = torch.load("DataProcess/train_dialogs.pkl")
        val_data = torch.load("DataProcess/val_dialogs.pkl")
        self.train_dataset = PersuadeDataset(train_data, tokenizer)
        self.val_dataset = PersuadeDataset(val_data, tokenizer)
        self.dialog_i = 0
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
                    if self.dialog_i > 0:#True: #self.dialog_i > 0:
                        mode = np.random.choice([cfg.self_play_mode, cfg.supervised_mode], replace=False, 
                                                p=[PpoParams.self_play_prob, 1-PpoParams.self_play_prob])
                    else:
                        mode = cfg.self_play_mode
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
                    for role_id, dial_turn_inputs, dial_sent in zip(role_ids, dial_inputs, dial_sents):
                        print(f"turn #: {self.turn_i}\n\n\n")
                        # pdb.set_trace()
                        # if self.turn_i > 9:
                        #     break
                        # if dial_turn_inputs[0]
                        if role_id == 0:
                            if self.past is None:
                                user_text = ""
                            response, [sents_success, sents_failed], have_enough_candidates = self.chat(input_text=user_text, mode=mode)
                            ground_truth = dial_sent
                            try:
                                assert not ground_truth.startswith("A:")
                            except:
                                pdb.set_trace()
                            cur_rewards = self.reward_func([ground_truth, sents_success, sents_failed], have_enough_candidates, with_ground_truth=True)

                            # print(f"truth: {ground_truth}")
                            # print(f"sent_success: \n{sents_success}")
                            # print(f"sent_failed: \n{sents_failed}")
                            # update
                            ground_truth_sents = sent_tokenize(ground_truth)                    
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
                                final_context_ids.append(f"{self.dialog_i}-{self.turn_i}")
                                # self.replay_buffer.add([deepcopy(self.contexts), "A:"+sent, reward])

                            # update contexts
                            logging.info(f"sys: {ground_truth}")
                            self.contexts.append("A:"+ground_truth)

                        else:
                            # breakpoint()
                            user_text = dial_sent
                            try:
                                assert not user_text.startswith("B:")
                            except:
                                pdb.set_trace()
                            self.contexts.append("B:"+user_text)
                            print(f"user: {user_text}")
                            logging.info(f"user: {user_text}")
                            # logits, past = model_B(dial_turn_inputs, past=past)
                            # all_logits.append(logits)

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
                            user_text = self.generate_user_utt_self_play()                
                        # system-side
                            if "bye" in user_text.lower() or "have a great day" in user_text.lower() \
                                or "have a great night" in user_text.lower() \
                                or "have a good day" in user_text.lower() \
                                or "have a good night" in user_text.lower() \
                                or "have a nice day" in user_text.lower() \
                                or "have a nice night" in user_text.lower() \
                                or self.turn_i >= 10:
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
                            final_context_ids.append(f"{self.dialog_i}-{self.turn_i}")
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
    def __init__(self, actor, model_A, model_B, device1, device2):
        self.sample_size = 1 # num of dialog to sample at a time
        self.maxlen = 1024
        self.replay_buffer = ReplayBuffer(self.maxlen)
        self.actor = actor
        self.tokenizer = tokenizer
        self.turn_ending = [628, 198]
        self.model_A = model_A
        self.model_B = model_B
        self.device1 = device1
        self.device2 = device2
        assert self.model_A.device is self.device1
        assert self.model_B.device is self.device2

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
            final_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-5)
        else:
            final_rewards = all_rewards
        
        # pdb.set_trace()
        assert len(all_contexts) == len(all_sents) == len(all_encoded_sents) == len(final_rewards) == len(all_context_ids)
        self.replay_buffer.add(zip(all_contexts, all_sents, all_encoded_sents, final_rewards, all_context_ids))
        
        # logging
        logger.info("Collecting Samples finished!")
        
        return all_rewards

    def calculate_old_logprobs(self, buffer_contexts, buffer_context_ids, buffer_sents, buffer_encoded_sents):
        assert self.model_A.device is self.device1
        assert self.model_B.device is self.device2

        start = time.time()
        buffer_old_logprobs = [None] * len(buffer_sents)
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

                # prepare the loss func inputs
                logits = logits[:, :-1].contiguous()
                target = batch_sents[:, 1:].contiguous()
                mask = mask[:, 1:].contiguous()

                sequences_logprobs = - criterion(logits, target, mask)
                old_logprobs = sequences_logprobs.sum(1)

                # store
                old_logprobs = old_logprobs.tolist()
                for i, j in enumerate(context_map[context_id]['sent_ids']):
                    buffer_old_logprobs[j] = old_logprobs[i]
        end = time.time()
        speed = (end - start) / len(buffer_sents)

        print(f"calculate_old_logprobs: {speed} per turn")
        logger.info(f"calculate_old_logprobs: {speed} per turn")
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
        for total_steps in tqdm(range(total_steps)):
            self.trained_steps += 1
            start = time.time()
            
            all_rewards = trainer.collect_generations(total_size=PpoParams.num_dialogs_to_sample)
            
            self.model_A.train()
            self.model_B.train()
            buffer_contexts, buffer_sents, buffer_encoded_sents, buffer_rewards, buffer_context_ids = zip(*trainer.replay_buffer)
            buffer_old_logprobs = self.calculate_old_logprobs(buffer_contexts, buffer_context_ids, buffer_sents, buffer_encoded_sents)

            for ppo_epoch in tqdm(range(PpoParams.ppo_epoch)):
                indices = np.arange(len(buffer_rewards))
                np.random.shuffle(indices)

                for i in range(PpoParams.batchsize // PpoParams.mini_batchsize):
                    sampled_indices = indices[i * PpoParams.mini_batchsize : (i + 1) * PpoParams.mini_batchsize]
                    if len(sampled_indices) == 0:
                        break
                    sampled_contexts = [buffer_contexts[j] for j in sampled_indices]
                    sampled_sents = [buffer_sents[j] for j in sampled_indices]
                    sampled_encoded_sents = [buffer_encoded_sents[j] for j in sampled_indices]
                    sampled_old_logprobs = [buffer_old_logprobs[j] for j in sampled_indices]
                    sampled_rewards = [buffer_rewards[j] for j in sampled_indices]

                    # make batches
                    # logits_list = []
                    # target_list = []
                    # mask_list = []
                    # torch.cuda.empty_cache()
                    sequence_logprob_list = []
                    try:
                        batch_encoded_sents = make_batch_sequences(sampled_encoded_sents, padding_value=PAD_TOKEN)
                    except:
                        pdb.set_trace()
                    for i, (contexts, old_logprob, reward) in enumerate(zip(sampled_contexts, sampled_old_logprobs, sampled_rewards)):
                        past = self.make_past(contexts)
                        encoded_sent = batch_encoded_sents[i, :].unsqueeze(0)
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
                        sequence_logprob_list.append(sequences_logprob)
                        # logits_list.append(logits)
                        # target_list.append(target)
                        # mask_list.append(mask)
                        # torch.cuda.empty_cache()
                    
                    # pdb.set_trace()
                    # logits = torch.cat(logits_list)
                    # target = torch.cat(target_list)
                    # mask = torch.cat(mask_list)
    
                    old_logprobs = torch.FloatTensor(sampled_old_logprobs).to(self.device1)
                    sampled_rewards = torch.FloatTensor(sampled_rewards).to(self.device1)

                    # calc advantages
                    advantages = sampled_rewards

                    # pdb.set_trace()
                    sequences_logprobs = torch.cat(sequence_logprob_list)#- criterion(logits, target, mask)

                    # here we need to calculate for the each token in the sequence
                    entropy = - (sequences_logprobs.exp() * sequences_logprobs).sum(1)
                    # print(f"entropy: {entropy}")
                    logger.info(f"entropy: {entropy}")
                    # entropy = entropy.clamp_min(min_entropy)
                    logprobs = sequences_logprobs.sum(1)

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

                    # print the final clipfrac and approxl
                    print(f"Approx KL {approx_kl.item()}, Clip Frac {clipfrac.item()}")
                    logger.info(f"Approx KL {approx_kl.item()}, Clip Frac {clipfrac.item()}")

                    # get the final loss
                    loss = policy_loss.mean() - entropy_coef * entropy.mean()

                    # update the model
                    optimizer.zero_grad()
                    loss.backward()
                    # must clip the gradient for stability
                    torch.nn.utils.clip_grad_norm_(self.model_A.parameters(), 0.5)
                    optimizer.step()
                    # scheduler.step()
                  
            print("Mean Reward", np.mean(all_rewards))
            logger.info(f"Mean Reward: {np.mean(all_rewards)}")
            end = time.time()
            speed = (end - start) / PpoParams.batchsize
            print("Speed", speed)
            logger.info(f"Speed: {speed}")

            print(f"max memory A: {torch.cuda.max_memory_allocated(model_A.device)}")
            logger.info(f"max memory A: {torch.cuda.max_memory_allocated(model_A.device)}")
            print(f"max memory B: {torch.cuda.max_memory_allocated(model_B.device)}")
            logger.info(f"max memory B: {torch.cuda.max_memory_allocated(model_B.device)}")
            torch.save((self.model_A.state_dict(), self.model_B.state_dict()), f"Checkpoint/{self.trained_steps}_steps_{np.mean(all_rewards)}_reward_model_A.pth")

NEW_MODEL_A_DIR = None#"Checkpoint/20_steps_0.049586776859504134_reward_model_A.pth"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")#torch.load(tokenizer_dir)
DEVICE1 = torch.device("cuda:1")
DEVICE2 = torch.device("cuda:0")
SPLIT_INTO = 2

model_A, model_B = load_model(cfg, device1=DEVICE1, device2=DEVICE2, split_into=SPLIT_INTO, model_A_dir=NEW_MODEL_A_DIR)
pdb.set_trace()

PAD_TOKEN = tokenizer.encoder["<|endoftext|>"]
clip_range = 0.2
entropy_coef = 1e-5
min_entropy = 10.0 # depends on the task
criterion = SequenceCrossEntropyLoss()


actor = Actor(model_A=model_A, model_B=model_B, tokenizer=tokenizer, device1=DEVICE1, device2=DEVICE2)

pdb.set_trace()
# optimizer
num_epochs = 10
num_gradients_accumulation = 1
num_train_optimization_steps = 1000

param_optimizer = list(model_A.named_parameters())# + list(model_B.named_parameters())
no_decay = ['ln', 'bias', 'LayerNorm']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

TOTAL_STEPS = 50
num_train_optimization_steps = PpoParams.ppo_epoch * (PpoParams.batchsize // PpoParams.mini_batchsize) * TOTAL_STEPS

from pytorch_pretrained_bert import OpenAIAdam
optimizer = OpenAIAdam(optimizer_grouped_parameters,
                       lr=2e-5,
                       warmup=0.1,
                       max_grad_norm=1.0,
                       weight_decay=0.01,
                       t_total=num_train_optimization_steps)
from transformers import get_linear_schedule_with_warmup
# optimizer = AdamW(optimizer_grouped_parameters,
#                 lr=3e-5,
#                 eps=1e-06)
# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                  warmup_steps=100,
#                                  num_training_steps=num_train_optimization_steps)
# optimizer = FusedAdam(optimizer_grouped_parameters, 
#                     lr=1e-6,
#                     eps=1e-06,
#                     bias_correction=False)









trainer = Trainer(actor=actor, model_A=model_A, model_B=model_B, device1=DEVICE1, device2=DEVICE2)

if NEW_MODEL_A_DIR is None:
    try:
        trainer.train_steps(total_steps=TOTAL_STEPS)
        print(torch.cuda.memory_summary(device=model_A.device, abbreviated=False))
        print(torch.cuda.memory_summary(device=model_B.device, abbreviated=False))

    except:
        print(torch.cuda.memory_summary(device=model_A.device, abbreviated=False))
        print(torch.cuda.memory_summary(device=model_B.device, abbreviated=False))
else:
    usr_text = ""
    while True:
        # interactive test
        sys_sent, [sents_success, sents_failed], have_enough_candidates = actor.chat(usr_text)
        print(f"sys: {sys_sent}\n")
        # print(f"success:\n {sents_success}")
        # print(f"failed:\n {sents_failed}")

        usr_text = input("usr: ")
# pdb.set_trace()





