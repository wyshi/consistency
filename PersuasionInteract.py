import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import re
import dialog_config
from AgentProfile.profiles_in_dev import GlobalProfile

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

import warnings
import signal
import sys
from sys import exit
warnings.filterwarnings('ignore')

from gpt_model import GPT2SimpleLM
from pytorch_pretrained_bert import GPT2Tokenizer
import config as cfg
from utils import is_repetition_with_context

from KnowledgeBase.KB import HumanRule
from KnowledgeBase import KB
from KnowledgeBase.KB import Domain
from nltk.tokenize import sent_tokenize
import logging

# def handler(signal_received, frame):
#     # Handle any cleanup here
#     print('SIGINT or CTRL-C detected. Exiting gracefully')
#     bot.save()
#     exit(0)

# In[3]:
tokenizer_dir = "/home/wyshi/persuasion/consistency/ARDM/persuasion/special3_gpt2_tokenizer.pkl"
model_dir = "/home/wyshi/persuasion/consistency/ARDM/persuasion/persuasion_medium_3.th"
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

tokenizer = torch.load(tokenizer_dir)

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


class PersuasiveBot:
    def __init__(self):
        self.tokenizer = torch.load(tokenizer_dir)
        # In[10]:

        # model_A = GPT2SimpleLM(GPT2SmallConfig)
        # model_B = GPT2SimpleLM(GPT2SmallConfig)
        # model_A_states, model_B_states = torch.load("CheckpointMedium/model_state_epoch_3.th")

        # define the model
        self.model_A = GPT2SimpleLM(GPT2MediumConfig)
        self.model_B = GPT2SimpleLM(GPT2MediumConfig)
        
        # load the model
        model_A_states, model_B_states = torch.load(model_dir)
        self.model_A.load_state_dict(model_A_states)
        self.model_B.load_state_dict(model_B_states)
        
        self.device = torch.device("cuda")
        self.model_A = self.model_A.to(self.device)
        self.model_B = self.model_B.to(self.device)

        self.model_A.eval()
        self.model_B.eval()

        self.eos = self.tokenizer.encode("\n\n\n")
        self.temperature = 0.7
        
        # Memory
        self.past = None
        self.domain = Domain(cfg.domain)
        self.global_profile = GlobalProfile(domain=self.domain)
        # self.usr_profile = UsrProfile()
        # self.sys_profile = SysProfile()
        self.human_rule = HumanRule(self)

        # self.logger = logger
        print("inited")
        self.logs = {'candidates':[],
                     'failed_candidates':[],
                     'global_profiles': [],
                     'responses': []}

        
    def chat(self, input_text, sid):
        sid = 0
        
        past_is_None = (self.past is None)
        if self.past is None:
            sys_sent = self.sys_respond_and_update(past_is_None=past_is_None)
            turn_responses = ["sys: "+sys_sent]
        else:
            # user-side
            if input_text == "quit":
                self.past = None
                return "ARDM MEMORY RESTARTS!"
            
            input_texts = sent_tokenize(input_text)
            input_texts_labels = [None]*len(input_texts)
            self.global_profile.update(sents=input_texts, sent_labels=input_texts_labels, who=self.domain.USR) #self.usr_profile.update(input_text, self.last_sys_labels)
            self.context = input_text
            self.turn_i += 1
            user = self.tokenizer.encode("B:" + input_text)
            prev_input = user + self.eos
            prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device)

            _, self.past = self.model_B(prev_input, past=self.past)

            # system-side
            sys_sent = self.sys_respond_and_update(past_is_None=past_is_None)
            turn_responses = ["usr: "+input_text,
                              "sys: "+sys_sent]
        
        self.logs['global_profiles'].append(self.global_profile.get_profiles())
        self.logs['responses'].append(turn_responses)
        return sys_sent

    def sample_one_sent(self, past):
        prev_input = self.tokenizer.encode("A:")
        prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device)

        """Sampling based method"""
        sent = []
        with torch.no_grad():
            for i in range(200):
                logits, past = self.model_A(prev_input, past=past)
                logits = logits[:, -1, :] / self.temperature
                logits = top_filtering(logits, top_k=500, top_p=0.9)
                # prev_input = logits.argmax(-1).unsqueeze(1)
                probs = F.softmax(logits, -1)
                prev_input = torch.multinomial(probs, num_samples=1)
                prev_word = prev_input.item()

                if prev_word == 628:
                    break
                elif prev_word == self.tokenizer.encoder["[EOS]"]:
                    past = None
                    return "ARDM MEMORY RESTARTS!", past
                    break
                else:
                    sent.append(prev_word)
        return self.tokenizer.decode(sent), past

    def reload(self):
        self.past = None

        # self.usr_profile.refresh()
        # self.sys_profile.refresh()
        self.global_profile.refresh()
        self.last_sys_labels = None

        # initialize params
        self.context = '<Start>'
        self.turn_i = 0
        self.cnt = 0
        
        self.logs = {'candidates':[],
                     'failed_candidates':[],
                     'global_profiles': [],
                     'responses': []}

        print("reloaded")

    def print_candidates(self, candidates, edited_candidates, sent_act_candidates, scores):
        log_this_turn = []
        print("=== candidates, len={} ===".format(len(candidates)))
        log_this_turn.append("=== candidates, len={} ===".format(len(candidates)))
        
        for c, edited_c, act, s in zip(candidates, edited_candidates, sent_act_candidates, scores):
            c = " ".join(c)
            edited_c = " ".join(edited_c)
            if c != edited_c:
                print("--------- different from edited candidates: act: {}, score: {}----------".format(act, s))
                print(c)
                print(edited_c)
                log_this_turn.append("--------- different from edited candidates: act: {}, score: {}----------".format(act, s))
                log_this_turn.append(c)
                log_this_turn.append(edited_c)
            else:
                print("----------------- act: {}, score : {}---------------------------".format(act, s))
                print(edited_c)
                log_this_turn.append("----------------- act: {}, score : {}---------------------------".format(act, s))
                log_this_turn.append(edited_c)
        print("==================")
        log_this_turn.append("==================")
        self.logs['candidates'].append(log_this_turn)

    def select_candidates(self, sent_candidates, sent_candidate_conflict_scores, sent_act_candidates, past_candidates):
        
        def select_index():
            if cfg.candidate_select_strategy == cfg.RANDOM_SELECT:
                return random.sample(range(len(sent_candidate_conflict_scores)), k=1)[0], [1/len(sent_candidate_conflict_scores)]*len(sent_candidate_conflict_scores)
            else:
                one_minus_score = 1 - np.array(sent_candidate_conflict_scores)
                normalized_score = one_minus_score/(one_minus_score.sum())

                if cfg.candidate_select_strategy == cfg.REPETITION_RATIO:
                    # if cfg.debug:
                    #     print("~~~~~~~~in select_candidates~~~~~~~~~")
                    #     print("normalized_score: {}".format(normlized_score))
                    #     print("original_score: {}".format(sent_candidate_conflict_scores))
                    #     print("~"*20)
                    return np.random.choice(range(len(sent_candidate_conflict_scores)), size=1, 
                                        replace=False, p=normalized_score)[0], normalized_score
                elif cfg.candidate_select_strategy == cfg.FIRST_OF_CANDIDATES:
                    return 0, normalized_score

        rule_result = self.human_rule.enforce(sent_candidates, sent_act_candidates, past_candidates)
        if cfg.debug:
            print("rule_result:\n")
            print(rule_result)
            print("rule_result:")
        if rule_result is None:
            selected_i, scores = select_index()
            sents, sent_acts, past = sent_candidates[selected_i], sent_act_candidates[selected_i], past_candidates[selected_i]
        elif type(rule_result) is int:
            selected_i = rule_result
            scores = 0
            sents, sent_acts, past = sent_candidates[selected_i], sent_act_candidates[selected_i], past_candidates[selected_i]
        else:
            enforced_sents, enforced_acts = rule_result
            selected_i, scores = select_index()
            sents, sent_acts, past = sent_candidates[selected_i], sent_act_candidates[selected_i], past_candidates[selected_i]
            
            sents = sents + enforced_sents
            sent_acts = sent_acts + enforced_acts

        return sents, sent_acts, past, scores

    def sys_respond_and_update(self, past_is_None):
        # start A's utterance
        sent_candidates, edited_sent_candidates, sent_candidate_conflict_scores, sent_act_candidates, past_candidates = [], [], [], [], []
        have_enough_candidates = False
        num_rounds = 0
        failed_candidates = []
        while not have_enough_candidates and num_rounds < int(cfg.MAX_NUM_CANDIDATES/cfg.NUM_CANDIDATES):
            num_rounds += 1
            for _ in range(cfg.NUM_CANDIDATES):
                sent, past = self.sample_one_sent(past=self.past)
                sents = sent_tokenize(sent)
                
                # use regex to re-label
                sent_acts = self.global_profile.regex_label(sents, self.context, self.turn_i)

                # check conflict condition
                # conflict_status_with_sys, conflict_amount_with_sys, edited_sents, edited_sent_acts = self.sys_profile.check_conflict(sents, sent_acts)
                # if past_is_None:
                #     conflict_condition = (conflict_status_with_sys in [cfg.PASS])                    
                # else:
                #     conflict_status_with_usr, conflict_amount_with_usr, edited_sents, edited_sent_acts = self.usr_profile.check_conflict(edited_sents, edited_sent_acts)                    
                #     conflict_condition = (conflict_status_with_sys in [cfg.PASS]) and (conflict_status_with_usr in [cfg.PASS])

                conflict_condition, conflict_amount, edited_sents, edited_sent_acts, fail_reason = self.global_profile.check_conflict(sents, sent_acts)  

                if conflict_condition:   
                    sent_candidates.append(sents)
                    edited_sent_candidates.append(edited_sents)
                    sent_candidate_conflict_scores.append(conflict_amount)
                    sent_act_candidates.append(edited_sent_acts)
                    past_candidates.append(past)
                else:
                    failed_candidates.append([sents, sent_acts, fail_reason])

            have_enough_candidates = (len(past_candidates) > 0)
        if not have_enough_candidates:
            # as long as it's not a contradiction, randomly pick one 
                if cfg.debug:
                    print("no enough candidates! randomly generate the next one!")
                sent, past = self.sample_one_sent(past=self.past)
                sents = sent_tokenize(sent)

                sent_acts = self.global_profile.regex_label(sents, self.context, self.turn_i)
                sent_candidates.append(sents)
                edited_sent_candidates.append(sents)
                sent_candidate_conflict_scores.append(0)
                sent_act_candidates.append(sent_acts)
                past_candidates.append(past)
        
        self.logs['failed_candidates'].append(failed_candidates)

        # check consistency and pick one candidate
        self.cnt += 1
        sents, sent_acts, past, scores = self.select_candidates(edited_sent_candidates, sent_candidate_conflict_scores, sent_act_candidates, past_candidates)
        self.print_candidates(sent_candidates, edited_sent_candidates, sent_act_candidates, scores)
        # check conflict within the sents
        sents, sent_acts = self.check_conflict_within_selected_sents(sents, sent_acts)
        
        self.global_profile.update(sents=sents, sent_labels=sent_acts, who=self.domain.SYS) #self.last_sys_labels = self.sys_profile.update(sys_texts=sents, sys_labels=sent_acts)

        # join sentences! finally!
        sent = " ".join(sents)
        if sent == "ARDM MEMORY RESTARTS!":
            self.past = None
            return "ARDM MEMORY RESTARTS!"
        self.past = past

        # print("A:" + tokenizer.decode(sent))
        # finish tail
        prev_input = torch.LongTensor(self.eos).unsqueeze(0).to(self.device)
        _, self.past = self.model_A(prev_input, past=self.past)
        
        return sent

    def check_conflict_within_selected_sents(self, sents, sent_acts):
        statuses = [True]*len(sents)
        for i in range(len(sents)):
            for j in range(i+1, len(sents)):
                if statuses[i] is True and statuses[j] is True:
                    is_repetition, repetition_ratio = is_repetition_with_context(sents[i], 
                                                                            [sents[j]], 
                                                                            threshold=cfg.repetition_threshold) 
                    if is_repetition:
                        if cfg.debug:
                            print("@@@ in check_conflict_within_selected_sents, repetition within the sentence candidate: {}\n{}".format(sents[i], sents[j]))
                        statuses[j] = False
                else:
                    pass
        edited_sents = [sents[i] for i, status in enumerate(statuses) if status]
        edited_sent_acts = [sent_acts[i] for i, status in enumerate(statuses) if status]
        
        return edited_sents, edited_sent_acts

    def save(self):
        print("\n")
        logging.debug("\n")
        logging.debug("*************************** new dialog ****************************************")
        for turn_i in range(len(self.logs['responses'])):
            for k in ['responses', 'candidates', 'failed_candidates', 'global_profiles']:
                print("{}\n".format(k))
                print("-"*100)
                logging.debug("{}\n".format(k))
                if k in ['responses', 'candidates', 'failed_candidates']:
                    for a in self.logs[k][turn_i]:
                        print(a)
                        logging.debug(a)
                else:
                    for world in self.logs[k][turn_i]:                        
                        for profile in self.logs[k][turn_i][world]:
                            print("*******{}: {}*******".format(world, profile))
                            logging.debug("*******{}: {}*******".format(world, profile))
                            for key, value in self.logs[k][turn_i][world][profile].items():
                                print("{}: {}".format(key, value))
                                logging.debug("{}: {}".format(key, value))
                            print("")
                            logging.debug("")
            print("\n")
            logging.debug("\n")
        
        logging.debug("*************************** dialog end ****************************************")
        



if __name__ == "__main__":
    logging.basicConfig(filename=cfg.log_file,level=logging.DEBUG)
    

    bot = PersuasiveBot()
    bot.reload()
    user_text = ""
    signal.signal(signal.SIGINT, signal.default_int_handler)

    while True:
        try:
            if bot.past is not None:
                user_text  = input("user: ")
            else:
                print("INIT MEMORY!")
                bot.save()
                bot.reload()
            
            response = bot.chat(user_text, 0)
            bot.global_profile.print()
            
            if response == "ARDM MEMORY RESTARTS!":
                print("ARDM MEMORY RESTARTS!")
            else:
                print("system: ", response)
            print("$$$$$$$$$$$$$$$$$$$$$\n\n\n\n\n\n")

        except KeyboardInterrupt:
            bot.save()
            sys.exit()