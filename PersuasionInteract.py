import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

import re
import dialog_config
from AgentProfile.profiles_in_dev import GlobalProfile
from model_clf import build_model_classifier

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
import pdb
import os
import csv
import pickle as pkl

# from gpt_model import GPT2SimpleLM
# from pytorch_pretrained_bert import GPT2Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from GPTModel import GPT2LMHeadModel_modified
import config as cfg
from utils import is_repetition_with_context

from KnowledgeBase.KB import HumanRule
from KnowledgeBase import KB
from KnowledgeBase.KB import Domain
from nltk.tokenize import sent_tokenize
import logging

from copy import deepcopy

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

class ModelClassifierConfig:
    model_dir = cfg.model_clf_dir
    device1 = torch.device(cfg.model_clf_device1)
    device2 = torch.device(cfg.model_clf_device2)

class PersuasiveBot:
    def __init__(self, model_A=None, model_B=None, tokenizer=None, device1=None, device2=None):
        if tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")#torch.load(tokenizer_dir)
        else:
            self.tokenizer = tokenizer
        self.max_sequence_len = cfg.max_sequence_len
        if device1 is None:
            self.device1 = torch.device("cuda:0")
            self.device2 = torch.device("cuda:0")
        else:
            self.device1 = device1
            self.device2 = device2
        self.load_models(model_A, model_B)

        # tokenizer weird behavior 
        # sep = tokenizer.encode("\n\n\n")
        self.eos = [628, 198]
        self.temperature = 0.7
        
        # Memory
        # self.need_reload = True
        self.past = None
        self.b_hidden_states = None
        self.human_demonstrations = []

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
        self.reload()

    def load_models(self, model_A, model_B):
        # define the model

        self.model_clf = build_model_classifier(ModelClassifierConfig.model_dir, 
                                                ModelClassifierConfig.device1, 
                                                ModelClassifierConfig.device2)

        if model_A is None and model_B is None:
            if cfg.model_size == "small":
                self.model_A = GPT2LMHeadModel_modified.from_pretrained("gpt2", output_hidden_states=True)
                self.model_B = GPT2LMHeadModel_modified.from_pretrained("gpt2", output_hidden_states=True)
            elif cfg.model_size == "medium":
                self.model_A = GPT2LMHeadModel_modified.from_pretrained("gpt2-medium", output_hidden_states=True)
                self.model_B = GPT2LMHeadModel_modified.from_pretrained("gpt2-medium", output_hidden_states=True)

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
                else:
                    model_A_states, model_B_states = torch.load(cfg.new_medium_model_dir)

            if cfg.use_old_model:
                strict = False
            else:
                strict = True
            self.model_A.load_state_dict(model_A_states, strict=strict)
            self.model_B.load_state_dict(model_B_states, strict=strict)

            self.model_A = self.model_A.to(self.device1)
            self.model_B = self.model_B.to(self.device2)
            self.model_A.device = self.device1
            self.model_B.device = self.device2

            self.model_A.eval()
            self.model_B.eval()

        else:
            assert model_A is not None and model_B is not None
            print("loaded predefined models!!!!!!\n\n\n")
            self.model_A = model_A
            self.model_B = model_B

    def chat(self, input_text=None, mode=cfg.interactive_mode):
        # sid = 0        
        # pdb.set_trace()
        if self.past is None:
            sys_sent, [sents_success, sents_failed], have_enough_candidates = self.sys_respond_and_update(mode=mode)
            turn_responses = ["sys: "+sys_sent]
        else:
            # user-side
            if mode != cfg.self_play_mode:
                assert input_text is not None
                if input_text == "quit":
                    # del self.past
                    self.past = None
                    # self.reload()
                    return
                
                input_texts = sent_tokenize(input_text)
                # input_texts_labels = [None]*len(input_texts)
                input_texts_labels, _ = self.global_profile.regex_label(self.model_clf,
                                                                        input_texts, 
                                                                        which_task="B")
                self.model_clf.set_past(sent=input_text, 
                                        which_task="B")
                
                self.global_profile.update(sents=input_texts, sent_labels=input_texts_labels, who=self.domain.USR) #self.usr_profile.update(input_text, self.last_sys_labels)
                self.last_sent = input_text
                self.turn_i += 1
                user = self.tokenizer.encode("B:" + input_text)
                prev_input = user + self.eos
                prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device2)

                if self.past is not None and self.model_B.device != self.past[0].device:
                    past = [p.to(self.model_B.device) for p in self.past]
                    self.past = past

                _, self.past, self.b_hidden_states = self.model_B(prev_input, past=self.past)



            elif mode == cfg.self_play_mode:
                input_text = self.generate_user_utt_self_play()

                if "bye" in input_text.lower() or "have a great day" in input_text.lower() \
                    or "have a great night" in input_text.lower() \
                    or self.turn_i >= 10:
                    print(f"user: {input_text}\n$$$$$$$$")
                    self.past = None
                    return
                    # return "ARDM MEMORY RESTARTS!"

                print(f"user: {input_text}\n$$$$$$$$")
            # system-side
            sys_sent, [sents_success, sents_failed], have_enough_candidates = self.sys_respond_and_update(mode=mode)
            turn_responses = ["usr: "+input_text,
                              "sys: "+sys_sent]
        
        self.logs['global_profiles'].append(self.global_profile.get_profiles())
        self.logs['responses'].append(turn_responses)
        return sys_sent, [sents_success, sents_failed], have_enough_candidates

    def generate_user_utt_self_play(self):
        input_text, self.past, self.b_hidden_states = self.sample_one_sent(past=self.past, model=self.model_B, prefix="B:")
        # 
        # input("pause")

        input_texts = sent_tokenize(input_text)
        input_texts_labels, _ = self.global_profile.regex_label(self.model_clf,
                                                                input_texts, 
                                                                which_task="B")
        self.model_clf.set_past(sent=input_text, 
                                which_task="B")
        self.global_profile.update(sents=input_texts, sent_labels=input_texts_labels, who=self.domain.USR) #self.usr_profile.update(input_text, self.last_sys_labels)
        self.last_sent = input_text
        self.turn_i += 1

        # finish tail
        if self.past is not None and self.model_B.device != self.past[0].device:
            past = [p.to(self.model_B.device) for p in self.past]
            self.past = past

        prev_input = torch.LongTensor(self.eos).unsqueeze(0).to(self.device2)
        _, self.past, self.b_hidden_states = self.model_B(prev_input, past=self.past)

        return input_text

    def sample_one_sent(self, past, model, prefix="A:"):
        prev_input = self.tokenizer.encode(prefix)
        if prefix == "A:":
           prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device1)
        else:
           prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device2)

        if past is not None and model.device != past[0].device:
            past = [p.to(model.device) for p in past]
        """Sampling based method"""
        sent = []
        # pdb.set_trace()
        with torch.no_grad():
            import pdb
            # pdb.set_trace()
            for i in range(self.max_sequence_len):
                logits, past, hidden_states = model(prev_input, past=past)
                logits = logits[:, -1, :] / self.temperature
                logits = top_filtering(logits, top_k=500, top_p=0.9)
                # prev_input = logits.argmax(-1).unsqueeze(1)
                probs = F.softmax(logits, -1)
            
                prev_input = torch.multinomial(probs, num_samples=1)
                prev_word = prev_input.item()
                

                if prev_word == 628:
                    break
                # elif prev_word == self.tokenizer.encoder["[EOS]"]:
                #     past = None
                #     return "ARDM MEMORY RESTARTS!", past
                #     break
                else:
                    sent.append(prev_word)
        return self.tokenizer.decode(sent), past, hidden_states

    def sample_one_sent_test(self, past, model, sent, prefix="A:"):
        prev_input = self.tokenizer.encode(prefix+sent)
        prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device1)
        
        """Sampling based method"""
        sent = []
        # pdb.set_trace()
        with torch.no_grad():
            logits, past, hidden_states = model(prev_input, past=past)
            # for i in range(200):
            #     logits, past, hidden_states = model(prev_input, past=past)
            #     logits = logits[:, -1, :] / self.temperature
            #     logits = top_filtering(logits, top_k=500, top_p=0.9)
            #     # prev_input = logits.argmax(-1).unsqueeze(1)
            #     probs = F.softmax(logits, -1)
            
            #     prev_input = torch.multinomial(probs, num_samples=1)
            #     prev_word = prev_input.item()
                

            #     if prev_word == 628:
            #         break
            #     # elif prev_word == self.tokenizer.encoder["[EOS]"]:
            #     #     past = None
            #     #     return "ARDM MEMORY RESTARTS!", past
            #     #     break
            #     else:
            #         sent.append(prev_word)
        return past, hidden_states

    def sample_one_sent_test_2(self, past, model, sent, prefix="A:"):
        prev_input = self.tokenizer.encode(prefix)
        prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device1)
        
        encoded_sent = self.tokenizer.encode(sent)
        """Sampling based method"""
        sent = []
        # pdb.set_trace()
        with torch.no_grad():
            logits, past, hidden_states = model(prev_input, past=past)
            for tok in encoded_sent:
                prev_input = torch.LongTensor(tok).unsqueeze(0).to(self.device1)
                logits, past, hidden_states = model(prev_input, past=past)
                # torch.cuda.empty_cache()
            # for i in range(200):
            #     logits, past, hidden_states = model(prev_input, past=past)
            #     logits = logits[:, -1, :] / self.temperature
            #     logits = top_filtering(logits, top_k=500, top_p=0.9)
            #     # prev_input = logits.argmax(-1).unsqueeze(1)
            #     probs = F.softmax(logits, -1)
            
            #     prev_input = torch.multinomial(probs, num_samples=1)
            #     prev_word = prev_input.item()
                

            #     if prev_word == 628:
            #         break
            #     # elif prev_word == self.tokenizer.encoder["[EOS]"]:
            #     #     past = None
            #     #     return "ARDM MEMORY RESTARTS!", past
            #     #     break
            #     else:
            #         sent.append(prev_word)
        return past, hidden_states

    def reload(self):
        # torch.cuda.empty_cache()
        self.model_clf.reload()
        self.past = None
        self.b_hidden_states = None
        self.human_demonstrations = []

        # self.usr_profile.refresh()
        # self.sys_profile.refresh()
        self.global_profile.refresh()
        self.last_sys_labels = None

        # initialize params
        self.last_sent = '<Start>'
        self.turn_i = 0
        self.cnt = 0
        
        self.logs = {'candidates':[],
                     'failed_candidates':[],
                     'global_profiles': [],
                     'responses': []}

        print("reloaded\n\n")

    def print_candidates(self, candidates, edited_candidates, sent_act_candidates, scores=None, failed_candidates=None):
        log_this_turn = []
        sents_success, sents_failed = [], []
        if cfg.print_candidates:
            print("=== candidates, len={} ===".format(len(candidates)))
        log_this_turn.append("=== candidates, len={} ===".format(len(candidates)))
        
        if type(scores[0]) is not bool:
            for c, edited_c, act, s in zip(candidates, edited_candidates, sent_act_candidates, scores):
                c = " ".join(c)
                edited_c = " ".join(edited_c)
                if c != edited_c:
                    if cfg.print_candidates:
                        print("--------- different from edited candidates: act: {}, score: {}----------".format(act, s))
                        print(c)
                        print(edited_c)
                    sents_success.append(edited_c)
                    log_this_turn.append("--------- different from edited candidates: act: {}, score: {}----------".format(act, s))
                    log_this_turn.append(c)
                    log_this_turn.append(edited_c)
                else:
                    if cfg.print_candidates:
                        print("----------------- act: {}, score : {}---------------------------".format(act, s))
                        print(edited_c)
                    log_this_turn.append("----------------- act: {}, score : {}---------------------------".format(act, s))
                    log_this_turn.append(edited_c)
                    sents_success.append(edited_c)
            if cfg.print_candidates:
                print("==================")
            log_this_turn.append("==================")
            self.logs['candidates'].append(log_this_turn)

            if failed_candidates:
                i = 0
                for sent, act, reason, past, hidden_states in failed_candidates:
                    sent = " ".join(sent)
                    sents_failed.append(sent)
                    if cfg.print_candidates:
                        print("----------------- failed candidates: reason: {}  ---------------------------".format(reason))
                        print("{}".format(sent))
                    i += 1
        else:
            scores, failed_scores = scores[:len(edited_candidates)], scores[len(edited_candidates):]
            for c, edited_c, act, s in zip(candidates, edited_candidates, sent_act_candidates, scores):
                c = " ".join(c)
                edited_c = " ".join(edited_c)
                if c != edited_c:
                    to_print = "--------- different from edited candidates: act: {}, score: {}----------".format(act, s)
                    if s is True:
                        to_print = "SELECTED " + to_print 
                        sents_success.append(edited_c)
                    else:
                        sents_failed.append(edited_c)
                    if cfg.print_candidates:
                        print(to_print)
                        print(c)
                        print(edited_c)
                    log_this_turn.append(to_print)
                    log_this_turn.append(c)
                    log_this_turn.append(edited_c)
                    
                else:
                    to_print = "----------------- act: {}, score : {} ---------------------".format(act, s)
                    if s is True:
                        to_print = "SELECTED " + to_print 
                        sents_success.append(edited_c)
                    else:
                        sents_failed.append(edited_c)
                    if cfg.print_candidates:
                        print(to_print)
                        print(edited_c)
                    log_this_turn.append(to_print)
                    log_this_turn.append(edited_c)
                    
            if cfg.print_candidates:    
                print("==================")
            log_this_turn.append("==================")
            self.logs['candidates'].append(log_this_turn)

            if failed_candidates:
                try:
                    i = 0
                    for sent, act, reason, past, hidden_states in failed_candidates:
                        sent = " ".join(sent)
                        to_print = "----------------- failed candidates: reason: {}  -------------------".format(reason)
                        if failed_scores[i] is True:
                            to_print = "SELECTED " + to_print
                        if cfg.print_candidates:
                            print(to_print)
                            print("{}".format(sent))
                        i += 1
                        sents_failed.append(sent)
                except:
                    import pdb
                    pdb.set_trace()

        return sents_success, sents_failed

    def print_all_generated_candidates(self, candidates, edited_candidates, sent_act_candidates,
                                             failed_candidates):
        log_this_turn = []
        print("=== candidates, len={} ===".format(len(candidates)))
        log_this_turn.append("=== candidates, len={} ===".format(len(candidates)))
        
        i = 0
        for c, edited_c, act in zip(candidates, edited_candidates, sent_act_candidates):
            c = " ".join(c)
            edited_c = " ".join(edited_c)
            if c != edited_c:
                print("--------- different from edited candidates: act: {} ----------".format(act))
                print("{}). {}".format(i, c))
                print("edit to -->")
                print(edited_c)
                log_this_turn.append("--------- different from edited candidates: act: {} ----------".format(act))
                log_this_turn.append(c)
                log_this_turn.append(edited_c)
            else:
                print("----------------- act: {} ---------------------------".format(act))
                print("{}). {}".format(i, edited_c))
                log_this_turn.append("----------------- act: {} ---------------------------".format(act))
                log_this_turn.append(edited_c)
            i += 1
        print("==================")

        for sent, act, reason, past, hidden_states in failed_candidates:
            print("----------------- failed candidates:  ---------------------------")
            print("{}). {}".format(i, sent))
            i += 1

        log_this_turn.append("==================")
        self.logs['candidates'].append(log_this_turn)

    def select_candidates(self, sent_candidates, sent_candidate_conflict_scores, sent_act_candidates, past_candidates):
        
        def select_index():
            if cfg.candidate_select_strategy == cfg.RANDOM_SELECT:
                return random.sample(range(len(sent_candidate_conflict_scores)), k=1)[0], [1/len(sent_candidate_conflict_scores)]*len(sent_candidate_conflict_scores)
            else:
                one_minus_score = 1 - np.array(sent_candidate_conflict_scores)
                normalized_score = one_minus_score/(one_minus_score.sum())

                if cfg.candidate_select_strategy in [cfg.REPETITION_RATIO, cfg.IMITATION_LEARNING_SELECTION]:
                    # if cfg.debug:
                    #     print("~~~~~~~~in select_candidates~~~~~~~~~")
                    #     print("normalized_score: {}".format(normlized_score))
                    #     print("original_score: {}".format(sent_candidate_conflict_scores))
                    #     print("~"*20)
                    return np.random.choice(range(len(sent_candidate_conflict_scores)), size=1, 
                                        replace=False, p=normalized_score)[0], normalized_score
                elif cfg.candidate_select_strategy == cfg.FIRST_OF_CANDIDATES:
                    return 0, normalized_score
                else:
                    raise ValueError(f"{cfg.candidate_select_strategy} is not supported in select_index()")

        selected_by_func = select_index()
        sents, sent_acts, past, scores = self.apply_human_rule(sent_candidates, sent_act_candidates, past_candidates,
                                                                selected_by_func)    

        return sents, sent_acts, past, scores

    def apply_human_rule(self, sent_candidates, sent_act_candidates, past_candidates,
                         selected_by_func):
        rule_results = [self.human_rule.enforce(sents, sent_acts, past)
                        for sents, sent_acts, past in zip(sent_candidates, sent_act_candidates, past_candidates)]
        
        if cfg.verbose:
            pass
            print("rule_result:\n")
            print(rule_results)
            print("rule_result:")

        # pdb.set_trace()
        if True in rule_results:
            for i, rule_result in enumerate(rule_results):
                if rule_result is True:
                    break
            selected_i, scores = i, 0
            sents, sent_acts, past = sent_candidates[selected_i], sent_act_candidates[selected_i], past_candidates[selected_i]
        else:
            if all(v is None for v in rule_results):
                selected_i, scores = selected_by_func
                sents, sent_acts, past = sent_candidates[selected_i], sent_act_candidates[selected_i], past_candidates[selected_i]
            else:
                # pdb.set_trace()
                selected_i, scores = selected_by_func
                enforced_sents, enforced_acts = rule_results[selected_i]
                sents, sent_acts, past = sent_candidates[selected_i], sent_act_candidates[selected_i], past_candidates[selected_i]

                # encode the enforced rule sentences
                prev_input = self.tokenizer.encode(" "+" ".join(enforced_sents))
                prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device1)

                if past is not None and self.model_A.device != past[0].device:
                    past = [p.to(self.model_A.device) for p in past]

                _, past, hidden_state = self.model_A(prev_input, past=past)

                # concatenate the enforced sentences from rule
                sents = sents + enforced_sents
                sent_acts = sent_acts + enforced_acts
                # pdb.set_trace()
        return sents, sent_acts, past, scores
        
        # if rule_result is None:
        #     selected_i, scores = selected_by_func
        #     sents, sent_acts, past = sent_candidates[selected_i], sent_act_candidates[selected_i], past_candidates[selected_i]
        # elif type(rule_result) is int:
        #     selected_i, scores = rule_result, 0
        #     sents, sent_acts, past = sent_candidates[selected_i], sent_act_candidates[selected_i], past_candidates[selected_i]
        # else:
        #     enforced_sents, enforced_acts = rule_result
        #     selected_i, scores = selected_by_func
        #     sents, sent_acts, past = sent_candidates[selected_i], sent_act_candidates[selected_i], past_candidates[selected_i]

        #     # encode the enforced rule sentences
        #     prev_input = self.tokenizer.encode(" "+" ".join(enforced_sents))
        #     prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device)
        #     _, past, hidden_state = self.model_A(prev_input, past=past)

        #     # concatenate the enforced sentences from rule
        #     sents = sents + enforced_sents
        #     sent_acts = sent_acts + enforced_acts
        
        # return sents, sent_acts, past, scores

    def edit_with_human_rule(self, sent_candidates, edited_sent_candidates, sent_act_candidates, past_candidates,\
                             failed_candidates):
        sent_candidates_after_rule, edited_sent_candidates_after_rule, sent_act_candidates_after_rule, \
        past_candidates_after_rule, failed_candidates_after_rule = [], [], [], [], []

        rule_results = [self.human_rule.enforce(sents, sent_acts, past)
                        for sents, sent_acts, past in zip(sent_candidates, sent_act_candidates, past_candidates)]
        failed_rule_results = [self.human_rule.enforce(sents, sent_acts, past) \
                              for sents, sent_acts, fail_reason, past, hidden_states in failed_candidates]
        
        if cfg.verbose:
            pass
            print("rule_result:\n")
            print(rule_results)
            print("rule_result:")

        for sents, edited_sents, sent_acts, past, rule_result in zip(sent_candidates, edited_sent_candidates, sent_act_candidates, past_candidates, rule_results):
            if not (rule_result is None or rule_result is True):
                enforced_sents, enforced_acts = rule_result
                # encode the enforced rule sentences
                prev_input = self.tokenizer.encode(" "+" ".join(enforced_sents))
                prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device1)
                
                if past is not None and self.model_A.device != past[0].device:
                    past = [p.to(self.model_A.device) for p in past]

                _, past, hidden_state = self.model_A(prev_input, past=past)

                # concatenate the enforced sentences from rule
                sents = sents + enforced_sents
                edited_sents = edited_sents + enforced_sents
                sent_acts = sent_acts + enforced_acts

            sent_candidates_after_rule.append(sents)
            edited_sent_candidates_after_rule.append(edited_sents)
            sent_act_candidates_after_rule.append(sent_acts)
            past_candidates_after_rule.append(past)

        for (sents, sent_acts, fail_reason, past, hidden_states), rule_result in zip(failed_candidates, failed_rule_results):
            if not (rule_result is None or rule_result is True):
                enforced_sents, enforced_acts = rule_result
                # encode the enforced rule sentences
                prev_input = self.tokenizer.encode(" "+" ".join(enforced_sents))
                prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device1)

                if past is not None and self.model_A.device != past[0].device:
                    past = [p.to(self.model_A.device) for p in past]

                _, past, hidden_state = self.model_A(prev_input, past=past)

                # concatenate the enforced sentences from rule
                sents = sents + enforced_sents
                sent_acts = sent_acts + enforced_acts

            failed_candidates_after_rule.append([sents, sent_acts, fail_reason, past, hidden_states])
        
        return sent_candidates_after_rule, edited_sent_candidates_after_rule, sent_act_candidates_after_rule, \
               past_candidates_after_rule, failed_candidates_after_rule


    def human_select_candidates(self):
        usr_input = input("select: ")
        usr_selections = usr_input.split(",")
        try:
            usr_selections = [int(usr_selection) for usr_selection in usr_selections]
            return sorted(usr_selections)
        except:
            return usr_input        

    def sys_respond_and_update(self, mode):
        # start A's utterance
        past_is_None = (self.past is None)
        sent_candidates, edited_sent_candidates, sent_candidate_conflict_scores, sent_act_candidates, past_candidates, hidden_states_candidates = [], [], [], [], [], []
        have_enough_candidates = False
        num_rounds = 0
        failed_candidates = []
        while not have_enough_candidates and num_rounds < int(cfg.MAX_NUM_CANDIDATES/cfg.NUM_CANDIDATES):
            num_rounds += 1
            for _ in range(cfg.NUM_CANDIDATES):
                sent, past, hidden_states = self.sample_one_sent(past=self.past, model=self.model_A)                
                # past_test, hidden_states_test = self.sample_one_sent_test(past=self.past, model=self.model_A, sent=sent)
                # # past_test_2, hidden_states_test_2 = self.sample_one_sent_test_2(past=self.past, model=self.model_A, sent=sent)
                # if self.turn_i in [0, 5] and _ in [0, 9]:
                #     pdb.set_trace()
                sents = sent_tokenize(sent)
                
                # use regex to re-label
                sent_acts, _ = self.global_profile.regex_label(self.model_clf,
                                                               sents, 
                                                               which_task="A")
                                                               

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
                    hidden_states_candidates.append(hidden_states)
                else:
                    failed_candidates.append([sents, sent_acts, fail_reason, past, hidden_states])

            have_enough_candidates = (len(past_candidates) > 0)
        if (not have_enough_candidates):
            # as long as it's not a contradiction, randomly pick one 
            if cfg.debug:
                print("no enough candidates! randomly generate the next one!")
            sent, past, hidden_states = self.sample_one_sent(past=self.past, model=self.model_A)
            sents = sent_tokenize(sent)

            sent_acts, _ = self.global_profile.regex_label(self.model_clf,
                                                           sents, 
                                                           which_task="A")
            sent_candidates.append(sents)
            edited_sent_candidates.append(sents)
            sent_candidate_conflict_scores.append(0)
            sent_act_candidates.append(sent_acts)
            past_candidates.append(past)
            hidden_states_candidates.append(hidden_states)


        self.logs['failed_candidates'].append(failed_candidates)

        # check consistency and pick one candidate
        self.cnt += 1
        if cfg.candidate_select_strategy == cfg.HUMAN_SELECTION:
            sent_candidates, edited_sent_candidates, sent_act_candidates, past_candidates,\
            failed_candidates = self.edit_with_human_rule(sent_candidates, edited_sent_candidates, sent_act_candidates, past_candidates,\
                             failed_candidates)            
            self.print_all_generated_candidates(sent_candidates, edited_sent_candidates, sent_act_candidates,
                                                failed_candidates)
            human_selected_ids = self.human_select_candidates()
            if type(human_selected_ids) is list:
                selected_i = np.random.choice(human_selected_ids)
                if False:
                    if selected_i < len(sent_candidates):
                        sents, sent_acts, past = sent_candidates[selected_i], sent_act_candidates[selected_i], past_candidates[selected_i]
                    else:
                        failed_candidate = failed_candidates[selected_i-len(sent_candidates)]
                        sents, sent_acts, past = failed_candidate[0], failed_candidate[1], failed_candidate[3]
                else:
                    if selected_i < len(edited_sent_candidates):
                        sents, sent_acts, past = edited_sent_candidates[selected_i], sent_act_candidates[selected_i], past_candidates[selected_i]
                    else:
                        failed_candidate = failed_candidates[selected_i-len(edited_sent_candidates)]
                        sents, sent_acts, past = failed_candidate[0], failed_candidate[1], failed_candidate[3]
                sents_success, sents_failed = self.log_human_demonstration(sent_candidates, edited_sent_candidates, sent_act_candidates, past_candidates, 
                                            hidden_states_candidates,
                                            failed_candidates, human_selected_ids)
            else:
                prev_input = self.tokenizer.encode("A:" + human_selected_ids)
                prev_input = torch.LongTensor(prev_input).unsqueeze(0).to(self.device1)

                if self.past is not None and self.model_A.device != self.past[0].device:
                    past = [p.to(self.model_A.device) for p in self.past]
                    self.past = past

                _, past, hidden_state = self.model_A(prev_input, past=self.past)

                sent = human_selected_ids
                sents = sent_tokenize(sent)
                sent_acts, _ = self.global_profile.regex_label(self.model_clf,
                                                               sents, 
                                                               which_task="A")
                
                sents_success, sents_failed = self.log_human_demonstration(sent_candidates, edited_sent_candidates, sent_act_candidates, past_candidates, hidden_states_candidates,
                                            failed_candidates, human_selected_ids=[], human_added_candidates=[[sents, sent_acts, past, hidden_state]])
                
            
        elif cfg.candidate_select_strategy == cfg.IMITATION_LEARNING_SELECTION:
            sent_candidates, edited_sent_candidates, sent_act_candidates, past_candidates,\
            failed_candidates = self.edit_with_human_rule(sent_candidates, edited_sent_candidates, sent_act_candidates, past_candidates,\
                             failed_candidates)            
            model_selected_id_for_success_candidates, model_selected_id_for_failed_candidates = self.il_model_select_candidates(sent_candidates, 
                                                              edited_sent_candidates, 
                                                              sent_act_candidates, 
                                                              past_candidates, 
                                                              hidden_states_candidates,
                                                              failed_candidates)
            selected_T_F = [True if i in model_selected_id_for_success_candidates + model_selected_id_for_failed_candidates else False \
                            for i in range(len(edited_sent_candidates)+len(failed_candidates))]
            if len(model_selected_id_for_success_candidates) > 0:
                selected_i = np.random.choice(model_selected_id_for_success_candidates)
                sents, sent_acts, past = edited_sent_candidates[selected_i], sent_act_candidates[selected_i], \
                                                 past_candidates[selected_i]

            # elif len(model_selected_id_for_failed_candidates) > 0:
            #     selected_i = np.random.choice(model_selected_id_for_failed_candidates)
            #     failed_candidate = failed_candidates[selected_i-len(edited_sent_candidates)]
            #     sents, sent_acts, past = failed_candidate[0], failed_candidate[1], failed_candidate[3]

            else:
                sents, sent_acts, past, scores = self.select_candidates(edited_sent_candidates, sent_candidate_conflict_scores, sent_act_candidates, past_candidates)
            # if not cfg.self_play_mode:
            sents_success, sents_failed = self.print_candidates(sent_candidates, edited_sent_candidates, sent_act_candidates, selected_T_F, failed_candidates)
        else:
            sents, sent_acts, past, scores = self.select_candidates(edited_sent_candidates, sent_candidate_conflict_scores, sent_act_candidates, past_candidates)
            # if not cfg.self_play_mode:
            sents_success, sents_failed = self.print_candidates(sent_candidates, edited_sent_candidates, sent_act_candidates, scores, failed_candidates)
        
        # check conflict within the sents
        sents, sent_acts = self.check_conflict_within_selected_sents(sents, sent_acts)
        

        # update
        if mode != cfg.supervised_mode:
            self.global_profile.update(sents=sents, sent_labels=sent_acts, who=self.domain.SYS) #self.last_sys_labels = self.sys_profile.update(sys_texts=sents, sys_labels=sent_acts)

            # join sentences! finally!
            sent = " ".join(sents)
            # if sent == "ARDM MEMORY RESTARTS!":
            #     self.past = None
            #     return "ARDM MEMORY RESTARTS!"
            self.past = past
            # set past for model_clf
            self.model_clf.set_past(sent=sent, 
                                    which_task="A")

            # print("A:" + tokenizer.decode(sent))
            # finish tail
            prev_input = torch.LongTensor(self.eos).unsqueeze(0).to(self.device1)
            # pdb.set_trace()
            if self.past is not None and self.model_A.device != self.past[0].device:
                past = [p.to(self.model_A.device) for p in self.past]
                self.past = past

            _, self.past, hidden_states = self.model_A(prev_input, past=self.past)


        else:
            sent = " ".join(sents)

        return sent, [sents_success, sents_failed], have_enough_candidates

    def il_model_select_candidates(self, sent_candidates, 
                                   edited_sent_candidates, 
                                   sent_act_candidates, 
                                   past_candidates,
                                   hidden_states_candidates, 
                                   failed_candidates):
        left_id_for_success_candidates = []
        left_id_for_failed_candidates = []
        i = 0
        for sent, edited_sent, sent_act, past, hidden_state in zip(sent_candidates, edited_sent_candidates,
                                                                   sent_act_candidates, past_candidates, hidden_states_candidates):
            # pdb.set_trace()
            if type(edited_sent) is not list:
                edited_sent = [edited_sent]
            clf_predicted_acts, clf_past = self.model_clf.predict(separate_sents=edited_sent, 
                                                                  which_task="TF")
            
            # if hidden_state[-1] is not None:
            #     if hidden_state[-1].device != self.model_clf.device:
            #         last_hidden_state = hidden_state[-1].to(self.model_clf.device)
            #     else:
            #         last_hidden_state = hidden_state[-1]
            # outputs = self.model_clf(hidden_states=last_hidden_state)
            # _, predicted_label = torch.max(outputs, 1)

            if clf_predicted_acts == 1:#type(predicted_label.item()) is int and predicted_label.item() == 1:
                left_id_for_success_candidates.append(i)
            i += 1
        
        for sent, sent_act, reason, past, hidden_state in failed_candidates:
            # if hidden_state[-1] is not None and hidden_state[-1].device != self.model_clf.device:
            #     last_hidden_state = hidden_state[-1].to(self.model_clf.device)
            # outputs = self.model_clf(hidden_states=last_hidden_state)
            # _, predicted_label = torch.max(outputs, 1)

            # if type(predicted_label.item()) is int and predicted_label.item() == 1:
            #     left_id_for_failed_candidates.append(i)
            if type(sent) is not list:
                sent = [sent]
            clf_predicted_acts, clf_past = self.model_clf.predict(separate_sents=sent, 
                                                                  which_task="TF")
            if clf_predicted_acts == 1:
                left_id_for_failed_candidates.append(i)

        return left_id_for_success_candidates, left_id_for_failed_candidates

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
        # print("\n")
        logging.debug("\n")
        logging.debug("*************************** new dialog ****************************************")
        for turn_i in range(len(self.logs['responses'])):
            for k in ['responses', 'candidates', 'failed_candidates', 'global_profiles']:
                # print("{}\n".format(k))
                # print("-"*100)
                logging.debug("{}\n".format(k))
                if k in ['responses', 'candidates', 'failed_candidates']:
                    try:
                        if k == "failed_candidates":
                            for a in self.logs[k][turn_i]:
                                sents, sent_acts, fail_reason, past, hidden_states = a
                                logging.debug("------------failed_candidates: {} - reason: {}-----------".format(sent_acts, fail_reason))
                                logging.debug(sents)
                        for a in self.logs[k][turn_i]:
                            # print(a)
                            logging.debug(a)
                    except:
                        pdb.set_trace()
                else:
                    for world in self.logs[k][turn_i]:                        
                        for profile in self.logs[k][turn_i][world]:
                            # print("*******{}: {}*******".format(world, profile))
                            logging.debug("*******{}: {}*******".format(world, profile))
                            for key, value in self.logs[k][turn_i][world][profile].items():
                                # print("{}: {}".format(key, value))
                                logging.debug("{}: {}".format(key, value))
                            # print("")
                            logging.debug("")
            # print("\n")
            logging.debug("\n")
        
        logging.debug("*************************** dialog end ****************************************")
        
    def log_human_demonstration(self, sent_candidates, edited_sent_candidates, sent_act_candidates, past_candidates, hidden_states_candidates,
                                      failed_candidates, human_selected_ids,
                                      human_added_candidates=None):
        # records = {}
        sents_success, sents_failed = [], []
        csv_records = []
        context = deepcopy(self.global_profile.history)
        context_act = deepcopy(self.global_profile.history_label)
        hidden_states_before_generation = deepcopy(self.b_hidden_states) if self.b_hidden_states is None else self.b_hidden_states[-1].clone().detach()#deepcopy(self.b_hidden_states)
        past_before_generation = deepcopy(self.past) if self.past is None else self.past[-1].clone().detach()#[p.clone().detach() for p in self.past]#deepcopy(self.past)
        sys_world_sys_profile = deepcopy(self.global_profile.sys_world.sys_profile)
        sys_world_usr_profile = deepcopy(self.global_profile.sys_world.usr_profile)
        usr_world_sys_profile = deepcopy(self.global_profile.usr_world.sys_profile)
        usr_world_usr_profile = deepcopy(self.global_profile.usr_world.usr_profile)

        shared_features = {'context': context,
                      'context_act': context_act,
                      'hidden_states_before_generation': hidden_states_before_generation,
                      'past_before_generation': past_before_generation,
                      'sys_world_sys_profile': sys_world_sys_profile,
                      'sys_world_usr_profile': sys_world_usr_profile,
                      'usr_world_sys_profile': usr_world_sys_profile,
                      'usr_world_usr_profile': usr_world_usr_profile}
        records = {'shared_features': shared_features,
                    'individual_features': []}
        i = 0
        for sent, edited_sent, sent_act, past, hidden_state in zip(sent_candidates, edited_sent_candidates,
                                                                   sent_act_candidates, past_candidates, hidden_states_candidates):
            # pdb.set_trace()
            record = {
                      'hidden_states_after_generation': hidden_state[-1].clone().detach(),#deepcopy(hidden_state),
                      'past_after_generation': past[-1].clone().detach(),#[p.clone().detach() for p in past],
                      'sent': sent,
                      'edited_sent': edited_sent,
                      'sent_act': sent_act,
                      'different_from_edition': sent != edited_sent,
                      'failed_candidates': False,
                      'pick_or_not': (i in human_selected_ids), 
                      }
            csv_record = [context, context_act, sys_world_sys_profile, sys_world_usr_profile, usr_world_sys_profile, usr_world_usr_profile, 
                          sent, edited_sent, sent_act, sent != edited_sent, False, i in human_selected_ids]
            records['individual_features'].append(record)
            csv_records.append(csv_record)
            i += 1
            if i in human_selected_ids:
                sents_success.append(edited_sent)
            else:
                sents_failed.append(edited_sent)

        for sent, sent_act, reason, past, hidden_state in failed_candidates:
            record = {
                      'hidden_states_after_generation': hidden_state[-1].clone().detach(),#deepcopy(hidden_state),
                      'past_after_generation': past[-1].clone().detach(),#[p.clone().detach() for p in past],,
                      'sent': sent,
                      'edited_sent': sent,
                      'sent_act': sent_act,
                      'different_from_edition': reason,
                      'failed_candidates': True,
                      'pick_or_not': (i in human_selected_ids), 
                      }
            csv_record = [context, context_act, sys_world_sys_profile, sys_world_usr_profile, usr_world_sys_profile, usr_world_usr_profile, 
                          sent, sent, sent_act, reason, True, i in human_selected_ids]
            records['individual_features'].append(record)
            csv_records.append(csv_record)
            i += 1
            if i in human_selected_ids:
                sents_success.append(sent)
            else:
                sents_failed.append(sent)


        if human_added_candidates is not None:
            for sent, sent_act, past, hidden_state in human_added_candidates:
                record = {
                        'hidden_states_after_generation': hidden_state[-1].clone().detach(),#deepcopy(hidden_state),
                        'past_after_generation': past[-1].clone().detach(),#[p.clone().detach() for p in past],,
                        'sent': sent,
                        'edited_sent': sent,
                        'sent_act': sent_act,
                        'different_from_edition': "human_added_sentence",
                        'failed_candidates': False,
                        'pick_or_not': True, 
                        }
                csv_record = [context, context_act, sys_world_sys_profile, sys_world_usr_profile, usr_world_sys_profile, usr_world_usr_profile, 
                            sent, sent, sent_act, "human_added_sentence", False, True]
                records['individual_features'].append(record)
                csv_records.append(csv_record)
                i += 1
                sents_success.append(sent)


        if not os.path.exists(cfg.demonstration_csv):
            with open(cfg.demonstration_csv, 'w') as fh:
                writer = csv.writer(fh)
                writer.writerow(["context", "context_act", 'sys_world_sys_profile', 'sys_world_usr_profile', 'usr_world_sys_profile', 'usr_world_usr_profile',
                                "sent", "edited_sent", "sent_act", "different_from_edition", "failed_candidates", "pick_or_not"])
                

        with open(cfg.demonstration_csv, "a") as fh:
            writer = csv.writer(fh)
            writer.writerow("\n")
            writer.writerows(csv_records)

        if os.path.exists(cfg.demonstration_pkl):
            with open(cfg.demonstration_pkl, "rb") as fh:
                prev_records = pkl.load(fh)

            records = prev_records + [records]
            with open(cfg.demonstration_pkl, "wb") as fh:
                pkl.dump(records, fh)

        else:           
            with open(cfg.demonstration_pkl, "wb") as fh:
                pkl.dump([records], fh)
        
        return sents_success, sents_failed

if __name__ == "__main__":
    logging.basicConfig(filename=cfg.log_file,level=logging.DEBUG)
    

    bot = PersuasiveBot()
    pdb.set_trace()
    # bot.reload()
    user_text = ""
    # signal.signal(signal.SIGINT, signal.default_int_handler)

    MAX_DIALOGS = 2
    dial_i = 0
    while dial_i < MAX_DIALOGS:
        try:
            if bot.past is not None:
                if cfg.mode != cfg.self_play_mode:
                    user_text  = input("user: ")
                else:
                    user_text = None
            else:
                dial_i += 1
                print("\n\n\n")
                print("INIT MEMORY!")
                bot.save()
                bot.reload()
            

            result = bot.chat(input_text=user_text, mode=cfg.mode)
            if result is not None:
                response, [sents_success, sents_failed], have_enough_candidates = result
            if cfg.candidate_select_strategy != cfg.HUMAN_SELECTION:
                if cfg.verbose:
                    bot.global_profile.print()
            
            # if response == "ARDM MEMORY RESTARTS!":
            #     print("ARDM MEMORY RESTARTS!")
            # else:
            if result is not None:
                print("Turn {}".format(bot.turn_i))
                print("system: ", response)
            print("$$$$$$$$$$$$$$$$$$$$$")

        except KeyboardInterrupt:
            bot.save()
            sys.exit()

        