import os
from pathlib import Path
from PersuasionInteract import top_filtering, sent_tokenize_modified
# paths = sorted(Path("Checkpoint").iterdir(), key=os.path.getmtime)
# rewards = []
# for p in paths:
#     p = str(p)
#     if "steps" in p and "exception" not in p:
#         rewards.append(str(p.split("_steps_")[1].split("_")[0])+"\n")

# with open("Checkpoint/rewards.txt", "w") as fh:
#     fh.writelines(rewards)

### ppl
import os
import time
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from GPTModel1 import GPT2LMHeadModel_modified
from PPO import load_model, Actor
from PersuasionInteract import PersuasiveBot
import config as cfg
import sys
from torchfly.modules.losses import SequenceFocalLoss, SequenceCrossEntropyLoss

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

def calculate_num_success_candidates(bot, MAX_DIALOGS, mode):
    dialogs_list = []
    cur_dialog = []
    user_text = ""
    # signal.signal(signal.SIGINT, signal.default_int_handler)
    MODE = mode

    # MAX_DIALOGS = 5
    TOTAL_TURNS = 0
    TOTAL_SUCCESS_CANDIDATES = 0
    dial_i = 0

    while dial_i < MAX_DIALOGS:
        try:
            if bot.past is not None:
                if MODE != cfg.self_play_mode:
                    user_text  = input("user: ")
                else:
                    user_text = None
            else:
                dial_i += 1
                print("\n\n\n")
                print("INIT MEMORY!")
                dialogs_list.append(cur_dialog)
                cur_dialog = []
                bot.save()
                bot.reload()
            

            result = bot.chat(input_text=user_text, mode=MODE)
            if result is not None:
                TOTAL_TURNS += 1
                response, [sents_success, sents_failed], have_enough_candidates, usr_input_text = result
                TOTAL_SUCCESS_CANDIDATES += len(sents_success)
            if CurrentModelConfig.candidate_select_strategy != cfg.HUMAN_SELECTION:
                if cfg.verbose:
                    bot.global_profile.print()
            
            # if response == "ARDM MEMORY RESTARTS!":
            #     print("ARDM MEMORY RESTARTS!")
            # else:
            if result is not None:
                print("Turn {}".format(bot.turn_i))
                print("system: ", response)
                if usr_input_text is not None:
                    cur_dialog.append("usr: " + usr_input_text)
                cur_dialog.append("sys: " + response)
            print("$$$$$$$$$$$$$$$$$$$$$")

        except KeyboardInterrupt:
            bot.save()
            break

    with open("Eval/simulated_dialogs.txt", "a") as fh:
        for i, dialog in enumerate(dialogs_list):
            if dialog != []:
                fh.write(f"dialog {i}:\n")
                fh.writelines([d+"\n" for d in dialog])
                fh.write(f"-------------------------\n")
            
        print(f"finally {TOTAL_SUCCESS_CANDIDATES}, {TOTAL_TURNS}, num dialogs {dial_i-1}")
        fh.write(f"finally {TOTAL_SUCCESS_CANDIDATES}, {TOTAL_TURNS}, num dialogs {dial_i-1}\n")

def calculate_num_success_candidates_for_eval_set(bot, dataloader, mode):
    dialogs_list = []
    cur_dialog = []
    user_text = ""
    # signal.signal(signal.SIGINT, signal.default_int_handler)
    MODE = mode

    # MAX_DIALOGS = 5
    TOTAL_TURNS = 0
    TOTAL_SUCCESS_CANDIDATES = 0
    dial_i = 0

    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader)

        total_ppl_A, total_ppl_B = [], []

        for batch in pbar:
            
            # if sum([len(item) for item in batch[0][1]]) > 1024:
            #     continue
            
            role_ids, dialog_tokens = batch[0]
            # dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]

            past = None
            # all_logits = []
            A_logits = []
            B_logits = []
            A_target = []
            B_target = []

            for turn_num, dial_turn_inputs in enumerate(dialog_tokens):
                if role_ids[turn_num] == 0:
                    device = model_A.device
                else:
                    device = model_B.device
                dial_turn_inputs = torch.LongTensor(dial_turn_inputs).unsqueeze(0).to(device)

                if role_ids[turn_num] == 0:
                    bot.chat()
                    logits, past, _ = model_A(dial_turn_inputs, past=past)
                    A_logits.append(logits)
                    A_target.append(dial_turn_inputs)
                    # all_logits.append(logits)
                else:
                    logits, past, _ = model_B(dial_turn_inputs, past=past)
                    B_logits.append(logits)
                    B_target.append(dial_turn_inputs)
                    # all_logits.append(logits)




    while dial_i < MAX_DIALOGS:
        try:
            if bot.past is not None:
                if MODE != cfg.self_play_mode:
                    user_text  = input("user: ")
                else:
                    user_text = None
            else:
                dial_i += 1
                print("\n\n\n")
                print("INIT MEMORY!")
                dialogs_list.append(cur_dialog)
                cur_dialog = []
                bot.save()
                bot.reload()
            

            result = bot.chat(input_text=user_text, mode=MODE)
            if result is not None:
                TOTAL_TURNS += 1
                response, [sents_success, sents_failed], have_enough_candidates, usr_input_text = result
                TOTAL_SUCCESS_CANDIDATES += len(sents_success)
            if CurrentModelConfig.candidate_select_strategy != cfg.HUMAN_SELECTION:
                if cfg.verbose:
                    bot.global_profile.print()
            
            # if response == "ARDM MEMORY RESTARTS!":
            #     print("ARDM MEMORY RESTARTS!")
            # else:
            if result is not None:
                print("Turn {}".format(bot.turn_i))
                print("system: ", response)
                if usr_input_text is not None:
                    cur_dialog.append("usr: " + usr_input_text)
                cur_dialog.append("sys: " + response)
            print("$$$$$$$$$$$$$$$$$$$$$")

        except KeyboardInterrupt:
            bot.save()
            break

    with open("Eval/simulated_dialogs.txt", "a") as fh:
        for i, dialog in enumerate(dialogs_list):
            if dialog != []:
                fh.write(f"dialog {i}:\n")
                fh.writelines([d+"\n" for d in dialog])
                fh.write(f"-------------------------\n")
            
        print(f"finally {TOTAL_SUCCESS_CANDIDATES}, {TOTAL_TURNS}, num dialogs {dial_i-1}")
        fh.write(f"finally {TOTAL_SUCCESS_CANDIDATES}, {TOTAL_TURNS}, num dialogs {dial_i-1}\n")


def get_val_dataloader(tokenizer):
    val_data = torch.load("DataProcess/val_dialogs.pkl")
    val_dataset = PersuadeDataset(val_data, tokenizer)
    BATCH_SIZE = 1
    val_dataloader = DataLoader(dataset=val_dataset, 
                                shuffle=False, 
                                batch_size=BATCH_SIZE, 
                                collate_fn=val_dataset.collate)
    return val_dataloader

def load_original_model_B(cfg, device2, split_into, dropout):
    if cfg.model_size == "small":
        model_B = GPT2LMHeadModel_modified.from_pretrained("gpt2", output_hidden_states=True,
                                                            resid_pdrop =dropout,
                                                            embd_pdrop  =dropout,
                                                            attn_pdrop  =dropout,
                                                            summary_first_dropout=dropout
        )
        model_B.set_variables(device=device2, split_into=split_into)
    elif cfg.model_size == "medium":
        model_B = GPT2LMHeadModel_modified.from_pretrained("gpt2-medium", output_hidden_states=True,
                                                            resid_pdrop =dropout,
                                                            embd_pdrop  =dropout,
                                                            attn_pdrop  =dropout,
                                                            summary_first_dropout=dropout
                                                            )
        model_B.set_variables(device=device2, split_into=split_into)

   # pdb.set_trace()
    # load the model
    if cfg.model_size == "small":
        if cfg.use_old_model:
            _, model_B_states = torch.load(cfg.old_medium_model_dir)
            # model_A_states['transformer.wte.weight'] = model_A_states['transformer.wte.weight'][:50257,:]
            # model_A_states['lm_head.weight'] = model_A_states['lm_head.decoder.weight'][:50257,:]
            model_B_states['transformer.wte.weight'] = model_B_states['transformer.wte.weight'][:50257,:]
            model_B_states['lm_head.weight'] = model_B_states['lm_head.decoder.weight'][:50257,:]
            print("loaded old small model")
        else:
            _, model_B_states = torch.load(cfg.new_small_model_dir)
            print("loaded new small model")
    elif cfg.model_size == "medium":
        if cfg.use_old_model:
            _, model_B_states = torch.load(cfg.old_medium_model_dir)
            # model_A_states['transformer.wte.weight'] = model_A_states['transformer.wte.weight'][:50257,:]
            # model_A_states['lm_head.weight'] = model_A_states['lm_head.decoder.weight'][:50257,:]
            model_B_states['transformer.wte.weight'] = model_B_states['transformer.wte.weight'][:50257,:]
            model_B_states['lm_head.weight'] = model_B_states['lm_head.decoder.weight'][:50257,:]
            print("loaded old medium model")
        else:
            _, model_B_states = torch.load(cfg.new_medium_model_dir)
            print("loaded new medium model")
    
    if cfg.use_old_model:
        # strict_A = False
        strict_B = False
    else:
        # strict_A = True
        strict_B = True
    # model_A.load_state_dict(model_A_states, strict=strict_A)
    model_B.load_state_dict(model_B_states, strict=strict_B)

    # to device
    # model_A.to(device1)
    model_B.to(device2)
 
    return model_B


def sample_one_sent(past, model, prefix="A:"):
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
        for i in tqdm.tqdm(range(self.max_sequence_len)):
            # try:
            # pdb.set_trace()
            logits, past, hidden_states = model(prev_input, past=past)
            
            logits = logits[:, -1, :] / self.temperature
            logits = top_filtering(logits, top_k=500, top_p=0.9)
            # prev_input = logits.argmax(-1).unsqueeze(1)
            probs = F.softmax(logits, -1)
            
            
            prev_input = torch.multinomial(probs, num_samples=1)
            prev_word = prev_input.item()
            # except:
            #     pdb.set_trace()  

            if prev_word == 628:
                break
            # elif prev_word == self.tokenizer.encoder["[EOS]"]:
            #     past = None
            #     return "ARDM MEMORY RESTARTS!", past
            #     break
            else:
                sent.append(prev_word)
    return self.tokenizer.decode(sent), past, hidden_states

def validate(dataloader, model_A, model_B, ep=0):
    eval_criterion = SequenceCrossEntropyLoss()
    # device = 

    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader)

        total_ppl_A, total_ppl_B = [], []

        for batch in pbar:
            
            # if sum([len(item) for item in batch[0][1]]) > 1024:
            #     continue
            
            role_ids, dialog_tokens = batch[0]
            # dial_inputs = [torch.LongTensor(item).unsqueeze(0).to(device) for item in dialog_tokens]

            past = None
            # all_logits = []
            A_logits = []
            B_logits = []
            A_target = []
            B_target = []

            for turn_num, dial_turn_inputs in enumerate(dialog_tokens):
                if role_ids[turn_num] == 0:
                    device = model_A.device
                else:
                    device = model_B.device
                dial_turn_inputs = torch.LongTensor(dial_turn_inputs).unsqueeze(0).to(device)

                if role_ids[turn_num] == 0:
                    logits, past, _ = model_A(dial_turn_inputs, past=past)
                    A_logits.append(logits)
                    A_target.append(dial_turn_inputs)
                    # all_logits.append(logits)
                else:
                    logits, past, _ = model_B(dial_turn_inputs, past=past)
                    B_logits.append(logits)
                    B_target.append(dial_turn_inputs)
                    # all_logits.append(logits)

            A_logits = torch.cat(A_logits, dim=1)
            B_logits = torch.cat(B_logits, dim=1)
            
            # target
            A_logits = A_logits[:, :-1].contiguous()
            B_logits = B_logits[:, :-1].contiguous()
            A_target = torch.cat(A_target, dim=1)[:, 1:].contiguous()
            B_target = torch.cat(B_target, dim=1)[:, 1:].contiguous()
            A_target_mask = torch.ones_like(A_target).float()
            B_target_mask = torch.ones_like(B_target).float()
            
            loss_A = eval_criterion(A_logits, A_target, A_target_mask, label_smoothing=-1, reduce="sentence")      
            loss_B = eval_criterion(B_logits, B_target, B_target_mask, label_smoothing=-1, reduce="sentence")      

            ppl_A = torch.exp(loss_A)
            ppl_B = torch.exp(loss_B)
            total_ppl_A.extend(ppl_A.tolist())
            total_ppl_B.extend(ppl_B.tolist())

        with open("Eval/simulated_dialogs.txt", "a") as fh:
            print(f"Epoch {ep} Validation Perplexity A: {np.mean(total_ppl_A)} Variance: {np.var(total_ppl_A)}")
            print(f"Epoch {ep} Validation Perplexity B: {np.mean(total_ppl_B)} Variance: {np.var(total_ppl_B)}")
            fh.write(f"Epoch {ep} Validation Perplexity A: {np.mean(total_ppl_A)} Variance: {np.var(total_ppl_A)}\n")
            fh.write(f"Epoch {ep} Validation Perplexity B: {np.mean(total_ppl_B)} Variance: {np.var(total_ppl_B)}\n")

        return np.mean(total_ppl_A), np.mean(total_ppl_B)

from copy import deepcopy
import logging
class EvalActor(Actor):
    def __init__(self, model_config, model_A, model_B, tokenizer, device1, device2, dialog_i):
        super().__init__(model_config=model_config,
                         model_A=model_A, model_B=model_B, 
                         tokenizer=tokenizer, 
                         device1=device1, device2=device2, 
                         dialog_i=dialog_i)

    def evaluation_valset(self):
        """
        the sample unit is one dialog, sample_size=1 means at a time we sample one dialog
        """
        self.model_A.eval()
        self.model_B.eval()
        
        final_contexts, final_sents, final_rewards, final_context_ids = [], [], [], []
        final_targets_A, final_responses_A = [], []
        final_targets_B, final_responses_B = [], []
        TOTAL_NUM_SUCCESS_SENTS = []
        TOTAL_NUM_TURNS = []
        with torch.no_grad():
            for i in range(len(self.val_dataset)):
                mode = cfg.supervised_mode
                logging.info(f"in mode: {mode}")
                if mode == cfg.supervised_mode:
                    batch = self.val_dataset[i]
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
                            final_responses_A.append(response)
                            final_targets_A.append(dial_sent)
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
                            _, self.past, hidden_states = self.model_A(dial_turn_inputs, past=self.past)
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
                            logging.info(f"success candidates avg len: {np.mean([len(one_sent.split()) for one_sent in sents_success])}")
                            logging.info(f"failed candidates: {sents_failed}")
                            logging.info(f"failed candidates avg len: {np.mean([len(one_sent.split()) for one_sent in sents_failed])}")
                            logging.info(f"----------------------")
                            self.contexts.append("A:"+ground_truth)

                        else:
                            # breakpoint()
                            user_text = dial_sent

                            generated_user_text, _, _ = self.sample_one_sent(past=self.past, model=self.model_B, prefix="B:")
                            final_responses_B.append(generated_user_text)
                            final_targets_B.append(dial_sent)
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
                    logging.info(f"avg success sent: {NUM_SUCCESS_SENTS/NUM_TURNS}")
                    TOTAL_NUM_SUCCESS_SENTS.append(NUM_SUCCESS_SENTS)
                    TOTAL_NUM_TURNS.append(NUM_TURNS)
                    # finish tail
                    if role_id == 1: # the last sent is user
                        # throw away the last user sentence
                        pass

                self.dialog_i += 1
                self.reload()
            assert len(final_contexts) == len(final_sents) == len(final_rewards) == len(final_context_ids)
            return final_contexts, final_sents, final_rewards, final_context_ids,\
                   final_targets_A, final_responses_A, final_targets_B, final_responses_B,\
                       TOTAL_NUM_SUCCESS_SENTS, TOTAL_NUM_TURNS



# load models
TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")#torch.load(tokenizer_dir)
# EVAL_MODEL_A_DIR = "/home/wyshi/persuasion/consistency/ARDM/persuasion/persuasion_medium_3.th"
# EVAL_MODEL_A_DIR = "/data/wyshi/persuasion/consistency/Checkpoint/first_train-32,32*3, 256,1e-2/48_steps_2.727272727272727_reward_model_A_kl_5.73.pth"#good, but with "what's your name" questions
# EVAL_MODEL_A_DIR = "/home/wyshi/persuasion/consistency/Checkpoint/1_steps_1.86_2.6_reward_model_A_kl_13.28_ppo5.pth"#not so good
EVAL_MODEL_A_DIR = "/home/wyshi/persuasion/consistency/Checkpoint/23_steps_1.79_2.536363636363636_reward_model_A_kl_7.53_ppo5.pth"#good
# EVAL_MODEL_A_DIR = "/home/wyshi/persuasion/consistency/Checkpoint/33_steps_1.76_2.5090909090909093_reward_model_A_kl_1.2_ppo4.pth"#bad
# EVAL_MODEL_A_DIR = "models/persuasion-gpt2-medium.pth"
with open("Eval/simulated_dialogs.txt", "a") as fh:
    fh.write(f"{EVAL_MODEL_A_DIR}\n")

DEVICE1 = torch.device("cuda:2")#torch.device(cfg.model_A_device)
DEVICE1_list = ["cuda:2"]
SPLIT_INTO1= 1

DEVICE2 = torch.device("cuda:3")
DEVICE2_list = ['cuda:3']
SPLIT_INTO2= 1
val_dataloader = get_val_dataloader(TOKENIZER)


model_A, model_B = load_model(cfg=cfg, device1=DEVICE1, device2=DEVICE2, split_into1=SPLIT_INTO1, split_into2=SPLIT_INTO2,
                             dropout=0, device_list1=DEVICE1_list, device_list2=DEVICE2_list,
                                model_A_dir=EVAL_MODEL_A_DIR, use_old_model_B=False)

model_A.eval()
model_B.eval()
import pdb
# pdb.set_trace()
validate(val_dataloader, model_A, model_B, ep=0)

# try:

# so that the human rule can be applied
# cfg.rl_finetune = False
# cfg.candidate_select_strategy = cfg.IMITATION_LEARNING_SELECTION

if False:
    # del model_B
    # torch.cuda.empty_cache()

    model_B_old_for_selfplay = load_original_model_B(cfg=cfg, device2=DEVICE2, split_into=SPLIT_INTO, dropout=0)
    model_B_old_for_selfplay.eval()
    bot = PersuasiveBot(model_A=model_A, model_B=model_B_old_for_selfplay, tokenizer=TOKENIZER, 
                        device1=DEVICE1, device2=DEVICE2)
    calculate_num_success_candidates(bot, MAX_DIALOGS=3, mode=cfg.self_play_mode)
    del model_B_old_for_selfplay
else:
    # model_B_old_for_selfplay = load_original_model_B(cfg=cfg, device2=DEVICE2, split_into=SPLIT_INTO)
    pass


if True:
    LOG_FILE = "Eval/automatic_metrics.log"
    class CurrentModelConfig:
        with_rule = True
        log_file = LOG_FILE
        
        with_baseline =  True
        with_repetition_module = False
        with_consistency_module = False
        with_sentence_clf = False
        with_RL_finetune_model = False

        candidate_select_strategy = cfg.RANDOM_SELECT
        candidate_select_strategy = cfg.REPETITION_RATIO
        candidate_select_strategy = cfg.IMITATION_LEARNING_SELECTION

        NUM_CANDIDATES = 1

        # if not with_repetition_module and with_consistency_module:
        #     candidate_select_strategy = cfg.RANDOM_SELECT
        # elif not with_repetition_module and not with_consistency_module:
        #     candidate_select_strategy = cfg.RANDOM_SELECT
        # elif with_repetition_module and not with_consistency_module:
        #     candidate_select_strategy = cfg.REPETITION_RATIO
        # elif with_repetition_module and with_consistency_module:
        #     candidate_select_strategy = cfg.REPETITION_RATIO

        # if with_sentence_clf:
        #     candidate_select_strategy = cfg.IMITATION_LEARNING_SELECTION

        # if with_baseline and (not with_repetition_module) and (not with_consistency_module) and (not with_sentence_clf)\
        #     and (not with_RL_finetune_model):
        #     NUM_CANDIDATES = 1
        #     with_rule = False
        # else:
        #     NUM_CANDIDATES = cfg.NUM_CANDIDATES
    
    torch.cuda.empty_cache()
    print(f"with_baseline: {CurrentModelConfig.with_baseline}")
    print(f"with_repetition_module: {CurrentModelConfig.with_repetition_module}")
    print(f"with_consistency_module: {CurrentModelConfig.with_consistency_module}")
    print(f"with_sentence_clf: {CurrentModelConfig.with_sentence_clf}")
    print(f"with_RL_finetune_model: {CurrentModelConfig.with_RL_finetune_model}")
    print(f"candidate_select_strategy: {CurrentModelConfig.candidate_select_strategy}")
    print(f"NUM_CANDIDATES: {CurrentModelConfig.NUM_CANDIDATES}")
    print(f"with_rule: {CurrentModelConfig.with_rule}")


    # calculate_num_success_candidates(bot, MAX_DIALOGS=3, mode=cfg.self_play_mode)

    # bot = PersuasiveBot(CurrentModelConfig, model_A=model_A, model_B=model_B, tokenizer=TOKENIZER, 
    #                     device1=DEVICE1, device2=DEVICE2)

    # calculate_num_success_candidates(bot, MAX_DIALOGS=100, mode=cfg.interactive_mode)

    # for automatic evaluation of each model
    actor = EvalActor(CurrentModelConfig, model_A=model_A, model_B=model_B, tokenizer=TOKENIZER, 
                  device1=DEVICE1, device2=DEVICE2, dialog_i=0)
    _, _, _, _, final_targets_A, final_responses_A, final_targets_B, final_responses_B,\
                       TOTAL_NUM_SUCCESS_SENTS, TOTAL_NUM_TURNS = actor.evaluation_valset()
    pdb.set_trace()
import collections
import nltk
from nltk import ngrams
# automatic metrics
def compute_distinct(inputs, n=4):
    counter = collections.Counter()
    total_count = 0
   
    for item in inputs:
        hyp = nltk.word_tokenize(item[1].lower())
        n_grams = list(ngrams(hyp, n=n))  
        counter.update(n_grams)
        total_count += len(n_grams)
    return len(counter) / total_count

def get_human_n_grams(inputs, n=4):
    human_n_grams = collections.Counter()

    for item in tqdm.tqdm(inputs):
        list_n_grams = ngrams(nltk.word_tokenize(item.lower()), n=n)
        human_n_grams.update(list_n_grams)
       
    human_n_grams = {k:v for k,v in human_n_grams.items() if v > 1}
    return human_n_grams

def compute_sentence_repeat(inputs, human_n_grams, n=4):
    scores = []
    for item in inputs:
        count = 0
        tokens = nltk.word_tokenize(item[1].lower())
        n_grams = list(ngrams(tokens, n=n))
        for n_gram in n_grams:
            if n_gram in human_n_grams:
                count += 1
        if len(n_grams) == 0:
            scores.append(0)
        else:
            scores.append(count/len(n_grams))

    return np.mean(scores)

from nltk.translate.bleu_score import sentence_bleu
def compute_bleu(inputs, n=2):
    if n==3:
        weights=(0.333, 0.333, 0.333, 0)
    elif n==2:
        weights=(0.5, 0.5, 0.0, 0)
    elif n==4:
        weights=(0.25, 0.25, 0.25, 0.25)
    else:
        assert False

    scores = []
   
    for item in inputs:
        ref = nltk.word_tokenize(item[0].lower())
        hyp = nltk.word_tokenize(item[1].lower())
       
        score = sentence_bleu([ref], hyp, weights=weights)
        scores.append(score)
    return np.mean(scores)