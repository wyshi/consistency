import os
from pathlib import Path

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
from PPO import load_model
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
    LOG_FILE = "Eval/me_talking_to_the_bot_to_decide_which_to_use_for_RL_good_model.log"
    class CurrentModelConfig:
        with_rule = True
        log_file = LOG_FILE
        
        with_baseline =  True
        with_repetition_module = True
        with_consistency_module = True
        with_sentence_clf = True
        with_RL_finetune_model = False

        if not with_repetition_module and with_consistency_module:
            candidate_select_strategy = cfg.RANDOM_SELECT
        elif not with_repetition_module and not with_consistency_module:
            candidate_select_strategy = cfg.RANDOM_SELECT
        elif with_repetition_module and not with_consistency_module:
            candidate_select_strategy = cfg.REPETITION_RATIO
        elif with_repetition_module and with_consistency_module:
            candidate_select_strategy = cfg.REPETITION_RATIO

        if with_sentence_clf:
            candidate_select_strategy = cfg.IMITATION_LEARNING_SELECTION

        if with_baseline and (not with_repetition_module) and (not with_consistency_module) and (not with_sentence_clf)\
            and (not with_RL_finetune_model):
            NUM_CANDIDATES = 1
            with_rule = False
        else:
            NUM_CANDIDATES = cfg.NUM_CANDIDATES
    
    torch.cuda.empty_cache()
    bot = PersuasiveBot(CurrentModelConfig, model_A=model_A, model_B=model_B, tokenizer=TOKENIZER, 
                        device1=DEVICE1, device2=DEVICE2)

    print(f"with_baseline: {CurrentModelConfig.with_baseline}")
    print(f"with_repetition_module: {CurrentModelConfig.with_repetition_module}")
    print(f"with_consistency_module: {CurrentModelConfig.with_consistency_module}")
    print(f"with_sentence_clf: {CurrentModelConfig.with_sentence_clf}")
    print(f"with_RL_finetune_model: {CurrentModelConfig.with_RL_finetune_model}")
    print(f"candidate_select_strategy: {CurrentModelConfig.candidate_select_strategy}")
    print(f"NUM_CANDIDATES: {CurrentModelConfig.NUM_CANDIDATES}")
    print(f"with_rule: {CurrentModelConfig.with_rule}")

    # calculate_num_success_candidates(bot, MAX_DIALOGS=3, mode=cfg.self_play_mode)

    calculate_num_success_candidates(bot, MAX_DIALOGS=100, mode=cfg.interactive_mode)

