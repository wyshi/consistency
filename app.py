from __future__ import absolute_import, division, print_function
from flask import Flask, request, jsonify
# from service_missa import HuggingfaceModel
import json
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

TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")#torch.load(tokenizer_dir)
EVAL_MODEL_A_DIR = "/home/wyshi/persuasion/consistency/ARDM/persuasion/persuasion_medium_3.th"

DEVICE1 = torch.device(cfg.model_A_device)
DEVICE1_list = cfg.model_A_device_list
SPLIT_INTO1= cfg.split_into_A

DEVICE2 = torch.device(cfg.model_B_device)
DEVICE2_list = cfg.model_B_device_list
SPLIT_INTO2= cfg.split_into_B
# val_dataloader = get_val_dataloader(TOKENIZER)


model_A, model_B = load_model(cfg=cfg, device1=DEVICE1, device2=DEVICE2, split_into1=SPLIT_INTO1, split_into2=SPLIT_INTO2,
                             dropout=0, device_list1=DEVICE1_list, device_list2=DEVICE2_list,
                                model_A_dir=EVAL_MODEL_A_DIR, use_old_model_B=False)

model_A.eval()
model_B.eval()

cfg.rl_finetune = False
cfg.candidate_select_strategy = cfg.IMITATION_LEARNING_SELECTION
model = PersuasiveBot(model_A=model_A, model_B=model_B, tokenizer=TOKENIZER, 
                    device1=DEVICE1, device2=DEVICE2)



app = Flask(__name__)
# model = HuggingfaceModel("./runs/1000pretrained")
model.reload()

def end_condition(usr_input):

    return False


@app.route("/missa", methods=['POST'])
def getResponse():
    sid = request.json.get('sid')
    input_text = request.json.get('input_text')
    print(sid)

    # added by me
    # user_text = ""
    MODE = cfg.interactive_mode

    result = model.chat(input_text=input_text, mode=MODE, sid=sid)
    if result is not None:
        response, [sents_success, sents_failed], have_enough_candidates, usr_input_text = result
        # TOTAL_SUCCESS_CANDIDATES += len(sents_success)


    # [output_text, sys_da_output, sys_se_output, usr_da_output, usr_se_outpu] = model.chat(input_text, sid)
    return jsonify({"response": response, 
                    "exitbutton_or_not": exitbutton_or_not,              
                    "sents_success": sents_success, 
                    "sents_failed": sents_failed, 
                    "have_enough_candidates": have_enough_candidates, 
                    "usr_input_text": usr_input_text})
    #return jsonify(ed_result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8089)
    #socketio.run(app, host='0.0.0.0', port = 8087, use_reloader=False)
