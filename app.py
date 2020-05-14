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
import logging

EVAL_MODEL_A_DIR = "/home/wyshi/persuasion/consistency/ARDM/persuasion/persuasion_medium_3.th"
DEVICE1 = torch.device("cuda:5")
DEVICE1_list = ["cuda:5"]
SPLIT_INTO1= 1

DEVICE2 = torch.device("cuda:6")
DEVICE2_list = ["cuda:6"]
SPLIT_INTO2= 1

class CurrentModelConfig:
    with_rule = True
    log_file = 'logs/amt_baseline_test_app.log'
    
    with_baseline =  True
    with_repetition_module = False
    with_consistency_module = False
    with_sentence_clf = False
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
    else:
        NUM_CANDIDATES = cfg.NUM_CANDIDATES

def load_model_for_AMT(EVAL_MODEL_A_DIR):
    TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")#torch.load(tokenizer_dir)

    # val_dataloader = get_val_dataloader(TOKENIZER)
    model_A, model_B = load_model(cfg=cfg, device1=DEVICE1, device2=DEVICE2, 
                                split_into1=SPLIT_INTO1, split_into2=SPLIT_INTO2,
                                dropout=0, device_list1=DEVICE1_list, device_list2=DEVICE2_list,
                                model_A_dir=EVAL_MODEL_A_DIR, use_old_model_B=False)

    model_A.eval()
    model_B.eval()

    return model_A, model_B, TOKENIZER, DEVICE1, DEVICE2

model_A, model_B, TOKENIZER, DEVICE1, DEVICE2 = load_model_for_AMT(EVAL_MODEL_A_DIR)

model = PersuasiveBot(model_config=CurrentModelConfig, 
                      model_A=model_A, model_B=model_B, tokenizer=TOKENIZER, 
                      device1=DEVICE1, device2=DEVICE2)

import time
import datetime
TIME = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
logging.info(f"!!!!!--------- AMT test: datetime {TIME}----------")
app = Flask(__name__)
# model = HuggingfaceModel("./runs/1000pretrained")
model.reload()

def end_condition(usr_input):

    return False

def delay_for_typing(RECEIVED_TIME, response):
    response_len = len(response)
    AVG_TIME_TO_TYPE = 210/60
    TIME_TO_TYPE_RESPONSE = response_len/AVG_TIME_TO_TYPE

    RESPONDED_TIME = time.time()

    TIME_ALREADY_PASSED = RESPONDED_TIME - RECEIVED_TIME
    
    TIME_TO_SLEEP = TIME_TO_TYPE_RESPONSE - TIME_ALREADY_PASSED

    if TIME_TO_SLEEP > 0:
        TIME_TO_SLEEP = min(TIME_TO_SLEEP, 30)
        time.sleep(TIME_TO_SLEEP)

@app.route("/user_stop", methods=['POST'])
def userStop():
    model.reload()
    return jsonify({"reload_success": True})

@app.route("/persuasion_bot", methods=['POST'])
def getResponse():
    exitbutton_appear = False
    sid = request.json.get('sid')
    input_text = request.json.get('input_text')
    RECEIVED_TIME = time.time()
    print(sid)

    # exit button condition
    if model.turn_i >= 9 or ("closing" in model.global_profile.sys_world.sent_profile.keys()):
        exitbutton_appear = True


    MODE = cfg.interactive_mode
    if input_text == "<start>":
        input_text = None
    result = model.chat(input_text=input_text, mode=MODE, sid=sid)
    if result is not None:
        response, [sents_success, sents_failed], have_enough_candidates, usr_input_text = result
        # TOTAL_SUCCESS_CANDIDATES += len(sents_success)

    delay_for_typing(RECEIVED_TIME, response)

    # [output_text, sys_da_output, sys_se_output, usr_da_output, usr_se_outpu] = model.chat(input_text, sid)
    return jsonify({"response": response, 
                    "exitbutton_appear": exitbutton_appear
                    })#,     #T/F only          
                    # "sents_success": sents_success, 
                    # "sents_failed": sents_failed, 
                    # "have_enough_candidates": have_enough_candidates, 
                    # "usr_input_text": usr_input_text})
    #return jsonify(ed_result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8089)
    #socketio.run(app, host='0.0.0.0', port = 8087, use_reloader=False)
