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
import pdb
import time
import datetime

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from GPTModel1 import GPT2LMHeadModel_modified
from PPO import load_model as load_model_in_PPO
from model_clf import load_model as load_model_in_MODEL_CLF, SequenceSummary, load_pkl
from PersuasionInteract import PersuasiveBot, ModelClassifierConfig
import config as cfg
import sys
from torchfly.modules.losses import SequenceFocalLoss, SequenceCrossEntropyLoss
import logging
from sentence_transformers import SentenceTransformer

LOG_FILE = 'logs/amt_new_model-with_RL-real-real.log'
logging.basicConfig(filename=LOG_FILE,level=logging.DEBUG)
TIME = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
logging.info(f"!!!!!--------- AMT test: datetime {TIME}----------")
app = Flask(__name__)

# EVAL_MODEL_A_DIR = "/home/wyshi/persuasion/consistency/ARDM/persuasion/persuasion_medium_3.th"
# EVAL_MODEL_A_DIR = "/home/wyshi/persuasion/consistency/Checkpoint/23_steps_1.79_2.536363636363636_reward_model_A_kl_7.53_ppo5.pth"
# EVAL_MODEL_A_DIR = "/home/wyshi/persuasion/consistency/Checkpoint/23_steps_1.79_2.536363636363636_reward_model_A_kl_7.53_ppo5.pth"

class CurrentModelConfig:
    with_rule = True
    log_file = LOG_FILE
    
    with_baseline =  True
    with_repetition_module = True
    with_consistency_module = True
    with_sentence_clf = True
    with_RL_finetune_model = True

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
    
def load_model_for_AMT(EVAL_MODEL_A_DIR, DEVICE1, DEVICE1_list, SPLIT_INTO1, DEVICE2, DEVICE2_list, SPLIT_INTO2):
    TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")#torch.load(tokenizer_dir)

    # val_dataloader = get_val_dataloader(TOKENIZER)
    model_A, model_B = load_model_in_PPO(cfg=cfg, device1=DEVICE1, device2=DEVICE2, 
                                split_into1=SPLIT_INTO1, split_into2=SPLIT_INTO2,
                                dropout=0, device_list1=DEVICE1_list, device_list2=DEVICE2_list,
                                model_A_dir=EVAL_MODEL_A_DIR, use_old_model_B=False)

    model_A.eval()
    model_B.eval()

    return model_A, model_B, TOKENIZER, DEVICE1, DEVICE2

def load_model_clf_for_AMT(model_clf_dir, device1, device2):
    config = GPT2Config()
    config = config.from_pretrained('gpt2')#config.from_pretrained('gpt2-medium')
    config.summary_first_dropout = 0.2
    config.summary_type = "cls_index"
    le_A = load_pkl("training/data/labelencoder_A.pkl")
    le_B = load_pkl("training/data/labelencoder_B.pkl")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")#torch.load(tokenizer_dir)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    
    model_A, model_B = load_model_in_MODEL_CLF(cfg, "small", tokenizer, device1, device2)
    clf_A = SequenceSummary(num_labels=le_A.classes_.shape[0], config=config)
    clf_B = SequenceSummary(num_labels=le_B.classes_.shape[0], config=config)
    clf_TF = SequenceSummary(num_labels=2, config=config)
    
    model_A_state, model_B_state, clf_A_state, clf_B_state, clf_TF_state = torch.load(model_clf_dir)
    model_A.load_state_dict(model_A_state)
    model_B.load_state_dict(model_B_state)
    clf_A.load_state_dict(clf_A_state)
    clf_B.load_state_dict(clf_B_state)
    clf_TF.load_state_dict(clf_TF_state)
    print(f"all models loaded")

    return tokenizer, model_A, model_B, clf_A, clf_B, clf_TF

DEVICE1 = torch.device("cuda:5")
DEVICE1_list = ["cuda:5"]
SPLIT_INTO1= 1

DEVICE2 = torch.device("cuda:6")
DEVICE2_list = ["cuda:6"]
SPLIT_INTO2= 1

MODEL_A, MODEL_B, TOKENIZER, DEVICE1, DEVICE2 = load_model_for_AMT(EVAL_MODEL_A_DIR, 
                                                                   DEVICE1, DEVICE1_list, SPLIT_INTO1, 
                                                                   DEVICE2, DEVICE2_list, SPLIT_INTO2)
MODELS_USED_IN_MODEL_CLF = load_model_clf_for_AMT(ModelClassifierConfig.model_dir, 
                                                  ModelClassifierConfig.device1, 
                                                  ModelClassifierConfig.device2)
SENT_EMBEDDING_MODEL = SentenceTransformer('bert-base-nli-mean-tokens', 
                                            device=torch.device(cfg.sent_embedding_model_device))
def build_one_model(MAX_USER):
    # keeps a copy of model
    a = time.time()

    model = PersuasiveBot(model_config=CurrentModelConfig, 
                        model_A=MODEL_A, model_B=MODEL_B, tokenizer=TOKENIZER, 
                        device1=DEVICE1, device2=DEVICE2, models_used_in_model_clf=MODELS_USED_IN_MODEL_CLF, sent_embedding_model=SENT_EMBEDDING_MODEL)

    # model = HuggingfaceModel("./runs/1000pretrained")
    model.reload()
    logging.info(f"model_to_Eval: {EVAL_MODEL_A_DIR}")
    logging.info(f"with_baseline: {CurrentModelConfig.with_baseline}")
    logging.info(f"with_repetition_module: {CurrentModelConfig.with_repetition_module}")
    logging.info(f"with_consistency_module: {CurrentModelConfig.with_consistency_module}")
    logging.info(f"with_sentence_clf: {CurrentModelConfig.with_sentence_clf}")
    logging.info(f"with_RL_finetune_model: {CurrentModelConfig.with_RL_finetune_model}")
    logging.info(f"candidate_select_strategy: {CurrentModelConfig.candidate_select_strategy}")
    logging.info(f"NUM_CANDIDATES: {CurrentModelConfig.NUM_CANDIDATES}")
    logging.info(f"with_rule: {CurrentModelConfig.with_rule}")
    b = time.time()

    logging.info(f"building one take {b-a} time")
    print(f"building one take {b-a} time")
    return model

def end_condition(usr_input):

    return False

MODEL_MAP = {}
MAX_USER = -1
AVAILABEL_MODELS = [build_one_model(MAX_USER) for _ in range(2)]
# pdb.set_trace()

def delay_for_typing(RECEIVED_TIME, response):
    response_char_len = len(response)
    response_word_len = len(response.split())
    AVG_TIME_TO_TYPE_CHAR = 225/60
    AVE_TIME_TO_TYPE_WORD = 45/60
    time_to_type_char = response_char_len/AVG_TIME_TO_TYPE_CHAR
    time_to_type_word = response_word_len/AVE_TIME_TO_TYPE_WORD

    RESPONDED_TIME = time.time()

    time_already_passed = RESPONDED_TIME - RECEIVED_TIME
    
    time_to_sleep_char = time_to_type_char - time_already_passed
    time_to_sleep_word = time_to_type_word - time_already_passed

    time_to_sleep = min(time_to_sleep_char, time_to_sleep_word)
    # time_to_sleep -= 0.5
    
    if time_to_sleep > 0:
        time_to_sleep = min(time_to_sleep, 30)
        print(f"time should sleep: {min(time_to_type_char, time_to_type_word)}")
        print(f"time already passed: {time_already_passed}")
        print(f"sleep for {time_to_sleep}")
        time.sleep(time_to_sleep)

@app.route("/user_stop", methods=['POST'])
def userStop():
    sid = request.json.get('sid')
    print(f"sid to stop: {sid}")
    logging.info(f"sid to stop: {sid}")
    TIME = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    # garbage clean
    if sid in MODEL_MAP:
        logging.info(f"AVAILABEL_MODELS in userStop before: {len(AVAILABEL_MODELS)}")
        logging.info(f"MODEL_MAP in userStop before: {MODEL_MAP.keys()}")
        print(f"AVAILABEL_MODELS in userStop before: {len(AVAILABEL_MODELS)}")
        print(f"MODEL_MAP in userStop before: {MODEL_MAP.keys()}")

        logging.info(f"!!!!!---------{sid} AMT end test: datetime {TIME}----------")
        logging.info(f"{sid} history: {MODEL_MAP[sid].global_profile.history}")
        print(f"{sid} history: {MODEL_MAP[sid].global_profile.history}")
        MODEL_MAP[sid].reload()
        model_for_reuse = MODEL_MAP.pop(sid)
        AVAILABEL_MODELS.append(model_for_reuse)
        logging.info(f"AVAILABEL_MODELS in userStop: {len(AVAILABEL_MODELS)}")
        logging.info(f"MODEL_MAP in userStop: {MODEL_MAP.keys()}")
        print(f"AVAILABEL_MODELS in userStop: {len(AVAILABEL_MODELS)}")
        print(f"MODEL_MAP in userStop: {MODEL_MAP.keys()}")
        assert sid not in MODEL_MAP

    return jsonify({"reload_success": True})

@app.route("/persuasion_bot", methods=['POST'])
def getResponse():
    global MAX_USER
    exitbutton_appear = False
    sid = request.json.get('sid')
    input_text = request.json.get('input_text')
    RECEIVED_TIME = time.time()

    if sid not in MODEL_MAP:
        # when a new user comes in
        logging.info(f"AVAILABEL_MODELS in getResponse before: {len(AVAILABEL_MODELS)}")
        logging.info(f"MODEL_MAP in getResponse before: {MODEL_MAP.keys()}")
        print(f"AVAILABEL_MODELS in getResponse before: {len(AVAILABEL_MODELS)}")
        print(f"MODEL_MAP in getResponse before: {MODEL_MAP.keys()}")

        if len(AVAILABEL_MODELS) == 0:
            print(f"------------------ building model for {sid} --------------------")
            logging.info(f"--------------------- building model for {sid} -------------------")
            new_model = build_one_model(MAX_USER)
            MODEL_MAP[sid] = new_model
            print(f"------------------ building model for {sid} finished--------------------")
            logging.info(f"--------------------- building model for {sid} finished-------------------")
        else:
            print(f"------------------ reusing model for {sid} --------------------")
            logging.info(f"--------------------- reusing model for {sid} -------------------")
            MODEL_MAP[sid] = AVAILABEL_MODELS.pop(0)
            print(f"------------------ reusing model for {sid} finished--------------------")
            logging.info(f"--------------------- reusing model for {sid} finished-------------------")
        # pdb.set_trace()
        logging.info(f"AVAILABEL_MODELS in getResponse: {len(AVAILABEL_MODELS)}")
        logging.info(f"MODEL_MAP in getResponse: {MODEL_MAP.keys()}")
        print(f"AVAILABEL_MODELS in getResponse: {len(AVAILABEL_MODELS)}")
        print(f"MODEL_MAP in getResponse: {MODEL_MAP.keys()}")
        assert MODEL_MAP[sid].past is None
        MODEL_MAP[sid].reload()

    # how many users are there concurrently
    MAX_USER = max(MAX_USER, len(MODEL_MAP))
    print(f" at the moment, there are {len(MODEL_MAP)} users, MAX_USER is {MAX_USER} ")
    logging.info(f" at the moment, there are {len(MODEL_MAP)} users, MAX_USER is {MAX_USER} ")
    
    # exit button condition
    if MODEL_MAP[sid].turn_i >= 9 or ("closing" in MODEL_MAP[sid].global_profile.sys_world.sent_profile.keys()):
        exitbutton_appear = True


    MODE = cfg.interactive_mode
    if input_text == "<start>":
        # a new dialog
        MODEL_MAP[sid].reload()
        input_text = None
        TIME = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"!!!!!---------{sid} AMT start test: datetime {TIME}----------")

    result = MODEL_MAP[sid].chat(input_text=input_text, mode=MODE, sid=sid)
    if result is not None:
        response, [sents_success, sents_failed], have_enough_candidates, usr_input_text, _ = result
        # TOTAL_SUCCESS_CANDIDATES += len(sents_success)

    delay_for_typing(RECEIVED_TIME, response)

    # exit button condition
    if ("closing" in MODEL_MAP[sid].global_profile.sys_world.sent_profile.keys()):
        exitbutton_appear = True

    if len(MODEL_MAP[sid].global_profile.history_label) >= 4 and not exitbutton_appear:
        # pdb.set_trace()
        usr_labels = MODEL_MAP[sid].global_profile.history_label[:-3]
        print(MODEL_MAP[sid].global_profile.history_label)
        logging.info(MODEL_MAP[sid].global_profile.history_label)
        any_is_agree = any([("provide-donation-amount" in usr_label) for usr_label in usr_labels])
        if any_is_agree:
            exitbutton_appear = True

    # try:
    #     pdb.set_trace()
    # except:
    logging.info(f"ids: {[(k, id(m.past), id(m.model_clf.past)) for k, m in MODEL_MAP.items()]}")
    print(f"ids: {[(k, id(m.past), id(m.model_clf.past)) for k, m in MODEL_MAP.items()]}")
    # print(id(AVAILABEL_MODELS[0].model_clf.past), AVAILABEL_MODELS[0].model_clf.history, id(AVAILABEL_MODELS[0].past), AVAILABEL_MODELS[0].past[0].shape)   
    # print(id(AVAILABEL_MODELS[1].model_clf.past), AVAILABEL_MODELS[1].model_clf.history, id(AVAILABEL_MODELS[1].past), AVAILABEL_MODELS[1].past[0].shape)   

    return jsonify({"response": response, 
                    "exitbutton_appear": exitbutton_appear
                    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8089)
    #socketio.run(app, host='0.0.0.0', port = 8087, use_reloader=False)
