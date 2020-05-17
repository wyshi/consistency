from PersuasionInteract import PersuasiveBot, sent_tokenize_modified
import config as cfg
from PPO import PersuadeDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import numpy as np
import pdb
import logging

class Labeler(PersuasiveBot):
    def __init__(self, model_config, 
                 model_A=None, model_B=None, tokenizer=None, device1=None, device2=None, 
                 models_used_in_model_clf=None, sent_embedding_model=None):
        super().__init__(model_config=model_config,
                         model_A=model_A, model_B=model_B, tokenizer=tokenizer, device1=device1, device2=device2)
        train_data = torch.load("DataProcess/train_dialogs.pkl")
        val_data = torch.load("DataProcess/val_dialogs.pkl")
        self.train_dataset = PersuadeDataset(train_data, tokenizer)
        self.val_dataset = PersuadeDataset(val_data, tokenizer)

    def chat(self, sys_text=None, usr_text=None, mode=cfg.interactive_mode, sid=None):
        # sid = 0   
        if usr_text is None:
            assert sys_text is not None
            self.label_ground_truth_qa_sys(sys_text) 
        else:    
            assert sys_text is None
            self.label_ground_truth_qa_usr(usr_text) 

    def label_ground_truth_qa_sys(self, ground_truth):
        sent = ground_truth
        sents = sent_tokenize_modified(sent)
        # use regex to re-label
        sent_acts, _ = self.global_profile.regex_label(self.model_clf,
                                                    sents, 
                                                    which_task="A")

        if self.model_config.with_repetition_module:                                                                
            conflict_condition, conflict_amount, edited_sents, edited_sent_acts, fail_reason = self.global_profile.check_conflict(sents, sent_acts)  
        else:
            conflict_condition = True
            edited_sents = sents
            conflict_amount = 0
            edited_sent_acts = sent_acts
            fail_reason = "None, because no repetition module"
            # past = past

        self.cnt += 1

        self.global_profile.update(sents=sents, sent_labels=sent_acts, who=self.domain.SYS) #self.last_sys_labels = self.sys_profile.update(sys_texts=sents, sys_labels=sent_acts)

        # join sentences! finally!
        sent = " ".join(edited_sents)
        if sent != ground_truth:
            print("different from ground truth!")
            print(sent)
            print(ground_truth)
            print("----------------------------")
        # if sent == "ARDM MEMORY RESTARTS!":
        #     self.past = None
        #     return "ARDM MEMORY RESTARTS!"
        # self.past = past
        # set past for model_clf
        self.model_clf.set_past(sent=sent, 
                                which_task="A")

        return conflict_condition

    def label_ground_truth_qa_usr(self, input_text):
        input_texts = sent_tokenize_modified(input_text)
        # input_texts_labels = [None]*len(input_texts)
        input_texts_labels, _ = self.global_profile.regex_label(self.model_clf,
                                                                input_texts, 
                                                                which_task="B")
        self.model_clf.set_past(sent=input_text, 
                                which_task="B")
        
        self.global_profile.update(sents=input_texts, sent_labels=input_texts_labels, who=self.domain.USR) #self.usr_profile.update(input_text, self.last_sys_labels)
        self.last_sent = input_text
        self.turn_i += 1

    def label_one_dialog(self, dialog_i):
        batch = self.train_dataset[dialog_i]
        role_ids, _,  dial_sents = batch

        print(f"len: {len(role_ids)}")
        NUM_SUCCESS_SENTS = 0
        NUM_TURNS = 0
        for role_id, dial_sent in zip(role_ids, dial_sents):
            print(f"turn #: {self.turn_i}\n\n\n")
            logging.info(f"turn #: {self.turn_i}\n\n\n")
            if role_id == 0:
                print(f"sys: {dial_sent}")
                logging.info(f"sys: {dial_sent}")
                is_pass = self.label_ground_truth_qa_sys(dial_sent)
                # self.global_profile.print_to_log()
                # if not is_pass:
                #     pdb.set_trace()
                # pdb.set_trace()
                # input("continue...")
            else:
                print(f"usr: {dial_sent}")
                logging.info(f"usr: {dial_sent}")
                self.label_ground_truth_qa_usr(dial_sent)
                # self.global_profile.print_to_log()
                # pdb.set_trace()
                # input("continue...")

        if role_id == 1: # the last sent is user
            # throw away the last user sentence
            pass
   
if __name__ == "__main__":

    from PPO import load_model

    EVAL_MODEL_A_DIR = "/home/wyshi/persuasion/consistency/ARDM/persuasion/persuasion_medium_3.th"
    DEVICE1 = torch.device("cuda:2")
    DEVICE1_list = ["cuda:2"]
    SPLIT_INTO1= 1

    DEVICE2 = torch.device("cuda:3")
    DEVICE2_list = ["cuda:3"]
    SPLIT_INTO2= 1

    LOG_FILE = "logs/labeler.log"
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

    MODEL_A, MODEL_B, TOKENIZER, DEVICE1, DEVICE2 = load_model_for_AMT(EVAL_MODEL_A_DIR)

    MODE = cfg.interactive_mode

    model = Labeler(model_config=CurrentModelConfig, 
                        model_A=MODEL_A, model_B=MODEL_B, tokenizer=TOKENIZER, 
                        device1=DEVICE1, device2=DEVICE2)
    OFF_SET = 0
    DIALOG_i = 0 + OFF_SET
    MAX_DIALOG = 10 + OFF_SET
    while DIALOG_i <= MAX_DIALOG:
        logging.info("-------------------------- one dialog start ------------------------------------------")
        model.label_one_dialog(DIALOG_i)
        model.global_profile.print_to_log()
        logging.info("-------------------------- one dialog done ------------------------------------------")
        DIALOG_i += 1
        model.reload()

