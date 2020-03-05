import re
import sys
sys.path.append("../")
from classifier.pred import strategy_model
import config as cfg
from utils import is_repetition_with_context
import itertools
from AgentProfile.core import SystemAct
from KnowledgeBase import KB, Domain

from nltk.tokenize import sent_tokenize


class GlobalProfile(object):
    def __init__(self, domain=cfg.domain):
        self.domain = Domain(domain)
        
        self.profile = {'system': {att: [self.domain.INIT] for att in self.domain.attributes},
                        'user': {att: [self.domain.INIT] for att in self.domain.attributes}
                        }
        self.history = []
        self.history_label = []
    
    def extract_info(sent, who):
        last_sent = self.history[-1]
        if "how are you" in sent:


        el

    def refresh():
        self.profile = {'system': {att: [self.domain.INIT] for att in self.domain.attributes},
                        'user': {att: [self.domain.INIT] for att in self.domain.attributes}
                        }
        self.history = []
        self.history_label = []

class IndividualProfile(object):
    def __init__(self, domain=cfg.domain):        
        self.profile = {}

    def refresh(self): 
        self.profile = {}
    
    def __call__(self):
        return self.profile

    def print(self):
        print("*******profile*******")
        for k, v in self.profile.items():
            print("{}: {}".format(k, v))
        print("*********************")

    def keys(self):
        return list(self.profile.keys())

    def values(self):
        return list(self.profile.values())

    def items(self):
        return self.profile.items()

    def extract_info(sent, who, last_sys_label=None, last_usr_label=None, last):
        last_sent = self.history[-1]
        if "how are you" in 


to_update_usr, to_update_sys = extract_info(cur_sent, history, history_act):
self.usr_profile.update(to_update)

for k, v in to_update:
    if v == YES and self.sys_profile.profile[k] == NO
 self.profile


class UsrProfile(IndividualProfile):
    def __init__(self, global_profile):
        super().__init__()
        self.global_profile = global_profile

    def update(self, usr_text, last_sys_labels):
        # update using the user inputs
        for last_sys_label in last_sys_labels:
            if last_sys_label in self.profile:
                self.profile[last_sys_label].append(usr_text)
            else:
                self.profile[last_sys_label] = [usr_text]
    
    def check_conflict(self, sys_texts, sys_labels):
        # check system candidates' confict with the system profiles
        # e.g. user: i want to donate a dollar
        #      sys: how much will you donate?
        def check_conflict_for_one_utt(sys_text, sys_label):
            is_repetition, repetition_ratio = is_repetition_with_context(sys_text, 
                                                                        itertools.chain(*self.profile.values()), 
                                                                        threshold=cfg.repetition_threshold)

            if "inquiry" in sys_label:
                # e.g. sys: have you heard of save the children?
                #      usr: i have heard of it
                #      sys: do you know about save the children?
                if self.is_inquiry_answered(sys_text, sys_label):
                    # 1.1 real repetition, 
                    # this is repetition inquiry
                    if cfg.debug:
                        print("{} inquiry encountered in user_profile check! {}: {}".format(cfg.REPETITION, sys_label, sys_text))
                    return cfg.REPETITION, repetition_ratio
                else:
                    # 1.2 fake repetition,
                    # where the user never replies the inquiry
                    return cfg.PASS, repetition_ratio
            elif is_repetition:
                if cfg.debug:
                    print("exact repetition with user utterance encountered in user_profile check! {}: {}".format(sys_label, sys_text))
                
                return cfg.PASS, repetition_ratio
            else:
                return cfg.PASS, repetition_ratio

        conflict_statuses, conflict_amounts = [], []
        for sys_text, sys_label in zip(sys_texts, sys_labels):
            conflict_status, conflict_amount = check_conflict_for_one_utt(sys_text, sys_label)
            conflict_statuses.append(conflict_status)
            conflict_amounts.append(conflict_amount)

        if len(sys_texts) == 1:
            return conflict_statuses[0], conflict_amounts[0], sys_texts, sys_labels
        else:
            edited_sents = []
            edited_sent_acts = []
            for status, sys_text, sys_label in zip(conflict_statuses, sys_texts, sys_labels):
                if status not in [cfg.PASS]:
                    pass
                else:
                    edited_sents.append(sys_text)
                    edited_sent_acts.append(sys_label)
            if len(edited_sents) == 0:
                is_repetition, repetition_ratio = is_repetition_with_context(" ".join(sys_texts), 
                                                                        itertools.chain(*self.profile.values()), 
                                                                        threshold=cfg.repetition_threshold)

                return cfg.NOT_PASS, repetition_ratio, sys_texts, sys_labels
            else:
                is_repetition, repetition_ratio = is_repetition_with_context(" ".join(edited_sents), 
                                                                        itertools.chain(*self.profile.values()), 
                                                                        threshold=cfg.repetition_threshold)

                return cfg.PASS, repetition_ratio, edited_sents, edited_sent_acts
  

    def is_inquiry_answered(self, sys_text, sys_label):
        # temporary!!!
        if sys_label not in self.profile:
            return False
        usr_responses_to_inqury = self.profile[sys_label] #todo, temporarily, should use all user context to find if the answer is already there
                                                          # should be self.is_qa_pair(sys_text, sys_label, self.profile.values())
        return self.is_qa_pair(sys_text, sys_label, usr_responses_to_inqury)

    def is_qa_pair(self, sys_text, sys_label, usr_response_list):
        # if we cannot extract the answer to the sys_text from the usr_response_list, then it's not answered.
        #===QA=== improvements comes in 
        if sys_label in [SystemAct.kids_related_inquiry, 
                         SystemAct.donation_related_inquiry, 
                         SystemAct.organization_related_inquiry, 
                         SystemAct.propose_donation_inquiry]:
            for usr_response in usr_response_list:
                if ("yes" in usr_response) or ("no" in usr_response) or ("not" in usr_response):
                    return True
            return False
        elif sys_label in [SystemAct.other_inquiry]:
            return True
        else:
            return True
    
    def refresh(self):
        super().refresh()
        self.global_profile.refresh()

class SysProfile(IndividualProfile):
    def __init__(self, global_profile):
        super().__init__()
        self.pred_model = strategy_model(model_to_load="./classifier/best_model_state_er.pkl")
        self.global_profile = global_profile

    def update(self, sys_texts, sys_labels):
        # update using the system inputs
        # label = self.regex_label(sys_text, context, turn_i)

        for sys_text, sys_label in zip(sys_texts, sys_labels):
            if sys_label in self.profile:
                self.profile[sys_label].append(sys_text)
            else:
                self.profile[sys_label] = [sys_text]

        return sys_labels

    def check_conflict(self, sys_texts, sys_labels):
        # check system candidates' confict with the user profiles
        # label = self.regex_label(sys_text, context, turn_i)
        def check_conflict_for_one_utt(sys_text, sys_label):
            is_repetition, repetition_ratio = is_repetition_with_context(sys_text, 
                                                                        itertools.chain(*self.profile.values()), 
                                                                        threshold=cfg.repetition_threshold)

            
            if sys_label in self.profile:
                if "inquiry" in sys_label:
                    # 1. inquiry
                    if False:
                        # this is repetition inquiry
                        print("{} encountered! {}: {}".format(cfg.REPETITION, sys_label, sys_text))
                        return cfg.REPETITION, repetition_ratio
                    else:
                        return cfg.PASS, repetition_ratio
                else:
                    # 2. statement                    
                    if is_repetition:
                        # 2.1 fake repetition
                        if self.is_qa_pair():
                            # elif (usr_label, sys_label) in cfg.QA_PAIR_WITH_UNIQUE_ANSWER_DB: 
                            #     # case 2: potentially be fake repetitions 
                            #     # (when user asks, system repeats similar answers)
                            #     # therefore, temporarily solve by 
                            #     # 1) external database support, or 
                            #     # 2) if sample 20 candidates, all the same, then that probably means there is only one best answer to the question
                            return cfg.PASS, repetition_ratio
                        # 2.2 real repetition
                        else:
                            print("{} encountered! {}: {}".format(cfg.REPETITION, sys_label, sys_text))
                            return cfg.REPETITION, repetition_ratio

                    else:
                        return cfg.PASS, repetition_ratio

            else:
                if is_repetition:
                    return cfg.REPETITION, repetition_ratio
                else:
                    return cfg.PASS, repetition_ratio

        conflict_statuses, conflict_amounts = [], []
        for sys_text, sys_label in zip(sys_texts, sys_labels):
            conflict_status, conflict_amount = check_conflict_for_one_utt(sys_text, sys_label)
            conflict_statuses.append(conflict_status)
            conflict_amounts.append(conflict_amount)
        
        if len(sys_texts) == 1:
            return conflict_statuses[0], conflict_amounts[0], sys_texts, sys_labels
        else:
            edited_sents = []
            edited_sent_acts = []
            for status, sys_text, sys_label in zip(conflict_statuses, sys_texts, sys_labels):
                if status not in [cfg.PASS]:
                    pass
                else:
                    edited_sents.append(sys_text)
                    edited_sent_acts.append(sys_label)
            if len(edited_sents) == 0:
                is_repetition, repetition_ratio = is_repetition_with_context(" ".join(sys_texts), 
                                                                        itertools.chain(*self.profile.values()), 
                                                                        threshold=cfg.repetition_threshold)

                return cfg.NOT_PASS, repetition_ratio, sys_texts, sys_labels
            else:
                is_repetition, repetition_ratio = is_repetition_with_context(" ".join(edited_sents), 
                                                                        itertools.chain(*self.profile.values()), 
                                                                        threshold=cfg.repetition_threshold)

                return cfg.PASS, repetition_ratio, edited_sents, edited_sent_acts
            

    def is_qa_pair(self, sys_text=None, sys_label=None, usr_response_list=None):
        # if we cannot extract the answer to the sys_text from the usr_response_list, then it's not answered.
        #===QA=== improvements comes in 
        #todo
        return False
        
        if sys_label in [SystemAct.kids_related_inquiry, 
                         SystemAct.donation_related_inquiry, 
                         SystemAct.organization_related_inquiry, 
                         SystemAct.propose_donation_inquiry]:
            for usr_response in usr_response_list:
                if ("yes" in usr_response) or ("no" in usr_response) or ("not" in usr_response):
                    return True
            return False
        elif sys_label in [SystemAct.other_inquiry]:
            return True
        else:
            return True


    def regex_label(self, sys_texts, context, turn_i):
        """
        regex to re-label 
        vs 
        QA    to re-label
        """
        def regex_label_for_one_utt(utt):
            predicted_label = self.pred_model.predict(text=utt, his=context, turn=turn_i)

            if predicted_label in ["task-related-inquiry", "personal-related-inquiry"]:#, "have-you-heard-of-the-org"]:
                sent = utt.lower()
                if "have" in sent and (("kid" in sent) or ("children" in sent)):
                    label = SystemAct.kids_related_inquiry

                elif "donate" in sent:
                    label = SystemAct.donation_related_inquiry

                # elif "save the children" in sent:
                #     #todo: "Would you be interested in donating to Save the Children today?"
                #     label = "organization-related-inquiry"

                else:
                    label = SystemAct.other_inquiry

            elif predicted_label in ["have-you-heard-of-the-org"]:
                label = SystemAct.organization_related_inquiry

            elif predicted_label in ["propose-donation"]:
                label = SystemAct.propose_donation_inquiry 

            elif "ask" in predicted_label: 
                label = predicted_label + "-inquiry"

            ##====== above are all inquiries =======

            elif predicted_label in ['provide-org-facts', 'provide-donation-procedure']:
                label = predicted_label

            else:
                label = predicted_label

            return label

        labels = [regex_label_for_one_utt(utt) for utt in sys_texts]

        return labels

    def refresh(self):
        super().refresh()
        self.global_profile.refresh()

