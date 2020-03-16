import re
import sys
sys.path.append("../")
from classifier.pred import strategy_model
import config as cfg
from utils import is_repetition_with_context
import itertools
from AgentProfile.core import SystemAct
from KnowledgeBase import KB
from KnowledgeBase.KB import Domain

from nltk.tokenize import sent_tokenize
from copy import deepcopy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

from utils import toNumReg
# HOW_ARE_YOU = 'how-are-you'
# HEARD_OF_THE_ORG = 'heard-of-the-org'
# HAVE_KIDS = 'have-kids'
# DONATED_BEFORE = 'donated-before'
# WANT_TO_DONATE = 'want-to-donate'
# DONATION_AMOUNT = 'donation-amount'


class GlobalProfile(object):
    def __init__(self, domain):
        self.domain = domain
        self.pred_model = strategy_model(model_to_load="./classifier/best_model_state_er.pkl")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        self.sys_world = SysWorld(domain=self.domain, name="system_world")
        self.usr_world = UsrWorld(domain=self.domain, name="user_world")
        self.history = []
        self.history_label = []
        self.last_sents = None
        self.last_labels = None
        self.answers_extracted_from_last_sent = None

        self.greeting_re = re.compile(r"((how are you)|(how're you)|(how you doing)|(how about you))", re.IGNORECASE)
        
    # def __getitem__(self, key):
    #     return self.profiles[key.lower()]

    def extract_info(self, sents, who):

        qa_dict_usr, qa_dict_sys = self.contain_answers(sents, who)
        
        return qa_dict_usr, qa_dict_sys

    def contain_answers(self, sents, who):
        qa_dict = {
                    self.domain.USR:{},
                    self.domain.SYS:{}
                  }
        for q in self.domain.attributes:
            answers = self.answer_question(sents, q, who)
            
            if answers['usr']:
                qa_dict[self.domain.USR].update({q: answers['usr']})
            if answers['sys']:
                qa_dict[self.domain.SYS].update({q: answers['sys']})
                
        for side in qa_dict:
            if self.domain.DONATION_AMOUNT in qa_dict[side]:
                if type(qa_dict[side][self.domain.DONATION_AMOUNT]) is int:
                    if qa_dict[side][self.domain.DONATION_AMOUNT] > 0:
                        qa_dict[side][self.domain.WANT_TO_DONATE] = self.domain.YES
                    elif qa_dict[side][self.domain.DONATION_AMOUNT] <= 0:
                        qa_dict[side][self.domain.WANT_TO_DONATE] = self.domain.NO

        sent = " ".join(sents).lower()
        if re.search(r"((me,)|(me)) too", sent, re.IGNORECASE) \
            or "same here" in sent:
            if who == self.domain.USR:
                qa_dict[self.domain.USR].update(self.answers_extracted_from_last_sent[self.domain.SYS])
            else:
                qa_dict[self.domain.SYS].update(self.answers_extracted_from_last_sent[self.domain.USR])
        return qa_dict[self.domain.USR], qa_dict[self.domain.SYS]

    def answer_question(self, sents, q, who):
        """answer the question q based on the current input sents from who.
        return
        answers = {}
        
        """
        last_sent = " ".join(self.last_sents) if self.last_sents else None
        sent = " ".join(sents)
        answers = {'usr': None,
                   'sys': None}

        sent = sent.lower()

        if q == self.domain.HOW_ARE_YOU:
            answers = self.answer_HOW_ARE_YOU(sent, who, last_sent)

        elif q == self.domain.HEARD_OF_THE_ORG:    
            answers = self.answer_HEARD_OF_THE_ORG(sent, who, last_sent)

        elif q == self.domain.HAVE_KIDS:
            answers = self.answer_HAVE_KIDS(sent, who, last_sent)

        elif q == self.domain.DONATED_BEFORE:
            answers = self.answer_DONATED_BEFORE(sent, who, last_sent)

        elif q == self.domain.WANT_TO_DONATE:
            answers = self.answer_WANT_TO_DONATE(sent, who, last_sent)

        elif q == self.domain.DONATION_AMOUNT:
            answers = self.answer_DONATION_AMOUNT(sent, who, last_sent)

        else:
            raise ValueError("question {} not supported".format(q))
        

        if "as well" in sent:
            if who == self.domain.USR:
                answers['sys'] = answers['usr']
            else:
                answers['usr'] = answers['sys']

            
        return answers

    def answer_HOW_ARE_YOU(self, sent, who, last_sent):
        # self.domain.ATT_TO_QUESTION[self.domain.HOW_ARE_YOU]
        answers = {'usr': None,
                   'sys': None}
        if last_sent and re.search(self.greeting_re, last_sent): 
            # asked
            if re.search(r"((you are)|(you're)) doing ((great)|(well)|(good))", sent):
                if who == self.domain.USR:
                    answers['sys'] = "good"
                elif who == self.domain.SYS:
                    answers['usr'] = "good"
            elif re.search(r"doing ((good)|(well)|(alright)|(ok))", sent):
                if who == self.domain.USR:
                    answers['usr'] = "good"
                elif who == self.domain.SYS:
                    answers['sys'] = "good"
        else:
            # not asked
            if re.search(r"((you are)|(you're)) doing ((great)|(well)|(good))", sent):
                if who == self.domain.USR:
                    answers['sys'] = "good"
                elif who == self.domain.SYS:
                    answers['usr'] = "good"
            elif re.search(r"((doing )| )((good)|(well)|(alright)|(ok))", sent):
                if who == self.domain.USR:
                    answers['usr'] = "good"
                elif who == self.domain.SYS:
                    answers['sys'] = "good"
        return answers

    def answer_HEARD_OF_THE_ORG(self, sent, who, last_sent):
        answers = {'usr': None,
                   'sys': None}
        if last_sent and SystemAct.organization_related_inquiry in self.history_label[-1]:
            # 1) asked
            score = self.sentiment_analyzer.polarity_scores(sent)
            if who == self.domain.USR:
                # 1.1) asked-user speak
                # 1.1.1) asked-user speak-about self
                if "yes" in sent:
                    answers['usr'] = self.domain.YES
                elif "no" in sent: 
                    answers['usr'] = self.domain.NO
                else:
                    if score['compound'] >= 0.05:
                        # positive
                        answers['usr'] = self.domain.YES
                    elif score['compound'] <= -0.05:
                        answers['usr'] = self.domain.NO
                    else:
                        answers['usr'] = self.domain.NOT_SURE

                # 1.1.2) asked-user speak-about system
                if re.search(r"((you've)|(you have)) heard", sent, re.IGNORECASE):
                    answers['sys'] = self.domain.YES

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if "yes" in sent:
                    answers['sys'] = self.domain.YES
                elif "no" in sent: 
                    answers['sys'] = self.domain.NO
                else:
                    if score['compound'] >= 0.05:
                        # positive
                        answers['sys'] = self.domain.YES
                    elif score['compound'] <= -0.05:
                        answers['sys'] = self.domain.NO
                    else:
                        answers['sys'] = self.domain.NOT_SURE
            
                # 1.2.2) asked-system speak-about user
                if re.search(r"((you've)|(you have)) heard", sent, re.IGNORECASE)\
                     or "you know about" in sent:
                    answers['usr'] = self.domain.YES

        else:
            # not asked
            if who == self.domain.USR:
                # 2.1) asked-user speak
                # 2.1.1) asked-user speak-about self
                if re.search(r"((i've)|(i have)) heard of ((save the children)|(them))", sent, re.IGNORECASE):
                    answers['usr'] = self.domain.YES

                # 1.1.2) asked-user speak-about system
                if re.search(r"((you've)|(you have)) heard of ((save the children)|(them))", sent, re.IGNORECASE):
                    answers['sys'] = self.domain.YES

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if re.search(r"((i've)|(i have)) heard of ((save the children)|(them))", sent, re.IGNORECASE):
                    answers['sys'] = self.domain.YES
            
                # 1.2.2) asked-system speak-about user
                if re.search(r"((you've)|(you have)) heard of ((save the children)|(them))", sent, re.IGNORECASE):
                    answers['usr'] = self.domain.YES
                
                if "i've heard of them as well" in sent:
                    answers['usr'] = self.domain.YES

        return answers

    def answer_HAVE_KIDS(self, sent, who, last_sent):
        answers = {'usr': None,
                   'sys': None}
        num_kids_re = "(([-+]?[0-9]+)|(one)|(two)|(three)|(four)|(five)|(six)|(seven)|(eight)|(nine)|(zero))"
        if last_sent and SystemAct.kids_related_inquiry in self.history_label[-1]:
            # 1) asked
            score = self.sentiment_analyzer.polarity_scores(sent)
            if who == self.domain.USR:
                # 1.1) asked-user speak
                # 1.1.1) asked-user speak-about self
                if "yes" in sent:
                    answers['usr'] = self.domain.YES
                elif "no" in sent: 
                    answers['usr'] = self.domain.NO
                else:
                    if score['compound'] >= 0.05:
                        # positive
                        answers['usr'] = self.domain.YES
                    elif score['compound'] <= -0.05:
                        answers['usr'] = self.domain.NO
                    else:
                        answers['usr'] = self.domain.NOT_SURE

                # 1.1.2) asked-user speak-about system
                if re.search(r"((you've)|(you have))", sent, re.IGNORECASE) and "kids" in sent:
                    answers['sys'] = self.domain.YES

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if "yes" in sent:
                    answers['sys'] = self.domain.YES
                elif "no" in sent: 
                    answers['sys'] = self.domain.NO
                else:
                    if score['compound'] >= 0.05:
                        # positive
                        answers['sys'] = self.domain.YES
                    elif score['compound'] <= -0.05:
                        answers['sys'] = self.domain.NO
                    else:
                        answers['sys'] = self.domain.NOT_SURE
            
                # 1.2.2) asked-system speak-about user
                if re.search(r"((you've)|(you have))", sent, re.IGNORECASE) and "kids" in sent:
                    answers['usr'] = self.domain.YES

        else:
            # not asked
            if who == self.domain.USR:
                # 2.1) asked-user speak
                # 2.1.1) asked-user speak-about self
                if re.search(r"((i've)|(i have)) (([-+]?[0-9]+)|(one)|(two)|(three)|(four)|(five)|(six)|(seven)|(eight)|(nine)|(zero)) ((child)|(kid))", sent, re.IGNORECASE):
                    answers['usr'] = self.domain.YES

                # 1.1.2) asked-user speak-about system
                if re.search(r"((you've)|(you have)) (([-+]?[0-9]+)|(one)|(two)|(three)|(four)|(five)|(six)|(seven)|(eight)|(nine)|(zero)) ((child)|(kid))", sent, re.IGNORECASE):
                    answers['sys'] = self.domain.YES

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if re.search(r"((i've)|(i have)) (([-+]?[0-9]+)|(one)|(two)|(three)|(four)|(five)|(six)|(seven)|(eight)|(nine)|(zero)) ((child)|(kid))", sent, re.IGNORECASE):
                    answers['sys'] = self.domain.YES
            
                # 1.2.2) asked-system speak-about user
                if re.search(r"((you've)|(you have)) (([-+]?[0-9]+)|(one)|(two)|(three)|(four)|(five)|(six)|(seven)|(eight)|(nine)|(zero)) ((child)|(kid))", sent, re.IGNORECASE):
                    answers['usr'] = self.domain.YES
                
        return answers

    def answer_DONATED_BEFORE(self, sent, who, last_sent):
        answers = {'usr': None,
                   'sys': None}
        if last_sent and SystemAct.donation_related_inquiry in self.history_label[-1]:
            # 1) asked
            score = self.sentiment_analyzer.polarity_scores(sent)
            if who == self.domain.USR:
                # 1.1) asked-user speak
                # 1.1.1) asked-user speak-about self
                if "yes" in sent:
                    answers['usr'] = self.domain.YES
                elif "no" in sent: 
                    answers['usr'] = self.domain.NO
                else:
                    if score['compound'] >= 0.05:
                        # positive
                        answers['usr'] = self.domain.YES
                    elif score['compound'] <= -0.05:
                        answers['usr'] = self.domain.NO
                    else:
                        answers['usr'] = self.domain.NOT_SURE

                # 1.1.2) asked-user speak-about system
                if re.search(r"((you've)|(you have)) donated", sent, re.IGNORECASE):
                    answers['sys'] = self.domain.YES

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if "yes" in sent:
                    answers['sys'] = self.domain.YES
                elif "no" in sent: 
                    answers['sys'] = self.domain.NO
                else:
                    if score['compound'] >= 0.05:
                        # positive
                        answers['sys'] = self.domain.YES
                    elif score['compound'] <= -0.05:
                        answers['sys'] = self.domain.NO
                    else:
                        answers['sys'] = self.domain.NOT_SURE
            
                # 1.2.2) asked-system speak-about user
                if re.search(r"((you've)|(you have)) donated", sent, re.IGNORECASE):
                    answers['usr'] = self.domain.YES

        else:
            # not asked
            if who == self.domain.USR:
                # 2.1) asked-user speak
                # 2.1.1) asked-user speak-about self
                if re.search(r"((i've)|(i have)) donated", sent, re.IGNORECASE):
                    answers['usr'] = self.domain.YES

                # 1.1.2) asked-user speak-about system
                if re.search(r"((you've)|(you have)) donated", sent, re.IGNORECASE):
                    answers['sys'] = self.domain.YES

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if re.search(r"((i've)|(i have)) donated", sent, re.IGNORECASE):
                    answers['sys'] = self.domain.YES
            
                # 1.2.2) asked-system speak-about user
                if re.search(r"((you've)|(you have)) donated", sent, re.IGNORECASE):
                    answers['usr'] = self.domain.YES
                
                # if "i've heard of them as well" in sent:
                #     answers['usr'] = self.domain.YES

        return answers

    def answer_WANT_TO_DONATE(self, sent, who, last_sent):
        answers = {'usr': None,
                   'sys': None}
        if last_sent and SystemAct.propose_donation_inquiry in self.history_label[-1]:
            # 1) asked
            score = self.sentiment_analyzer.polarity_scores(sent)
            if who == self.domain.USR:
                # 1.1) asked-user speak
                # 1.1.1) asked-user speak-about self
                if "yes" in sent or "i would like to donate some money" in sent or "i can donate a bit" in sent\
                    or "how much do you suggest" in sent:
                    answers['usr'] = self.domain.YES
                elif "prefer to donate time" in sent or "next time" in sent: 
                    answers['usr'] = self.domain.NO
                else:
                    if score['compound'] >= 0.05:
                        # positive
                        answers['usr'] = self.domain.YES
                    elif score['compound'] <= -0.05:
                        answers['usr'] = self.domain.NO
                    else:
                        answers['usr'] = self.domain.NOT_SURE

                # 1.1.2) asked-user speak-about system
                if "that's kind of you" in sent:
                    answers['sys'] = self.domain.YES

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if "yes" in sent or "i would like to donate some money" in sent or "i can donate a bit" in sent\
                    or "how much do you suggest" in sent:
                    answers['sys'] = self.domain.YES
                elif "prefer to donate time" in sent or "next time" in sent: 
                    answers['sys'] = self.domain.NO
                else:
                    if score['compound'] >= 0.05:
                        # positive
                        answers['sys'] = self.domain.YES
                    elif score['compound'] <= -0.05:
                        answers['sys'] = self.domain.NO
                    else:
                        answers['sys'] = self.domain.NOT_SURE
            
                # 1.2.2) asked-system speak-about user
                if "thank you so much" in sent or "you are very kind" in sent:
                    answers['usr'] = self.domain.YES

        else:
            # not asked
            if who == self.domain.USR:
                if "i would like to donate some money" in sent or "i can donate a bit" in sent\
                    or "how much do you suggest" in sent:
                    answers['usr'] = self.domain.YES
                elif "prefer to donate time" in sent or "next time" in sent: 
                    answers['usr'] = self.domain.NO

                # 2.1.2) asked-user speak-about system
                if "that's kind of you" in sent:
                    answers['sys'] = self.domain.YES

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if "i would like to donate some money" in sent or "i can donate a bit" in sent\
                    or "how much do you suggest" in sent:
                    answers['sys'] = self.domain.YES
                elif "prefer to donate time" in sent or "next time" in sent: 
                    answers['sys'] = self.domain.NO
            
                # 1.2.2) asked-system speak-about user
                if "thank you so much" in sent or "you are very kind" in sent:
                    answers['usr'] = self.domain.YES

        return answers

    def answer_DONATION_AMOUNT(self, sent, who, last_sent):
        answers = {'usr': None,
                   'sys': None}
        num_in_sent = toNumReg(sent)

        if last_sent and SystemAct.propose_donation_inquiry in self.history_label[-1]:
            # 1) asked
            score = self.sentiment_analyzer.polarity_scores(sent)
            if who == self.domain.USR:
                # 1.1) asked-user speak
                # 1.1.1) asked-user speak-about self
                if "yes" in sent or "i would like to donate" in sent or "i can donate a bit" in sent\
                    or "how much do you suggest" in sent:
                    if num_in_sent:
                        answers['usr'] = num_in_sent
                    else:
                        answers['usr'] = self.domain.NOT_SURE
                elif "prefer to donate time" in sent or "next time" in sent: 
                    answers['usr'] = 0

               # 1.1.2) asked-user speak-about system
                if "that's kind of you" in sent:
                    answers['sys'] = self.domain.NOT_SURE

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if "yes" in sent or "i would like to donate some money" in sent or "i can donate a bit" in sent\
                    or "how much do you suggest" in sent:
                    if num_in_sent:
                        answers['sys'] = num_in_sent
                elif "prefer to donate time" in sent or "next time" in sent: 
                    answers['sys'] = 0
            
                # 1.2.2) asked-system speak-about user
                if "thank you so much" in sent or "you are very kind" in sent:
                    if num_in_sent:
                        answers['usr'] = num_in_sent
                    else:
                        answers['usr'] = self.domain.NOT_SURE

        else:
            # not asked
            if who == self.domain.USR:
                # 1.1) asked-user speak
                # 1.1.1) asked-user speak-about self
                if "yes" in sent or "i would like to donate" in sent or "i can donate a bit" in sent\
                    or "how much do you suggest" in sent:
                    if num_in_sent:
                        answers['usr'] = num_in_sent
                    else:
                        answers['usr'] = self.domain.NOT_SURE
                elif "prefer to donate time" in sent or "next time" in sent: 
                    answers['usr'] = 0

               # 1.1.2) asked-user speak-about system
                if "that's kind of you" in sent:
                    answers['sys'] = self.domain.NOT_SURE

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if "yes" in sent or "i would like to donate some money" in sent or "i can donate a bit" in sent\
                    or "how much do you suggest" in sent:
                    if num_in_sent:
                        answers['sys'] = num_in_sent
                elif "prefer to donate time" in sent or "next time" in sent: 
                    answers['sys'] = 0
            
                # 1.2.2) asked-system speak-about user
                if "thank you so much" in sent or "you are very kind" in sent:
                    if num_in_sent:
                        answers['usr'] = num_in_sent
                    else:
                        answers['usr'] = self.domain.NOT_SURE

        return answers

    def refresh(self):
        self.sys_world.refresh()
        self.usr_world.refresh()
        self.history = []
        self.history_label = []
        self.last_sents = None
        self.last_labels = None
        self.answers_extracted_from_last_sent = None

    def update(self, sents, sent_labels, who):
        to_update_dic_usr, to_update_dic_sys = self.extract_info(sents, who=who)

        if who == self.domain.USR:
            self.usr_world.update(to_update_dic_usr=to_update_dic_usr, 
                                  to_update_dic_sys=to_update_dic_sys,
                                  usr_texts=sents, 
                                  last_sys_labels=self.history_label[-1])
            # sync up with the partner, for potential conditional generation
            self.sys_world.syncup(self.usr_world.usr_profile)

        elif who == self.domain.SYS:
            self.sys_world.update(to_update_dic_usr=to_update_dic_usr, 
                                  to_update_dic_sys=to_update_dic_sys,
                                  sys_texts=sents, 
                                  sys_labels=sent_labels)
            # sync up with the partner, for potential conditional generation
            self.usr_world.syncup(self.sys_world.sys_profile)

        
        self.history.append(sents)
        self.history_label.append(sent_labels)
        self.last_sents = sents
        self.last_labels = sent_labels
        self.answers_extracted_from_last_sent = {self.domain.USR: to_update_dic_usr, 
                                                 self.domain.SYS: to_update_dic_sys}

    def print(self):
        print("\n*******system profile*******")
        self.sys_world.print()

        print("\n*******user profile*******")
        self.usr_world.print()
        print("****************************\n")

    def get_profiles(self):
        profile = {'sys_world': {'sys_profile': deepcopy(self.sys_world.sys_profile),
                       'usr_profile': deepcopy(self.sys_world.usr_profile)},
         'usr_world': {'sys_profile': deepcopy(self.usr_world.sys_profile),
                       'usr_profile': deepcopy(self.usr_world.usr_profile)}}
        return profile

    def check_conflict(self, sents, sent_acts):
        # 1. repetition
        rep_status_with_sys, rep_amount_with_sys, edited_sents, edited_sent_acts = self.sys_world.check_conflict(sents, sent_acts)
        if self.last_sents is None:
            rep_condition = (rep_status_with_sys in [cfg.PASS])    
            rep_amount = rep_amount_with_sys               
        else:
            rep_status_with_usr, rep_amount_with_usr, edited_sents, edited_sent_acts = self.usr_world.check_conflict(edited_sents, edited_sent_acts)                    
            rep_condition = (rep_status_with_sys in [cfg.PASS]) and (rep_status_with_usr in [cfg.PASS])
            rep_amount = max(rep_amount_with_usr, rep_amount_with_sys)

        # 2. inconsistency
        if rep_condition:
            # if it's not a repetition, then we need to check for consistency
            consis_status = self.check_consistency(edited_sents, edited_sent_acts)
            consis_condition = consis_status in [cfg.PASS]

            rep_consis_condition = rep_condition and consis_condition
        else:
            rep_consis_condition = rep_condition
            
        return rep_consis_condition, rep_amount, edited_sents, edited_sent_acts

    def check_consistency(self, sents, sent_acts):
        to_update_dic_usr, to_update_dic_sys = self.extract_info(sents, who=self.domain.SYS)
        consis_status = cfg.PASS
        for att, answer in to_update_dic_usr.items():
            # if the system says something that's different from the user profile before
            if self.usr_world.usr_profile[att] != answer:
                consis_status = cfg.INCONSISTENCY

        for att, answer in to_update_dic_sys.items():
            # if the system says something that's different from the system profile before
            if self.sys_world.sys_profile[att] != self.domain.INIT \
               and self.sys_world.sys_profile[att] != answer:
                consis_status = cfg.INCONSISTENCY
        
        return consis_status

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

            elif re.search(r"((how are)|(how're)) you", utt):
                label = SystemAct.greeting_inquiry
            ##====== above are all inquiries =======

            elif predicted_label in ['provide-org-facts', 'provide-donation-procedure']:
                label = predicted_label

            else:
                label = predicted_label

            return label

        labels = [regex_label_for_one_utt(utt) for utt in sys_texts]

        return labels


class IndividualWorld(object):
    def __init__(self, domain, name): 
        self.domain = domain  
        self.name = name    
        self.sys_profile = {att: self.domain.INIT for att in self.domain.attributes}
        self.usr_profile = {att: self.domain.INIT for att in self.domain.attributes}

        self.sent_profile = {}

    def refresh(self): 
        self.sys_profile = {att: self.domain.INIT for att in self.domain.attributes}
        self.usr_profile = {att: self.domain.INIT for att in self.domain.attributes}
        self.sent_profile = {}
    
    # def __call__(self):
    #     return self.profile

    def print(self):
        print("********* {} *********".format(self.name))
        print("******* system profile *******")
        for k, v in self.sys_profile.items():
            print("{}: {}".format(k, v))
        print("******* user profile *******")
        for k, v in self.usr_profile.items():
            print("{}: {}".format(k, v))
        print("******* sent profile *******")
        for k, v in self.sent_profile.items():
            print("{}: {}".format(k, v))
        print("*********************")

    def keys(self, who):
        if who == self.domain.SYS:
            return list(self.sys_profile.keys())
        else:
            return list(self.usr_profile.keys())

    def values(self, who):
        if who == self.domain.SYS:
            return list(self.sys_profile.values())
        else:
            return list(self.usr_profile.values())

    def items(self, who):
        if who == self.domain.SYS:
            return self.sys_profile.items()
        else:
            return list(self.usr_profile.values())
           

class UsrWorld(IndividualWorld):
    def __init__(self, domain, name="user_world"):
        super().__init__(domain, name)

    def update(self, to_update_dic_usr, to_update_dic_sys, usr_texts, last_sys_labels):
        # update using the user inputs
        self.usr_profile.update(to_update_dic_usr)
        self.sys_profile.update(to_update_dic_sys)

        usr_text = " ".join(usr_texts)
        for last_sys_label in last_sys_labels:
            if last_sys_label in self.sent_profile:
                self.sent_profile[last_sys_label].append(usr_text)
            else:
                self.sent_profile[last_sys_label] = [usr_text]
    
    def check_conflict(self, sys_texts, sys_labels):
        # check system candidates' confict with the system profiles
        # e.g. user: i want to donate a dollar
        #      sys: how much will you donate?
        def check_conflict_for_one_utt(sys_text, sys_label):
            is_repetition, repetition_ratio = is_repetition_with_context(sys_text, 
                                                                        itertools.chain(*self.sent_profile.values()), 
                                                                        threshold=cfg.repetition_threshold)

            if "inquiry" in sys_label:
                # e.g. sys: have you heard of save the children?
                #      usr: i have heard of it
                #      sys: do you know about save the children?
                if self.is_inquiry_answered(sys_text, sys_label):
                    # 1.1 real repetition, 
                    # this is repetition inquiry
                    if cfg.debug:
                        print("{} inquiry encountered in user_profile check! {}: {}\n".format(cfg.REPETITION, sys_label, sys_text))
                    return cfg.REPETITION, repetition_ratio
                else:
                    # 1.2 fake repetition,
                    # where the user never replies the inquiry
                    return cfg.PASS, repetition_ratio
            elif is_repetition:
                # if cfg.debug:
                #     print("exact repetition with user utterance encountered in user_profile check! {}: {}".format(sys_label, sys_text))
                
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
                                                                        itertools.chain(*self.sent_profile.values()), 
                                                                        threshold=cfg.repetition_threshold)

                return cfg.NOT_PASS, repetition_ratio, sys_texts, sys_labels
            else:
                is_repetition, repetition_ratio = is_repetition_with_context(" ".join(edited_sents), 
                                                                        itertools.chain(*self.sent_profile.values()), 
                                                                        threshold=cfg.repetition_threshold)

                return cfg.PASS, repetition_ratio, edited_sents, edited_sent_acts
  

    def is_inquiry_answered(self, sys_text, sys_label):
        # temporary!!!
        act_to_attributes = {SystemAct.kids_related_inquiry: self.domain.HAVE_KIDS,
                             SystemAct.donation_related_inquiry: self.domain.DONATED_BEFORE,
                             SystemAct.organization_related_inquiry: self.domain.HEARD_OF_THE_ORG,
                             SystemAct.propose_donation_inquiry: self.domain.WANT_TO_DONATE,
                             SystemAct.greeting_inquiry: self.domain.HOW_ARE_YOU}
        # kids_related_inquiry = "kids-related-inquiry"
        # donation_related_inquiry = "donation-related-inquiry"
        # organization_related_inquiry = "organization-related-inquiry"
        # other_inquiry = "other-inquiry"
        # propose_donation_inquiry = "propose-donation-inquiry"

        # print("in is_inquiry_answered")
        # print(sys_text)
        # print(sys_label)
        if sys_label in act_to_attributes:
            if self.usr_profile[act_to_attributes[sys_label]] != self.domain.INIT:
                return True
            else:
                return False
        else:        
            if sys_label not in self.sent_profile:
                return False
            usr_responses_to_inqury = self.sent_profile[sys_label] #todo, temporarily, should use all user context to find if the answer is already there
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

    def syncup(self, profile_to_sync, whose=None):
        if whose is None:
            whose = self.domain.SYS
        if whose == self.domain.USR:
            for k, v in profile_to_sync.items():
                self.usr_profile[k] = v
        else:
            for k, v in profile_to_sync.items():
                self.sys_profile[k] = v
    

class SysWorld(IndividualWorld):
    def __init__(self, domain, name="user_world"):
        super().__init__(domain, name)
        self.sys_profile[self.domain.HEARD_OF_THE_ORG] = self.domain.YES
        self.sys_profile[self.domain.WANT_TO_DONATE] = self.domain.YES

    def update(self, to_update_dic_usr, to_update_dic_sys, sys_texts, sys_labels):
        # update using the system inputs
        # label = self.regex_label(sys_text, context, turn_i)
        self.usr_profile.update(to_update_dic_usr)
        self.sys_profile.update(to_update_dic_sys)

        for sys_text, sys_label in zip(sys_texts, sys_labels):
            if sys_label in self.sent_profile:
                self.sent_profile[sys_label].append(sys_text)
            else:
                self.sent_profile[sys_label] = [sys_text]
        
    def check_conflict(self, sys_texts, sys_labels):
        # check system candidates' confict with the user profiles
        # label = self.regex_label(sys_text, context, turn_i)
        def check_conflict_for_one_utt(sys_text, sys_label):
            is_repetition, repetition_ratio = is_repetition_with_context(sys_text, 
                                                                        itertools.chain(*self.sent_profile.values()), 
                                                                        threshold=cfg.repetition_threshold)

            
            if True: #sys_label in self.sent_profile:
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

            # else:
            #     if is_repetition:
            #         return cfg.REPETITION, repetition_ratio
            #     else:
            #         return cfg.PASS, repetition_ratio

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
                                                                        itertools.chain(*self.sent_profile.values()), 
                                                                        threshold=cfg.repetition_threshold)

                return cfg.NOT_PASS, repetition_ratio, sys_texts, sys_labels
            else:
                is_repetition, repetition_ratio = is_repetition_with_context(" ".join(edited_sents), 
                                                                        itertools.chain(*self.sent_profile.values()), 
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

    def refresh(self):
        super().refresh()
        self.sys_profile[self.domain.HEARD_OF_THE_ORG] = self.domain.YES
        self.sys_profile[self.domain.WANT_TO_DONATE] = self.domain.YES
        
    def syncup(self, profile_to_sync, whose=None):
        if whose is None:
            whose = self.domain.USR

        if whose == self.domain.USR:
            for k, v in profile_to_sync.items():
                self.usr_profile[k] = v
        else:
            for k, v in profile_to_sync.items():
                self.sys_profile[k] = v

    # def correct_user(self, ):


