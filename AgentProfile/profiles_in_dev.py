import re
import sys
sys.path.append("../")
# from classifier.pred import strategy_model
import config as cfg
from utils import is_repetition_with_context
import itertools
from AgentProfile.core import SystemAct, UserAct
from KnowledgeBase import KB
from KnowledgeBase.KB import Domain
import pdb
import time

from nltk.tokenize import sent_tokenize
from copy import deepcopy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

from utils import toNumReg
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import pdb
import logging

# HOW_ARE_YOU = 'how-are-you'
# HEARD_OF_THE_ORG = 'heard-of-the-org'
# HAVE_KIDS = 'have-kids'
# DONATED_BEFORE = 'donated-before'
# WANT_TO_DONATE = 'want-to-donate'
# DONATION_AMOUNT = 'donation-amount'


class GlobalProfile(object):
    def __init__(self, domain, model_config, sent_embedding_model=None):
        self.domain = domain
        self.model_config = model_config
        # self.act_clf_model = act_clf_model
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.sent_embedding_model = SentenceTransformer('bert-base-nli-mean-tokens', 
                                                        device=torch.device(cfg.sent_embedding_model_device)) if sent_embedding_model is None else sent_embedding_model#('roberta-large-nli-stsb-mean-tokens')

        self.sys_world = SysWorld(domain=self.domain, name="system_world")
        self.usr_world = UsrWorld(domain=self.domain, name="user_world")
        self.history = []
        self.history_label = []
        self.last_sents = None
        self.last_labels = None
        self.answers_extracted_from_last_sent = None

        self.greeting_re = re.compile(r"((how are you)|(how're you)|(how you doing)|(how are you doing)|(how're you doing))", re.IGNORECASE)
        
    # def __getitem__(self, key):
    #     return self.profiles[key.lower()]

    def extract_info(self, sents, who, sent_acts):

        qa_dict_usr, qa_dict_sys = self.contain_answers(sents, who, sent_acts)
        
        return qa_dict_usr, qa_dict_sys

    def sents_are_similar(self, sent, sent_list):
        sents = [sent] + sent_list
        sent_embeddings = self.sent_embedding_model.encode(sents)

        scores = []
        for i in range(1, len(sent_embeddings)):
            score = cosine_similarity(sent_embeddings[0].reshape(1, -1),
                                 sent_embeddings[i].reshape(1, -1)) 
            if score >= cfg.similarity_threshold:
                return True
            # scores.append(score)
        return False



    def contain_answers(self, sents, who, sent_acts):
        qa_dict = {
                    self.domain.USR:{},
                    self.domain.SYS:{}
                  }
        for q in self.domain.attributes:
            answers = self.answer_question(sents, q, who, sent_acts)
            
            if answers['usr'] is not None:
                qa_dict[self.domain.USR].update({q: answers['usr']})
            if answers['sys'] is not None:
                qa_dict[self.domain.SYS].update({q: answers['sys']})
                
        for side in qa_dict:
            if self.domain.DONATION_AMOUNT in qa_dict[side]:
                try:
                    donation_amt = float(qa_dict[side][self.domain.DONATION_AMOUNT])
                    if donation_amt > 0:
                        qa_dict[side][self.domain.WANT_TO_DONATE] = self.domain.YES
                    elif donation_amt <= 0:
                        qa_dict[side][self.domain.WANT_TO_DONATE] = self.domain.NO
                except:
                    pass

        sent = " ".join(sents).lower()
        if re.search(r"((me,)|(me)) too", sent, re.IGNORECASE) \
            or "same here" in sent:
            if who == self.domain.USR:
                qa_dict[self.domain.USR].update(self.answers_extracted_from_last_sent[self.domain.SYS])
            else:
                qa_dict[self.domain.SYS].update(self.answers_extracted_from_last_sent[self.domain.USR])
        return qa_dict[self.domain.USR], qa_dict[self.domain.SYS]

    def answer_question(self, sents, q, who, sent_acts):
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
            answers = self.answer_HOW_ARE_YOU(sent, who, last_sent, sent_acts)

        elif q == self.domain.HEARD_OF_THE_ORG:    
            answers = self.answer_HEARD_OF_THE_ORG(sent, who, last_sent, sent_acts)

        elif q == self.domain.HAVE_KIDS:
            answers = self.answer_HAVE_KIDS(sent, who, last_sent, sent_acts)

        elif q == self.domain.DONATED_BEFORE:
            answers = self.answer_DONATED_BEFORE(sent, who, last_sent, sent_acts)

        elif q == self.domain.WANT_TO_DONATE:
            answers = self.answer_WANT_TO_DONATE(sent, who, last_sent, sent_acts)

        elif q == self.domain.DONATION_AMOUNT:
            answers = self.answer_DONATION_AMOUNT(sent, who, last_sent, sent_acts)

        else:
            raise ValueError("question {} not supported".format(q))
        

        if "as well" in sent:
            if who == self.domain.USR:
                answers['sys'] = answers['usr']
            else:
                answers['usr'] = answers['sys']

            
        return answers

    def answer_HOW_ARE_YOU(self, sent, who, last_sent, sent_acts):
        # self.domain.ATT_TO_QUESTION[self.domain.HOW_ARE_YOU]
        answers = {'usr': None,
                   'sys': None}
        if last_sent and (re.search(self.greeting_re, last_sent) or (SystemAct.greeting_inquiry in self.last_labels) \
                          or (UserAct.greeting_inquiry in self.last_labels)): 
            # asked
            if re.search(r"((you are)|(you're)) doing ((great)|(well)|(good))", sent):
                if who == self.domain.USR:
                    answers['sys'] = "good"
                elif who == self.domain.SYS:
                    answers['usr'] = "good"
            elif re.search(r"((doing )| )((good)|(well)|(alright)|(ok)|(great))", sent) \
                or (UserAct.greeting_answer in sent_acts) or (SystemAct.greeting_answer in sent_acts):
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
            elif re.search(r"((doing )| )((good)|(well)|(alright)|(ok))", sent) \
                or (UserAct.greeting_answer in sent_acts) or (SystemAct.greeting_answer in sent_acts):
                if who == self.domain.USR:
                    answers['usr'] = "good"
                elif who == self.domain.SYS:
                    answers['sys'] = "good"
        return answers

    def answer_HEARD_OF_THE_ORG(self, sent, who, last_sent, sent_acts):
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
                    if UserAct.POSITIVE_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.YES
                    elif UserAct.NEUTRAL_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.NOT_SURE
                    elif UserAct.NEGATIVE_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.NO
                    else:
                        answers['usr'] = self.domain.NOT_SURE
                        # if score['compound'] >= 0.05:
                        #     # positive
                        #     answers['usr'] = self.domain.YES
                        # elif score['compound'] <= -0.05:
                        #     answers['usr'] = self.domain.NO
                        # else:
                        #     answers['usr'] = self.domain.NOT_SURE

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
                    answers['sys'] = self.domain.NOT_SURE
                    # if score['compound'] >= 0.05:
                    #     # positive
                    #     answers['sys'] = self.domain.YES
                    # elif score['compound'] <= -0.05:
                    #     answers['sys'] = self.domain.NO
                    # else:
                    #     answers['sys'] = self.domain.NOT_SURE
            
                # 1.2.2) asked-system speak-about user
                if re.search(r"((you've)|(you have)) heard", sent, re.IGNORECASE)\
                     or "you know about" in sent:
                    answers['usr'] = self.domain.YES

        # else:
        if True:
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

    def answer_HAVE_KIDS(self, sent, who, last_sent, sent_acts):
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
                    if UserAct.POSITIVE_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.YES
                    elif UserAct.NEUTRAL_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.NOT_SURE
                    elif UserAct.NEGATIVE_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.NO
                    else:
                        answers['usr'] = self.domain.NOT_SURE
                        # if score['compound'] >= 0.05:
                        #     # positive
                        #     answers['usr'] = self.domain.YES
                        # elif score['compound'] <= -0.05:
                        #     answers['usr'] = self.domain.NO
                        # else:
                        #     answers['usr'] = self.domain.NOT_SURE

                # 1.1.2) asked-user speak-about system
                if re.search(r"((you've)|(you have))", sent, re.IGNORECASE) and "kid" in sent:
                    answers['sys'] = self.domain.YES

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if "yes" in sent:
                    answers['sys'] = self.domain.YES
                elif "no" in sent: 
                    answers['sys'] = self.domain.NO
                else:
                    answers['sys'] = self.domain.NOT_SURE
                    # if score['compound'] >= 0.05:
                    #     # positive
                    #     answers['sys'] = self.domain.YES
                    # elif score['compound'] <= -0.05:
                    #     answers['sys'] = self.domain.NO
                    # else:
                    #     answers['sys'] = self.domain.NOT_SURE
            
                # 1.2.2) asked-system speak-about user
                if re.search(r"((you've)|(you have))", sent, re.IGNORECASE) and "kid" in sent:
                    answers['usr'] = self.domain.YES

        # else:
        if True:
            # not asked
            if who == self.domain.USR:
                # 2.1) asked-user speak
                # 2.1.1) asked-user speak-about self
                if re.search(r"(((i've)|(i have)) (([-+]?[1-9]+)|(one)|(two)|(three)|(four)|(five)|(six)|(seven)|(eight)|(nine)|(zero)) ((child)|(kid)))|(my ((kid)|(child)|(boy)|(girl)))", sent, re.IGNORECASE):
                    answers['usr'] = self.domain.YES

                # 1.1.2) asked-user speak-about system
                if re.search(r"(((you've)|(you have)) (([-+]?[1-9]+)|(one)|(two)|(three)|(four)|(five)|(six)|(seven)|(eight)|(nine)|(zero)) ((child)|(kid)))|(your ((kid)|(child)|(boy)|(girl)))", sent, re.IGNORECASE):
                    answers['sys'] = self.domain.YES

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if re.search(r"(((i've)|(i have)) (([-+]?[0-9]+)|(one)|(two)|(three)|(four)|(five)|(six)|(seven)|(eight)|(nine)|(zero)) ((child)|(kid)))|(my ((kid)|(child)|(boy)|(girl)))", sent, re.IGNORECASE):
                    answers['sys'] = self.domain.YES
            
                # 1.2.2) asked-system speak-about user
                if re.search(r"(((you've)|(you have)) (([-+]?[1-9]+)|(one)|(two)|(three)|(four)|(five)|(six)|(seven)|(eight)|(nine)|(zero)) ((child)|(kid)))|(your ((kid)|(child)|(boy)|(girl)))", sent, re.IGNORECASE):
                    answers['usr'] = self.domain.YES
                
        return answers

    def answer_DONATED_BEFORE(self, sent, who, last_sent, sent_acts):
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
                    if UserAct.POSITIVE_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.YES
                    elif UserAct.NEUTRAL_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.NOT_SURE
                    elif UserAct.NEGATIVE_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.NO
                    else:
                        answers['usr'] = self.domain.NOT_SURE
                        # if score['compound'] >= 0.05:
                        #     # positive
                        #     answers['usr'] = self.domain.YES
                        # elif score['compound'] <= -0.05:
                        #     answers['usr'] = self.domain.NO
                        # else:
                        #     answers['usr'] = self.domain.NOT_SURE

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
                    answers['sys'] = self.domain.NOT_SURE
                    # if score['compound'] >= 0.05:
                    #     # positive
                    #     answers['sys'] = self.domain.YES
                    # elif score['compound'] <= -0.05:
                    #     answers['sys'] = self.domain.NO
                    # else:
                    #     answers['sys'] = self.domain.NOT_SURE
            
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

    def answer_WANT_TO_DONATE(self, sent, who, last_sent, sent_acts):
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
                    or "how much do you suggest" in sent or \
                    UserAct.AGREE_DONATION in sent_acts or \
                    UserAct.PROVIDE_DONATION_AMOUNT in sent_acts:
                    if "how much do you suggest" in sent:
                        answers['usr'] = self.domain.NOT_SURE
                    else:
                        if num_in_sent is not None and num_in_sent == 0:
                            answers['usr'] = self.domain.NO
                        else:
                            answers['usr'] = self.domain.YES
                elif "prefer to donate time" in sent or "next time" in sent or "not today" in sent or\
                    UserAct.DISAGREE_DONATION in sent_acts: 
                    answers['usr'] = self.domain.NO
                else:
                    if UserAct.POSITIVE_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.YES
                    elif UserAct.NEUTRAL_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.NOT_SURE
                    elif UserAct.NEGATIVE_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.NO
                    else:
                        answers['usr'] = self.domain.NOT_SURE
                        # if score['compound'] >= 0.05:
                        #     # positive
                        #     if num_in_sent is not None and num_in_sent == 0:
                        #         answers['usr'] = self.domain.NO
                        #     else:
                        #         answers['usr'] = self.domain.YES
                        # elif score['compound'] <= -0.05:
                        #     answers['usr'] = self.domain.NO
                        # else:
                        #     answers['usr'] = self.domain.NOT_SURE

                # 1.1.2) asked-user speak-about system
                if "that's kind of you" in sent:
                    answers['sys'] = self.domain.YES

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if "yes" in sent or "i would like to donate some money" in sent or "i can donate a bit" in sent\
                    or "how much do you suggest" in sent:
                    if "how much do you suggest" in sent:
                        answers['sys'] = self.domain.NOT_SURE
                    else:
                        answers['sys'] = self.domain.YES
                elif "prefer to donate time" in sent or "next time" in sent or "not today" in sent: 
                    answers['sys'] = self.domain.NO
                else:
                    answers['sys'] = self.domain.NOT_SURE
                    # if score['compound'] >= 0.05:
                    #     # positive
                    #     answers['sys'] = self.domain.YES
                    # elif score['compound'] <= -0.05:
                    #     answers['sys'] = self.domain.NO
                    # else:
                    #     answers['sys'] = self.domain.NOT_SURE
            
                # 1.2.2) asked-system speak-about user
                if "thank you so much" in sent or "you are very kind" in sent:
                    answers['usr'] = self.domain.YES

        else:
            # not asked
            if who == self.domain.USR:
                if "i would like to donate" in sent or "i can donate a bit" in sent\
                    or "how much do you suggest" in sent or "glad to help" in sent\
                    or "i will donate" in sent or\
                        UserAct.AGREE_DONATION in sent_acts or\
                            UserAct.PROVIDE_DONATION_AMOUNT in sent_acts:
                    if "how much do you suggest" in sent:
                        answers['usr'] = self.domain.NOT_SURE
                    else:
                        if num_in_sent is not None and num_in_sent == 0:
                            answers['usr'] = self.domain.NO
                        else:
                            answers['usr'] = self.domain.YES
                    # answers['usr'] = self.domain.YES
                elif "prefer to donate time" in sent or "next time" in sent or "not today" in sent\
                or "not interested" in sent or "don't want to donate" in sent or \
                    UserAct.DISAGREE_DONATION in sent_acts: 
                    answers['usr'] = self.domain.NO

                # 2.1.2) asked-user speak-about system
                if "that's kind of you" in sent:
                    answers['sys'] = self.domain.YES

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if "i would like to donate some money" in sent or "i can donate a bit" in sent\
                    or "how much do you suggest" in sent:
                    if "how much do you suggest" in sent:
                        answers['sys'] = self.domain.NOT_SURE
                    else:
                        answers['sys'] = self.domain.YES
                elif "prefer to donate time" in sent or "next time" in sent or\
                    "i just don't donate" in sent or "i don't donate" in sent: 
                    answers['sys'] = self.domain.NO
            
                # 1.2.2) asked-system speak-about user
                if "thank you so much" in sent or "you are very kind" in sent \
                or "you decided to donate" in sent or "you decide to donate" in sent or "thank you for your donation today" in sent or "you are willing to donate" in sent\
                or "appreciate your donation" in sent or "appreciate your willingness to donate" in sent or\
                    SystemAct.THANK in sent_acts or SystemAct.CONFIRM_DONATION in sent_acts or\
                         SystemAct.PRAISE_USER in sent_acts:
                    # if num_in_sent is not None and num_in_sent == 0:
                    #     answers['usr'] = self.domain.NO
                    # else:
                    answers['usr'] = self.domain.YES
                
                # elif "ould you like to donate some of your task payment" in sent or \
                #     SystemAct.propose_donation_inquiry in sent_acts:
                #     answers['usr'] = self.domain.INIT
                
                elif SystemAct.ASK_NOT_DONATE_REASON in sent_acts:
                    answers['usr'] = self.domain.NO

        return answers

    def answer_DONATION_AMOUNT(self, sent, who, last_sent, sent_acts):
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
                    or "how much do you suggest" in sent or \
                    UserAct.AGREE_DONATION in sent_acts or \
                    UserAct.PROVIDE_DONATION_AMOUNT in sent_acts:
                    if "how much do you suggest" in sent:
                        answers['usr'] = self.domain.NOT_SURE
                    else:
                        if num_in_sent is not None:
                            answers['usr'] = num_in_sent
                        else:
                            answers['usr'] = self.domain.NOT_SURE
                elif "prefer to donate time" in sent or "next time" in sent or "not today" in sent or\
                    UserAct.DISAGREE_DONATION in sent_acts: 
                    answers['usr'] = 0
                else:
                    if UserAct.POSITIVE_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.YES
                    elif UserAct.NEUTRAL_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.NOT_SURE
                    elif UserAct.NEGATIVE_TO_INQUIRY in sent_acts:
                        answers['usr'] = self.domain.NO
                    else:
                        answers['usr'] = self.domain.NOT_SURE
                        # if score['compound'] >= 0.05:
                        #     # positive
                        #     answers['usr'] = self.domain.YES
                        # elif score['compound'] <= -0.05:
                        #     answers['usr'] = self.domain.NO
                        # else:
                        #     answers['usr'] = self.domain.NOT_SURE

               # 1.1.2) asked-user speak-about system
                if "that's kind of you" in sent:
                    answers['sys'] = self.domain.NOT_SURE

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if "yes" in sent or "i would like to donate some money" in sent or "i can donate a bit" in sent\
                    or "how much do you suggest" in sent:
                    if "how much do you suggest" in sent:
                        answers['sys'] = self.domain.NOT_SURE
                    else:
                        if num_in_sent is not None:
                            answers['sys'] = num_in_sent
                elif "prefer to donate time" in sent or "next time" in sent: 
                    answers['sys'] = 0
            
                # 1.2.2) asked-system speak-about user
                if "thank you so much" in sent or "you are very kind" in sent:
                    if num_in_sent is not None:
                        answers['usr'] = num_in_sent
                    else:
                        answers['usr'] = self.domain.NOT_SURE

        else:
            # not asked
            if who == self.domain.USR:
                # 1.1) asked-user speak
                # 1.1.1) asked-user speak-about self
                if "i would like to donate" in sent or "i can donate a bit" in sent\
                    or "how much do you suggest" in sent or "glad to help" in sent\
                    or "i will donate" in sent or\
                        UserAct.AGREE_DONATION in sent_acts or\
                            UserAct.PROVIDE_DONATION_AMOUNT in sent_acts:
                    if "how much do you suggest" in sent:
                        answers['usr'] = self.domain.NOT_SURE
                    else:
                        if num_in_sent is not None:
                            answers['usr'] = num_in_sent
                        else:
                            answers['usr'] = self.domain.NOT_SURE
                elif "prefer to donate time" in sent or "next time" in sent or "not today" in sent\
                or "not interested" in sent or "don't want to donate" in sent or \
                    UserAct.DISAGREE_DONATION in sent_acts: 
                    answers['usr'] = 0

               # 1.1.2) asked-user speak-about system
                if "that's kind of you" in sent:
                    answers['sys'] = self.domain.NOT_SURE

            elif who == self.domain.SYS:
                # 1.2) asked-system
                # 1.2.1) asked-system speak-about self
                if "yes" in sent or "i would like to donate some money" in sent or "i can donate a bit" in sent\
                    or "how much do you suggest" in sent\
                        or SystemAct.EXAMPLE_DONATION in sent_acts:
                    if "how much do you suggest" in sent:
                        answers['sys'] = self.domain.NOT_SURE
                    else:
                        if num_in_sent is not None:
                            answers['sys'] = num_in_sent
                elif "prefer to donate time" in sent or "next time" in sent: 
                    answers['sys'] = 0
            
                # 1.2.2) asked-system speak-about user                
                # if "ould you like to donate some of your task payment" in sent or \
                #     SystemAct.propose_donation_inquiry in sent_acts:
                #     answers['usr'] = self.domain.INIT
                
                if SystemAct.ASK_NOT_DONATE_REASON in sent_acts:
                    answers['usr'] = 0

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
        to_update_dic_usr, to_update_dic_sys = self.extract_info(sents, who=who, sent_acts=sent_labels)
        # 
        print("things to update****************************************************")
        print(f"{who}: {sent_labels}: {sents}")
        print(f"{to_update_dic_usr}\n{to_update_dic_sys}")
        print("********************************************************************")
        logging.info("things to update****************************************************")
        logging.info(f"{who}: {sent_labels}: {sents}")
        logging.info(f"{to_update_dic_usr}\n{to_update_dic_sys}")
        logging.info("********************************************************************")
        # pdb.set_trace()
        if who == self.domain.USR:
            # last_sys_labels_temp = self.history_label[-1] if len(self.history_label) else None
            # last_sys_sents_temp = self.history[-1] if len(self.history) else None
            self.usr_world.update(to_update_dic_usr=to_update_dic_usr, 
                                  to_update_dic_sys=to_update_dic_sys,
                                  usr_texts=sents, 
                                  usr_labels=sent_labels)#,
                                #   last_sys_labels=last_sys_labels_temp,
                                #   last_sys_sents=last_sys_sents_temp)
            # sync up with the partner, for potential conditional generation
            self.sys_world.syncup(self.usr_world.usr_profile, last_usr_labels=sent_labels, last_usr_sents=sents)

        elif who == self.domain.SYS:
            # pdb.set_trace()
            # last_usr_labels_temp = self.history_label[-1] if len(self.history_label) else None
            # last_usr_sents_temp = self.history[-1] if len(self.history) else None
            self.sys_world.update(to_update_dic_usr=to_update_dic_usr, 
                                  to_update_dic_sys=to_update_dic_sys,
                                  sys_texts=sents, 
                                  sys_labels=sent_labels)#,
                                #   last_usr_labels=last_usr_labels_temp,
                                #   last_usr_sents=last_usr_sents_temp)
            # sync up with the partner, for potential conditional generation
            self.usr_world.syncup(self.sys_world.sys_profile, last_sys_labels=sent_labels, last_sys_sents=sents)

        
        self.history.append(sents)
        self.history_label.append(sent_labels)
        self.last_sents = sents
        self.last_labels = sent_labels
        self.answers_extracted_from_last_sent = {self.domain.USR: to_update_dic_usr, 
                                                 self.domain.SYS: to_update_dic_sys}

    def print(self):
        print("\n*******system profile************************************")
        self.sys_world.print()
        print("***********************************************************\n")

        print("\n*******user profile**************************************")
        self.usr_world.print()
        print("***********************************************************\n")

    def print_to_log(self):
        logging.info("\n*******system profile************************************")
        self.sys_world.print_to_log()
        logging.info("***********************************************************\n")

        logging.info("\n*******user profile**************************************")
        self.usr_world.print_to_log()
        logging.info("***********************************************************\n")

    def get_profiles(self):
        profile = {'sys_world': {'sys_profile': deepcopy(self.sys_world.sys_profile),
                       'usr_profile': deepcopy(self.sys_world.usr_profile)},
         'usr_world': {'sys_profile': deepcopy(self.usr_world.sys_profile),
                       'usr_profile': deepcopy(self.usr_world.usr_profile)}}
        return profile

    def check_conflict(self, sents, sent_acts):
        if self.model_config.with_repetition_module:
            # 1. repetition
            # pdb.set_trace()
            fail_reason = None
            rep_status_with_sys, rep_amount_with_sys, edited_sents, edited_sent_acts = self.sys_world.check_conflict(sents, sent_acts)
            if self.last_sents is None:
                rep_condition = (rep_status_with_sys in [cfg.PASS])    
                rep_amount = rep_amount_with_sys               
            else:
                is_repetition_with_sys = not (rep_status_with_sys in [cfg.PASS]) 
                rep_status_with_usr, rep_amount_with_usr, edited_sents, edited_sent_acts = self.usr_world.check_conflict(edited_sents, edited_sent_acts,
                                                                                                                        is_repetition_with_sys=is_repetition_with_sys)                    
                rep_condition = (rep_status_with_sys in [cfg.PASS]) and (rep_status_with_usr in [cfg.PASS])
                rep_amount = max(rep_amount_with_usr, rep_amount_with_sys)
        else:
            rep_condition = True
            rep_amount = 0
            edited_sents = sents
            edited_sent_acts =  sent_acts
            rep_status_with_sys = "without_repetition_module"
            rep_status_with_usr = "without_repetition_module"

        if self.model_config.with_consistency_module:
            # 2. inconsistency
            if rep_condition:
                # if it's not a repetition, then we need to check for consistency
                consis_status, fail_consistency_reason = self.check_consistency(edited_sents, edited_sent_acts)
                # pdb.set_trace()
                consis_condition = consis_status in [cfg.PASS]

                rep_consis_condition = rep_condition and consis_condition
                if rep_consis_condition:
                    fail_reason = consis_status
                else:
                    fail_reason = f"<inconsistency>: {fail_consistency_reason}"
            else:
                # pdb.set_trace()
                rep_consis_condition = rep_condition
                if self.last_sents is None:
                    fail_reason = "<repetition> {} with sys, none with usr".format(rep_status_with_sys)
                else:
                    fail_reason = "<repetition> {} with sys, {} with usr".format(rep_status_with_sys, rep_status_with_usr)
        else:
            # rep_consis_condition = rep_condition
            # fail_reason = "without consistency module"
            rep_consis_condition = rep_condition
            if rep_condition:
                fail_reason = "pass repetition, without consistency module"
            else:
                if self.last_sents is None:
                    fail_reason = "<repetition> {} with sys, none with usr".format(rep_status_with_sys)
                else:
                    fail_reason = "<repetition> {} with sys, {} with usr".format(rep_status_with_sys, rep_status_with_usr)
            
        return rep_consis_condition, rep_amount, edited_sents, edited_sent_acts, fail_reason

    def check_consistency(self, sents, sent_acts):
        # pdb.set_trace()
        to_update_dic_usr, to_update_dic_sys = self.extract_info(sents, who=self.domain.SYS, sent_acts=sent_acts)
        consis_status = cfg.PASS
        fail_reason = ""
        for att, answer in to_update_dic_usr.items():
            # if the system says something that's different from the user profile before
            if self.usr_world.usr_profile[att] != answer:
                fail_reason += f"{att} is {answer}, but {self.usr_world.usr_profile[att]} in usr_world usr_profile; "
                consis_status = cfg.INCONSISTENCY

        for att, answer in to_update_dic_sys.items():
            # if the system says something that's different from the system profile before
            if self.sys_world.sys_profile[att] != self.domain.INIT \
               and self.sys_world.sys_profile[att] != answer:
                fail_reason += f"{att} is {answer}, but {self.sys_world.sys_profile[att]} in sys_world sys_profile; "
                consis_status = cfg.INCONSISTENCY
        for sent in sents:
            if "you decided to donate" in sent:
                pass
                # pdb.set_trace()
        if consis_status != cfg.PASS:
            # pdb.set_trace()
            pass
        return consis_status, fail_reason

    def regex_label(self, model_clf, sys_texts, which_task):
        """
        regex to re-label 
        vs 
        QA    to re-label
        """

        predicted_labels, past = model_clf.predict(separate_sents=sys_texts, 
                                                    which_task=which_task)
        # pdb.set_trace()
        print(f"{which_task} {predicted_labels}: {sys_texts}")
        def regex_label_for_one_utt(utt, predicted_label, which_task=which_task):
            # try:
            #     predicted_label = self.pred_model.predict(text=utt, his=context, turn=turn_i)
            # except:
            #     import pdb
            #     pdb.set_trace()
            
            sent = utt.lower()
            if "A" in which_task:
                if predicted_label in ["task-related-inquiry", "personal-related-inquiry"]:#, "have-you-heard-of-the-org"]:
                    
                    if self.sents_are_similar(sent, ['do you have children', 
                                                    'do you have kids',
                                                    'are you a parent']):#"have" in sent and (("kid" in sent) or ("children" in sent)):
                        label = SystemAct.kids_related_inquiry

                    elif self.sents_are_similar(sent, ['have you donated before', 
                                                    'do you donate to charities',
                                                    'are you involved with any charity']):
                        label = SystemAct.donation_related_inquiry

                    elif self.sents_are_similar(sent, [#'have you heard of save the children before', 
                                                    'have you heard of save the children',
                                                    'are you aware of save the children',
                                                    'are you familiar with save the children']):
                        label = SystemAct.organization_related_inquiry

                    # elif "save the children" in sent:
                    #     #todo: "Would you be interested in donating to Save the Children today?"
                    #     label = "organization-related-inquiry"

                    elif self.sents_are_similar(sent, ['how are you', 
                                                    'how are you doing today',
                                                    'how are you doing',
                                                    'how about you']):
                        label = SystemAct.greeting_inquiry

                    
                    else:
                        label = SystemAct.other_inquiry

                elif predicted_label in ["greeting"]:

                    if "?" in sent or re.search(self.greeting_re, utt) or "how" in sent:
                        label = SystemAct.greeting_inquiry
                    elif re.search(re.compile(r"((hi)|(hello)|(hi there)|(hello there)|(hey there)|(hey))", re.IGNORECASE), utt) or\
                        (len(utt.split())<=2 and not (re.search(re.compile(r"((good)|(alright)|(great)|(ok))", re.IGNORECASE), utt))):
                        label = "simple-greeting"
                    else:
                        label = SystemAct.greeting_answer

                elif predicted_label in ["source-related-inquiry"]:
                    label = SystemAct.organization_related_inquiry

                elif predicted_label in ["proposition-of-donation", "ask-donation-amount"]:
                    label = SystemAct.propose_donation_inquiry 

                elif "ask" in predicted_label: 
                    label = predicted_label + "-inquiry"

                elif re.search(self.greeting_re, utt):
                    label = SystemAct.greeting_inquiry
                ##====== above are all inquiries =======

                elif predicted_label in ['credibility-appeal', 'donation-information']:#['provide-org-facts', 'provide-donation-procedure']:
                    label = predicted_label

                else:
                    if self.sents_are_similar(sent, ['would you like to make a donation to Save the Children?', 
                                                    'I was just wondering if you would be willing to donate a portion of your task payment to Save the Children',
                                                    'Would you be willing to donate a portion of your payment',
                                                    'how much do you like to donate to the charity now',
                                                    'if you would like to consider a small portion of your payment']):
                        label = SystemAct.propose_donation_inquiry
                    else:
                        label = predicted_label
            else:
                # pdb.set_trace()
                if predicted_label in ["task-related-inquiry", "personal-related-inquiry"]:#, "have-you-heard-of-the-org"]:
                    
                    if self.sents_are_similar(sent, ['do you have children', 
                                                    'do you have kids',
                                                    'are you a parent']):#"have" in sent and (("kid" in sent) or ("children" in sent)):
                        label = UserAct.kids_related_inquiry

                    elif self.sents_are_similar(sent, ['have you donated before', 
                                                    'do you donate to charities',
                                                    'are you involved with any charity']):
                        label = UserAct.donation_related_inquiry

                    elif self.sents_are_similar(sent, [#'have you heard of save the children before', 
                                                    'how do i donate',
                                                    'how can i donate',
                                                    'how can i make a donation']):
                        label = UserAct.donation_procedure_inquiry

                    # elif "save the children" in sent:
                    #     #todo: "Would you be interested in donating to Save the Children today?"
                    #     label = "organization-related-inquiry"

                    elif self.sents_are_similar(sent, ['how are you', 
                                                    'how are you doing today',
                                                    'how are you doing',
                                                    'how about you']):
                        label = UserAct.greeting_inquiry

                    else:
                        label = UserAct.other_inquiry

                elif predicted_label in ["greeting"]:
                    if "?" in sent or re.search(self.greeting_re, utt) or "how" in sent:
                        label = UserAct.greeting_inquiry
                    elif re.search(re.compile(r"((hi)|(hello)|(hi there)|(hello there))", re.IGNORECASE), utt) and len(utt.split())<=2:
                        label = "simple-greeting"
                    else:
                        label = UserAct.greeting_answer

                elif predicted_label in [UserAct.ASK_ORG_INFO]:
                    label = UserAct.organization_info_inquiry

                elif predicted_label in [UserAct.ASK_DONATION_PROCEDURE]:
                    label = UserAct.donation_procedure_inquiry 

                elif predicted_label in [UserAct.ASK_DONATION_INTENTION]:
                    label = UserAct.persuader_intention_inquiry

                elif "ask" in predicted_label: 
                    label = predicted_label + "-inquiry"

                elif re.search(self.greeting_re, utt):
                    label = UserAct.greeting_inquiry
                ##====== above are all inquiries =======

                # elif predicted_label in ['credibility-appeal', 'donation-information']:#['provide-org-facts', 'provide-donation-procedure']:
                #     label = predicted_label

                else:
                    # if self.sents_are_similar(sent, ['would you like to make a donation to Save the Children?', 
                    #                                 'I was just wondering if you would be willing to donate a portion of your task payment to Save the Children',
                    #                                 'Would you be willing to donate a portion of your payment',
                    #                                 'how much do you like to donate to the charity now',
                    #                                 'if you would like to consider a small portion of your payment']):
                    #     label = SystemAct.propose_donation_inquiry
                    if predicted_label in [UserAct.PROVIDE_DONATION_AMOUNT, UserAct.AGREE_DONATION]:
                        num_in_sent = toNumReg(sent)
                        if num_in_sent is not None:
                            if (num_in_sent == 0):
                                label = UserAct.DISAGREE_DONATION
                            else:
                                label = predicted_label
                        else:
                            label = predicted_label
                    else:
                        label = predicted_label
            return label

        start = time.time()
        labels = [regex_label_for_one_utt(utt, predicted_label, which_task) for utt, predicted_label in zip(sys_texts, predicted_labels)]
        end = time.time()
        # print(f"regex_label takes {end-start}")
        # logging.info(f"regex_label takes {end-start}")
        return labels, past


class IndividualWorld(object):
    def __init__(self, domain, name): 
        self.domain = domain  
        self.name = name    
        self.sys_profile = {att: self.domain.INIT for att in self.domain.attributes}
        self.usr_profile = {att: self.domain.INIT for att in self.domain.attributes}

        self.sent_profile = {}
        self.act_sent_pair = []
        self.last_sents = None
        self.last_labels = None

    def refresh(self): 
        self.sys_profile = {att: self.domain.INIT for att in self.domain.attributes}
        self.usr_profile = {att: self.domain.INIT for att in self.domain.attributes}
        self.sent_profile = {}
        self.act_sent_pair = []
        self.last_sents = None
        self.last_labels = None
    
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
        print("******** act sent pair *************")
        # pdb.set_trace()
        for turn in self.act_sent_pair:
            for turn_i in turn:
                print(f"{turn_i[0]}: {turn_i[1]}")
        print("************ last sent *******************")
        print(f"last sents: {self.last_sents}")
        print(f"last labels: {self.last_labels}")
        print("************************************")

    def print_to_log(self):
        logging.info("********* {} *********".format(self.name))
        logging.info("******* system profile *******")
        for k, v in self.sys_profile.items():
            logging.info("{}: {}".format(k, v))
        logging.info("******* user profile *******")
        for k, v in self.usr_profile.items():
            logging.info("{}: {}".format(k, v))
        logging.info("******* sent profile *******")
        for k, v in self.sent_profile.items():
            logging.info("{}: {}".format(k, v))
        logging.info("******** act sent pair *************")
        # pdb.set_trace()
        for turn in self.act_sent_pair:
            for turn_i in turn:
                logging.info(f"{turn_i[0]}: {turn_i[1]}")
        logging.info("************ last sent *******************")
        logging.info(f"last sents: {self.last_sents}")
        logging.info(f"last labels: {self.last_labels}")
        logging.info("************************************")

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
        self.act_sent_pair = []

    def update(self, to_update_dic_usr, to_update_dic_sys, usr_texts, usr_labels):
        # update using the user inputs    
        if SystemAct.PROVIDE_ORG_FACTS in self.last_labels:
            # as long as the system talks about the org facts, then the user has heard of it.
            self.usr_profile[self.domain.HEARD_OF_THE_ORG] = self.domain.YES

        # self.last_sents = last_sys_sents
        # self.last_labels = last_sys_labels

        self.usr_profile.update(to_update_dic_usr)
        self.sys_profile.update(to_update_dic_sys)

        usr_text = " ".join(usr_texts)
        for last_sys_label in self.last_labels:
            if last_sys_label in self.sent_profile:
                self.sent_profile[last_sys_label].append(usr_text)
            else:
                self.sent_profile[last_sys_label] = [usr_text]

        # pdb.set_trace()    
        self.act_sent_pair.append(list(zip(usr_labels, usr_texts)))
    
    def check_conflict(self, sys_texts, sys_labels, is_repetition_with_sys):
        # check system candidates' confict with the system profiles
        # e.g. user: i want to donate a dollar
        #      sys: how much will you donate?
        def check_conflict_for_one_utt(sys_text, sys_label):
            is_repetition_with_user, repetition_ratio = is_repetition_with_context(sys_text, 
                                                                        itertools.chain(*self.sent_profile.values()), 
                                                                        threshold=cfg.repetition_threshold)

            if "inquiry" in sys_label and "to-inquiry" not in sys_label:
                # e.g. sys: have you heard of save the children?
                #      usr: i have heard of it
                #      sys: do you know about save the children?
                if self.is_inquiry_answered(sys_text, sys_label, is_repetition_with_sys):
                    # 1.1 real repetition, 
                    # this is repetition inquiry
                    if cfg.verbose:
                        print("{} inquiry encountered in user_profile check! {}: {}\n".format(cfg.REPETITION, sys_label, sys_text))
                    return cfg.REPETITION, repetition_ratio
                else:
                    # 1.2 fake repetition,
                    # where the user never replies the inquiry
                    return cfg.PASS, repetition_ratio
            elif is_repetition_with_user:
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
  

    def is_inquiry_answered(self, sys_text, sys_label, is_repetition_inquiry):
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
            if False:
                if sys_label == SystemAct.organization_related_inquiry:
                    if self.usr_profile[act_to_attributes[SystemAct.organization_related_inquiry]] in [self.domain.YES, self.domain.NO]: ### change here
                        return True
                    else:
                        return False
                elif sys_label == SystemAct.propose_donation_inquiry:
                    if self.usr_profile[act_to_attributes[SystemAct.propose_donation_inquiry]] in [self.domain.YES]:
                        return True
                    else: 
                        donation_amount = self.usr_profile[self.domain.DONATION_AMOUNT]
                        try:
                            donation_amount = float(donation_amount)
                            if donation_amount > 0:
                                return True
                            else:
                                return False
                        except:
                            return False
            else:
                if sys_label == SystemAct.propose_donation_inquiry:
                    if "how much" in sys_text.lower():
                        donation_amount = self.usr_profile[self.domain.DONATION_AMOUNT]
                        try:
                            donation_amount = float(donation_amount)
                            if donation_amount > 0:
                                return True
                            else:
                                return False
                        except:
                            any_provide_amount = any(["provide-donation-amount" in pair[0][0] for pair in self.act_sent_pair]) 
                            if any_provide_amount:
                                return True
                            else:
                                return False

                    else:
                        if self.usr_profile[act_to_attributes[SystemAct.propose_donation_inquiry]] in [self.domain.YES]:
                            return True
                        else: 
                            donation_amount = self.usr_profile[self.domain.DONATION_AMOUNT]
                            try:
                                donation_amount = float(donation_amount)
                                if donation_amount > 0:
                                    return True
                                else:
                                    return False
                            except:
                                return False
                elif self.usr_profile[act_to_attributes[sys_label]] != self.domain.INIT: ### change here
                    return True
                else:
                    return False
        else:        
            if sys_label not in self.sent_profile:
                return False
            usr_responses_to_inqury = self.sent_profile[sys_label] #todo, temporarily, should use all user context to find if the answer is already there
                                                            # should be self.is_qa_pair(sys_text, sys_label, self.profile.values())
            return self.is_qa_pair(sys_text, sys_label, usr_responses_to_inqury, is_repetition_inquiry)

    def is_qa_pair(self, sys_text, sys_label, usr_response_list, is_repetition_inquiry):
        # if we cannot extract the answer to the sys_text from the usr_response_list, then it's not answered.
        #===QA=== improvements comes in 
        # if sys_label in [SystemAct.kids_related_inquiry, 
        #                  SystemAct.donation_related_inquiry, 
        #                  SystemAct.organization_related_inquiry, 
        #                  SystemAct.propose_donation_inquiry]:
        #     for usr_response in usr_response_list:
        #         if ("yes" in usr_response) or ("no" in usr_response) or ("not" in usr_response):
        #             return True
        #     return False
        # if sys_label in [SystemAct.other_inquiry] and is_repetition_inquiry:
        #     return True
        # elif sys_label in [SystemAct.other_inquiry] and (not is_repetition_inquiry):
        #     return False
        # else:
        #     return True
        try:
            assert sys_label in [SystemAct.other_inquiry]
        except:
            logging.info(f"assert {sys_label} in [SystemAct.other_inquiry]")
            # pdb.set_trace()
        if is_repetition_inquiry:
            # inquiries with similar wording have been asked
            return True
        else:
            return False

    def syncup(self, profile_to_sync, last_sys_labels, last_sys_sents, whose=None):
        if whose is None:
            whose = self.domain.SYS
        if whose == self.domain.USR:
            for k, v in profile_to_sync.items():
                self.usr_profile[k] = v
        else:
            for k, v in profile_to_sync.items():
                self.sys_profile[k] = v

        self.last_labels = last_sys_labels
        self.last_sents = last_sys_sents
    

class SysWorld(IndividualWorld):
    def __init__(self, domain, name="user_world"):
        super().__init__(domain, name)
        self.sys_profile[self.domain.HEARD_OF_THE_ORG] = self.domain.YES
        self.sys_profile[self.domain.WANT_TO_DONATE] = self.domain.YES

    def update(self, to_update_dic_usr, to_update_dic_sys, sys_texts, sys_labels):
        # update using the system inputs
        # label = self.regex_label(sys_text, context, turn_i)
        # self.last_sents = last_usr_sents
        # self.last_labels = last_usr_labels

        self.usr_profile.update(to_update_dic_usr)
        self.sys_profile.update(to_update_dic_sys)

        for sys_text, sys_label in zip(sys_texts, sys_labels):
            # pdb.set_trace()
            if sys_label in self.sent_profile:
                self.sent_profile[sys_label].append(sys_text)
            else:
                self.sent_profile[sys_label] = [sys_text]
        self.act_sent_pair.append(list(zip(sys_labels, sys_texts)))

    def check_conflict(self, sys_texts, sys_labels):
        # check system candidates' confict with the user profiles
        # label = self.regex_label(sys_text, context, turn_i)
        def check_conflict_for_one_utt(sys_text, sys_label):
            is_repetition, repetition_ratio = is_repetition_with_context(sys_text, 
                                                                        itertools.chain(*self.sent_profile.values()), 
                                                                        threshold=cfg.repetition_threshold)

            
            if True: #sys_label in self.sent_profile:
                if "inquiry" in sys_label and "to-inquiry" not in sys_label:
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
                        if self.is_qa_pair(sys_text=sys_text, sys_label=sys_label):
                            # elif (usr_label, sys_label) in cfg.QA_PAIR_WITH_UNIQUE_ANSWER_DB: 
                            #     # case 2: potentially be fake repetitions 
                            #     # (when user asks, system repeats similar answers)
                            #     # therefore, temporarily solve by 
                            #     # 1) external database support, or 
                            #     # 2) if sample 20 candidates, all the same, then that probably means there is only one best answer to the question
                            return cfg.PASS, repetition_ratio
                        # 2.2 real repetition
                        else:
                            if cfg.verbose:
                                print("{} encountered in sys_check statements! don't form qa_pair {}: {}".format(cfg.REPETITION, sys_label, sys_text))
                                logging.info("{} encountered in sys_check statements! don't form qa_pair {}: {}".format(cfg.REPETITION, sys_label, sys_text))
                            # pdb.set_trace()
                            return cfg.REPETITION, repetition_ratio

                    else:
                        if self.answer_out_of_nowhere(sys_text=sys_text, sys_label=sys_label):
                            # pdb.set_trace()
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
        last_user_label_is_inquiry = any(["inquiry" in l and ("to-inquiry" not in sys_label) for l in self.last_labels])
        if last_user_label_is_inquiry:
            if UserAct.donation_procedure_inquiry in self.last_labels:
                if SystemAct.PROVIDE_DONATION_PROCEDURE in sys_label:
                    return True
                else:
                    return False
            elif UserAct.greeting_inquiry in self.last_labels:
                if SystemAct.greeting_answer in sys_label:
                    return True
                else:
                    return False
            else:
                return False
            # organization_info_inquiry = "organization-info-inquiry"
            # donation_procedure_inquiry = "donation-procedure-inquiry"
            # persuader_intention_inquiry = "persuader-intention-inquiry"
            # kids_related_inquiry = SharedDialogAct.kids_related_inquiry
            # donation_related_inquiry = SharedDialogAct.donation_related_inquiry    
            # other_inquiry = SharedDialogAct.other_inquiry    
            # greeting_inquiry = SharedDialogAct.greeting_inquiry
            # greeting_answer = SharedDialogAct.greeting_answer
        else:
            return False

    def answer_out_of_nowhere(self, sys_text, sys_label):
        """
        True: if the system says some answer out of no where
        False: the system didn't say anything irrelavent answers
        """
        if SystemAct.greeting_answer in sys_label:
            if self.last_labels and UserAct.greeting_inquiry in self.last_labels:
                return False 
            else:
                return True
        else:
            return False


    def refresh(self):
        super().refresh()
        self.sys_profile[self.domain.HEARD_OF_THE_ORG] = self.domain.YES
        self.sys_profile[self.domain.WANT_TO_DONATE] = self.domain.YES
        
    def syncup(self, profile_to_sync, last_usr_labels, last_usr_sents, whose=None):
        if whose is None:
            whose = self.domain.USR

        if whose == self.domain.USR:
            for k, v in profile_to_sync.items():
                self.usr_profile[k] = v
        else:
            for k, v in profile_to_sync.items():
                self.sys_profile[k] = v

        self.last_labels = last_usr_labels
        self.last_sents = last_usr_sents
    # def correct_user(self, ):


