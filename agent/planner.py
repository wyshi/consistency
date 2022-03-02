"""
Strategy Planner
"""
import sys
sys.path.insert(1, '../')
import torch
from persuasion_config import PRE_STRATEGY_ORDER, STRATEGY_ORDER, POST_STRATEGY_ORDER, ACT_TO_STRATEGY_DICT
import random
import copy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from multimodal_transformers.model import AutoModelWithTabular, TabularConfig, DistilBertWithTabular
import pickle as pkl
import numpy as np
import utils
import nltk
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_pkl(dir):
    with open(dir, "rb") as fh:
        obj = pkl.load(fh)
    return obj

class StrategyPlanner(object):
    # base generator
    def __init__(self, max_cycle=1):
        self.pre_strategy_dialogact_order = PRE_STRATEGY_ORDER
        self.post_strategy_dialogact_order = POST_STRATEGY_ORDER
        self.strategy_order = copy.deepcopy(STRATEGY_ORDER)
        self.max_cycle = max_cycle
        #TODO 
        assert max_cycle == 1, f"max_cycle only support 1 for now"

    def is_pre_strategy(self, history):
        return len(history.sys_act) < len(self.pre_strategy_dialogact_order)

    def start_post_strategy(self, history):
        """
        condition to stop the persuasion, and start the post strategy process
        """
        if len(history.sys_act) < (len(self.pre_strategy_dialogact_order) + self.max_cycle * len(self.strategy_order)):
            # pre-strategy finished, and max cycle of strategies
            return False
        else:
            return True

    def pre_strategy_plan(self, history):
        return [self.pre_strategy_dialogact_order[len(history.sys_act)]]
    
    def post_strategy_plan(self, history):
        left_acts = [act for act in history.sys_act if (act not in self.pre_strategy_dialogact_order) and (act not in self.strategy_order) and (act != ACT_TO_STRATEGY_DICT['other'])]
        return [self.post_strategy_dialogact_order[len(left_acts)]]

    def during_strategy_plan(self, history):
        raise NotImplementedError

    def plan(self, history):
        if self.is_pre_strategy(history):
            # pre-strategy
            return self.pre_strategy_plan(history)
        elif self.start_post_strategy(history):
            # post strategy
            return self.post_strategy_plan(history)
        else:
            return self.during_strategy_plan(history)

class RandomPlanner(StrategyPlanner):
    """
    random strategy baseline
    """
    def __init__(self, max_cycle=1, device="cuda:0"):
        super().__init__(max_cycle=max_cycle)

    def during_strategy_plan(self, history):
        # during strategy
        left_strategy = [strategy for strategy in self.strategy_order if strategy not in history.sys_act]
        strategy_this_turn = random.choice(left_strategy)
        return [strategy_this_turn]


class OptimalPlanner(StrategyPlanner):
    """
    optimal order based on previous study
    """
    def __init__(self, max_cycle=1, device="cuda:0"):
        super().__init__(max_cycle=max_cycle)
        

    def during_strategy_plan(self, history):
        # during strategy
        last_strategy = history.sys_act[-1]
        if last_strategy in self.strategy_order:
            next_strategy_id = self.strategy_order.index(last_strategy) + 1
        else:
            next_strategy_id = 0
        strategy_this_turn = self.strategy_order[next_strategy_id]
        return [strategy_this_turn]

class RulePlanner(StrategyPlanner):
    """
    optimal order based on previous study
    """    
    def __init__(self, max_cycle=1, device="cuda:0"):        
        super().__init__(max_cycle=max_cycle)
        self.next_sys_strategy_index = 0
        self.post_strategies_index = 0
        self.post_strategies = False
    
    def reset_counters(self):
        self.next_sys_strategy_index = 0
        self.post_strategies_index = 0
        self.post_strategies = False

    def map_strategies(self, history):
        usr_acts = [ACT_TO_STRATEGY_DICT[x.replace('_', '-')].replace('_', '-') for x in history.usr_act[history.start_of_most_recent_usr_acts:]]
        planned_acts = []
        if history.usr_utt:
            sentences = nltk.sent_tokenize(history.usr_utt[-1])
        # for act in usr_acts:
        for i in range(len(usr_acts)):
            act = usr_acts[i]
            sent = sentences[i]
            if act == 'acknowledgement':
                continue
            elif act not in ['personal-related-inquiry', 'greeting', 'ask-org-info', 'task-related-inquiry', 'positive-to-inquiry', 'neutral-to-inquiry', 'negative-to-inquiry', 'positive-reaction-to-donation', 'agree-donation',  'thank', 'negative-reaction-to-donation', 'disagree-donation', 'disagree-donation-more', 'neutral-to-donation', 'provide-donation-amount', 'ask-donation-procedure']:
                continue
            else:
                # if act in ['personal-related-inquiry']:#, 'task-related-inquiry']:
                    # planned_acts.append(ACT_TO_STRATEGY_DICT['inquiry-response'])
                if act in ['task-related-inquiry', 'personal-related-inquiry'] or utils.is_factual(sent) or utils.is_opinion(sent):
                    planned_acts.append((sent, 'DBCALL'))                
                    #pos to inquiry, neg to inquiry, neu to inquiry, personal stoery, logical appeal
                elif act in ['ask-org-info']:
                    planned_acts.append((sent, ACT_TO_STRATEGY_DICT['credibility-appeal']))
                elif act in ['greeting']:
                    planned_acts.append((sent, ACT_TO_STRATEGY_DICT['greeting']))
                elif act in ['positive-reaction-to-donation', 'positive-to-inquiry', 'neutral-to-inquiry',  'negative-to-inquiry']:
                    planned_acts.append((sent, ACT_TO_STRATEGY_DICT['comment-partner']))
                    # continue
                elif act in ['agree-donation']:
                    planned_acts.append((sent, ACT_TO_STRATEGY_DICT['thank']))
                elif act in ['thank']:
                    planned_acts.append((sent, ACT_TO_STRATEGY_DICT['you-are-welcome']))
                elif act in ['negative-reaction-to-donation', 'disagree-donation', 'disagree-donation-more']:
                    planned_acts.append((sent, ACT_TO_STRATEGY_DICT['ask-not-donate-reason']))
                elif act in ['neutral-to-donation']:
                    planned_acts.append((sent, ACT_TO_STRATEGY_DICT['comment-partner']))
                elif act in ['provide-donation-amount']:
                    planned_acts.append((sent, ACT_TO_STRATEGY_DICT['praise-user']))
                elif act in ['ask-donation-procedure']:
                    planned_acts.append((sent, ACT_TO_STRATEGY_DICT['donation-information']))
                else:
                    continue                
        return planned_acts
    def start_post_strategy(self, history):
        """
        condition to stop the persuasion, and start the post strategy process
        """
        return history.post_strategies
        # if len(history.sys_act) - self.inserted_strategy_offset + self.skipped_strategies < (len(self.pre_strategy_dialogact_order) + self.max_cycle * len(self.strategy_order)):
        #     # pre-strategy finished, and max cycle of strategies
        #     return False
        # else:
        #     return True
    def pre_strategy_plan(self, history):
        planned_acts = self.map_strategies(history)
        pre_strategy_act = self.pre_strategy_dialogact_order[len(history.sys_act)]
        acts = [x[1] for x in planned_acts]
        if pre_strategy_act not in acts:
            if history.usr_utt:
                planned_acts.append((history.usr_utt[-1], pre_strategy_act))
            else:
                planned_acts.append(('', pre_strategy_act))        
        return planned_acts
    
    def post_strategy_plan(self, history):
        planned_acts = self.map_strategies(history)
        # left_acts = [act for act in history.sys_act if (act not in self.pre_strategy_dialogact_order) and (act not in self.strategy_order) and (act != ACT_TO_STRATEGY_DICT['other'])]
        if history.post_strategies_index < len(self.post_strategy_dialogact_order):
            post_strategy_act = self.post_strategy_dialogact_order[history.post_strategies_index]
            history.post_strategies_index  += 1
            acts = [x[1] for x in planned_acts]
            if post_strategy_act not in acts:
                if history.usr_utt:
                    planned_acts.append((history.usr_utt[-1], post_strategy_act))
                else:
                    planned_acts.append(('', post_strategy_act))            
        else:
            planned_acts.append(("", POST_STRATEGY_ORDER[-1]))
            self.reset_counters()
        return planned_acts

    def during_strategy_plan(self, history):
        # during strategy
        planned_acts = self.map_strategies(history)

        #Substitute optimal order for supervised planner here, maybe?
        # last_strategy = history.sys_act[-1]
        last_strategy = history.next_sys_strategy_index        
        # if last_strategy in self.strategy_order:
        #     next_strategy_id = self.strategy_order.index(last_strategy) + 1
        # else:
        #     next_strategy_id = 0
        strategy_this_turn = self.strategy_order[last_strategy]
        acts = [x[1] for x in planned_acts]
        if strategy_this_turn not in acts:
            if history.usr_utt:
                planned_acts.append((history.usr_utt[-1], strategy_this_turn))
            else:
                planned_acts.append(('', strategy_this_turn))
        history.next_sys_strategy_index += 1
        if history.next_sys_strategy_index == len(self.strategy_order):
            history.post_strategies =  True    
        return planned_acts

class SupervisedlearningPlanner(StrategyPlanner):
    """
    supervised-learning based 
    """
    def __init__(self, model_path, max_cycle=1, device="cuda:0"):
        super().__init__(max_cycle=max_cycle)
        self.device=device
        if model_path[-1] == '/':
          self.model_path = model_path
        else:
          self.model_path = model_path + '/'
        self.load_model(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True, padding = True, model_max_length=512)

    def load_model(self, model_path):
        self.model = DistilBertWithTabular.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.label_encoder = load_pkl(model_path + '_labelencoder.pkl')
        self.user_encoder = load_pkl(model_path + '_user_act_encoder.pkl')
        self.sys_encoder = load_pkl(model_path + '_sys_act_encoder.pkl')
    
    def start_post_strategy(self, history):
        """
        condition to stop the persuasion, and start the post strategy process
        """
        num_other = 0
        for act in history.sys_act:
            if act == ACT_TO_STRATEGY_DICT['other']:
                num_other += 1
        #We don't want to count "other" as a strategy which would occupy part of the strategy cycle.
        if len(history.sys_act) - num_other < (len(self.pre_strategy_dialogact_order) + self.max_cycle * len(self.strategy_order)):
            # pre-strategy finished, and max cycle of strategies
            return False
        else:
            return True
                
    def during_strategy_plan(self, history):
        model_input = []
        user_idx = 0
        sys_idx = 0
        sys_act_idx = 0
        usr_act_idx = 0
        while user_idx < len(history.usr_utt) and sys_idx < len(history.sys_utt) and usr_act_idx < len(history.usr_act) and sys_act_idx < len(history.sys_act):
            model_input.append(history.sys_act[sys_act_idx])
            model_input.append(history.sys_utt[sys_idx])
            model_input.append(history.usr_act[usr_act_idx])
            model_input.append(history.usr_utt[user_idx])
            sys_idx += 1
            user_idx += 1
            sys_act_idx += 1
            usr_act_idx += 1
        while sys_idx < len(history.sys_utt):
            model_input.append(history.sys_utt[sys_idx])
            sys_idx += 1
        while sys_act_idx < len(history.sys_act):
            model_input.append(history.sys_act[sys_act_idx])
            sys_act_idx += 1
        while user_idx < len(history.usr_utt):
            model_input.append(history.usr_utt[user_idx])
            user_idx += 1
        while usr_act_idx < len(history.usr_act):
            model_input.append(history.usr_act[usr_act_idx])
            usr_act_idx += 1
        model_input = self.tokenizer(' '.join(model_input), truncation=True, return_tensors='pt') #self.tokenizer(' '.join(model_input), truncation=True, padding = True, return_tensors='pt')
        model_input = model_input.to(self.device)        
        if len(history.usr_act) >= 2:
            user_acts = history.usr_act[-2:]
        elif len(history.usr_act) == 1:
            user_acts = history.usr_act[0]
        else:
            user_acts = []
        if len(history.sys_act) >= 2:
            sys_acts = history.sys_act[-2:]
        elif len(history.sys_act) == 1:
            sys_acts = history.sys_act[0]
        else:
            sys_acts = []
        user_vector = np.zeros(self.user_encoder.classes_.shape[0])
        for act in user_acts:
            user_vector[self.user_encoder.transform([act.replace('_', '-').replace('provide-org-facts', 'credibility-appeal')])] += 1
        sys_vector = np.zeros(self.sys_encoder.classes_.shape[0])
        for act in sys_acts:
            sys_vector[self.sys_encoder.transform([act.replace('_', '-').replace('provide-org-facts', 'credibility-appeal')])] += 1
        numerical_feats = torch.cuda.DoubleTensor(np.array([np.concatenate((user_vector, sys_vector))])).to(self.device)
        # numerical_feats = numerical_feats.to(device)
        # print(numerical_feats)        
        out = self.model(input_ids=model_input.input_ids.int(), numerical_feats=numerical_feats.float())
        logits = out[1]
        prediction = logits.argmax(-1)
        return [ACT_TO_STRATEGY_DICT[self.label_encoder.inverse_transform(prediction.cpu())[0]]]
        #join together the utterances and dialog acts in the right order.
        # for i in range(len(history))

def main():
    planner = SupervisedlearningPlanner('../strategy_planner/classification_best_model_dialog')
    from dialog_history import DialogHistory
    history = DialogHistory(sys_first=True)
    history.update_sys_history('Hello. How are you?', 'greeting')
    history.update_usr_history("I'm good, how are you doing?", 'greeting')
    history.update_sys_history("Very well. I'm just up organizing info for my charity. Are you involved with charities?", 'greeting task-related-inquiry other')
    history.update_usr_history("Yes! I work with children who have terminal illnesses. What charity are you involved in?", 'positive-to-inquiry task-related-inquiry')
    history.update_sys_history("That's great! I help with Save The Children.", 'acknowledgement neutral-to-inquiry')
    history.update_usr_history("Amazing! Working with kids is the best. What do you do for Save the Children?", 'acknowledgement positive-reaction-to-donation task-related-inquiry')
    history.update_sys_history("I help raise donations and volunteer time.", 'neutral-to-inquiry')
    history.update_usr_history("That's so important. How do you raise donations?", 'acknowledgement task-related-inquiry')
    history.update_sys_history("By directly asking for aid. Do you currently donate to your charity?", 'task-related-inquiry neutral-to-inquiry')
    history.update_usr_history("Yes I do, but I'm happy to donate to yours as well!", 'agree-donation')
    
    print(planner.during_strategy_plan(history)[0])

if __name__ == "__main__":
    main()
    
    
