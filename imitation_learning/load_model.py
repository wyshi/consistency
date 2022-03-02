import torch
from nltk.tokenize import sent_tokenize
import os
os.environ["NCCL_P2P_LEVEL"]="2"
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from imitation_learning.transformers_old_for_il.tokenization_gpt2 import GPT2Tokenizer
from imitation_learning.transformers_old_for_il.configuration_gpt2 import GPT2Config

from imitation_learning.model_clf import load_model as load_model_in_MODEL_CLF, SequenceSummary, load_pkl, build_model_classifier
import imitation_learning.config as cfg
from imitation_learning.KnowledgeBase.KB import Domain
from imitation_learning.AgentProfile.core import SystemAct
from imitation_learning.AgentProfile.profiles_in_dev import GlobalProfile
from persuasion_config import ACT_TO_STRATEGY_DICT
import utils
from agent.dialog_history import DialogHistory

class ModelClassifierConfig:
    model_dir = cfg.model_clf_dir
    device1 = torch.device(cfg.model_clf_device1)
    device2 = torch.device(cfg.model_clf_device2)

class CurrentModelConfig:
    with_rule = True
    log_file = 'log/'
    strategy_selection_on = False#False
    
    with_baseline =  True
    with_repetition_module = True
    with_consistency_module = True
    with_sentence_clf = True
    with_RL_finetune_model = True

    candidate_select_strategy = cfg.IMITATION_LEARNING_SELECTION
    NUM_CANDIDATES = 10


def sent_tokenize_modified(sent):
    sents = sent_tokenize(sent)
    if sents == []:
        sents = ['']
    return sents


def load_model_clf_for_AMT(model_clf_dir, device1, device2):
    config = GPT2Config()
    config = config.from_pretrained('gpt2')#config.from_pretrained('gpt2-medium')
    config.summary_first_dropout = 0.2
    config.summary_type = "cls_index"
#    le_A = load_pkl("imitation_learning/data/labelencoder_A.pkl")
#    le_A = load_pkl("../imitation_learning/data/labelencoder_A.pkl")
    # le_A = load_pkl(os.path.join(os.getcwd(), "imitation_learning/data/labelencoder_A.pkl"))
    try:
        le_A = load_pkl(os.path.join(os.getcwd(), "imitation_learning/data/labelencoder_A.pkl"))
    except:
        le_A = load_pkl("../imitation_learning/data/labelencoder_A.pkl")
#    le_B = load_pkl("imitation_learning/data/labelencoder_B.pkl")
#    le_B = load_pkl("../imitation_learning/data/labelencoder_B.pkl")
    # le_B = load_pkl(os.path.join(os.getcwd(), "imitation_learning/data/labelencoder_B.pkl"))
    try:
        le_B = load_pkl(os.path.join(os.getcwd(), "imitation_learning/data/labelencoder_B.pkl"))
    except:
        le_B = load_pkl("../imitation_learning/data/labelencoder_B.pkl")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")#torch.load(tokenizer_dir)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    model_A, model_B = load_model_in_MODEL_CLF(cfg, "small", tokenizer, device1, device2)
    clf_A = SequenceSummary(num_labels=le_A.classes_.shape[0], config=config)
    clf_B = SequenceSummary(num_labels=le_B.classes_.shape[0], config=config)
    clf_TF = SequenceSummary(num_labels=2, config=config)
    
    model_A_state, model_B_state, clf_A_state, clf_B_state, clf_TF_state = torch.load(model_clf_dir, map_location=device1)
    model_A.load_state_dict(model_A_state)
    model_B.load_state_dict(model_B_state)
    clf_A.load_state_dict(clf_A_state)
    clf_B.load_state_dict(clf_B_state)
    clf_TF.load_state_dict(clf_TF_state)
    print(f"all models loaded")

    return tokenizer, model_A, model_B, clf_A, clf_B, clf_TF


class ImitationClassifier(object):
    def __init__(self, MODELS_USED_IN_MODEL_CLF) -> None:
        self.model_clf = build_model_classifier(ModelClassifierConfig.model_dir, 
                                                ModelClassifierConfig.device1, 
                                                ModelClassifierConfig.device2,
                                                models_used_in_model_clf=MODELS_USED_IN_MODEL_CLF)
        self.domain = Domain(cfg.domain)
        self.model_config = CurrentModelConfig
        self.global_profile = GlobalProfile(domain=self.domain, model_config=self.model_config, sent_embedding_model=None)


    def predict_dialog_act(self, context, sent, role):
        # role in ['A', 'B']
        # predic the dialog act for the sent
        sents = sent_tokenize_modified(sent)
        # input_texts_labels = [None]*len(input_texts)
        sents_labels, _ = self.global_profile.regex_label(model_clf=self.model_clf,
                                                          sys_texts=sents, 
                                                          which_task=role,
                                                          context=context, 
                                                          past=None)
        sents_labels = [ACT_TO_STRATEGY_DICT[label] if label in ACT_TO_STRATEGY_DICT else label for label in sents_labels]
        return sents_labels

    def predict_TF(self, context, next_response_candidate):
        '''if the candidate should be selected'''
        clf_predicted_acts, clf_past = self.model_clf.predict(separate_sents=next_response_candidate, 
                                                              which_task="TF",
                                                              context=context)
        if clf_predicted_acts == 1:
            return 1
        else:
            return 0

if __name__ == "__main__":
    MODELS_USED_IN_MODEL_CLF = load_model_clf_for_AMT(ModelClassifierConfig.model_dir, 
                                                    ModelClassifierConfig.device1, 
                                                    ModelClassifierConfig.device2)

    clf = ImitationClassifier(MODELS_USED_IN_MODEL_CLF)

    history = DialogHistory(sys_first=True)
    clf.predict_TF(context=utils.reconstruct_history(history), next_response_candidate=["have you heard of Save the Children?"])
    clf.predict_dialog_act(context=utils.reconstruct_history(history), sent="have you donated before?", role="A")
