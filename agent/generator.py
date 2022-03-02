import sys
sys.path.insert(1, '../')
sys.path.insert(1, 'agent/')
from persuasion_config import SYS_TEMPLATE_ACT_DIC, STRATEGY_TO_ACT_DICT, ACT_TO_STRATEGY_DICT
import random
from transformers import AutoModelForSeq2SeqLM, AutoConfig, AutoTokenizer
# from datasets import Dataset
import torch
import generation_config
from sentence_transformers import SentenceTransformer
import nltk
from scipy.spatial.distance import cosine
import itertools
from utils import robust_question_regex, reconstruct_history
import re
import json
import pickle
import time
import logging
import truecase 
import re


class Generator(object):
    # base generator
    def __init__(self):
        pass

    def generate(dialog_act):
        raise NotImplementedError

# temporary template-based generator

def str_contains_question(sentence):
    hasquestion = False
    for regex in robust_question_regex.keys():
        sentence = sentence.lower().lstrip().rstrip()
        match = re.match(robust_question_regex[regex], sentence)
    # print(match)
        if "?" in sentence or bool(match):
            hasquestion = True
    return hasquestion


class TemplateBasedGenerator(Generator):
    def __init__(self):
        super().__init__()
        self.method = "template-based"

    def generate(self, history, dialog_act_list):
        utts = [random.choice(SYS_TEMPLATE_ACT_DIC[dialog_act])
                for dialog_act in dialog_act_list]
        return " ".join(utts)

def postprocessing(potential_sentence):
    '''
    Takes a candidate sentence and does any rule-based filtering before delpoyment
    '''
    potential_sentence = potential_sentence.lower()
    about_turk = ('reward' in potential_sentence.lower()) or ('task' in potential_sentence.lower()) or ('payment' in potential_sentence.lower()) or ('earnings' in potential_sentence.lower())
    if about_turk and ('i am' in potential_sentence.lower() or "i'm" in potential_sentence.lower() or "i will" in potential_sentence.lower() or 'my' in potential_sentence.lower()) :
        return "I am going to donate a bit of my money to Save the Children"
    if about_turk and 'you' in potential_sentence.lower():
        return "Would you be interested in donating a bit of money to Save the Chilren?"
    if "i have a child" in potential_sentence.lower():
        return "I think we really need to help these children in warzones."
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", potential_sentence)
    if numbers and ('donating' in potential_sentence.lower() or 'donate' in potential_sentence.lower()):
        for i, num in enumerate(numbers):
            if float(num) < 1900:
                #number is not a year
                potential_sentence.replace(num, '0.50')
    return potential_sentence
    

class ConditionalGenerator(Generator):
    def __init__(self, model_path, device):
        super().__init__()
        self.method = "conditional-based"
        self.model_path = model_path
        self.device = device
        self.load_model(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, truncation=True, padding=True, model_max_length=1024, max_position_embeddings=1024)
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        if generation_config.regenerate_until_dialog_act:
            import imitation_learning.load_model as il
            import config as CONFIG
            loaded_model = il.load_model_clf_for_AMT(
                model_clf_dir=CONFIG.il_clf_dir,
                device1=CONFIG.il_clf_device1,
                device2=CONFIG.il_clf_device2,
            )
            self.il_classifier = il.ImitationClassifier(loaded_model)
        else:
            self.il_classifier = None

    def neural_generation(self, sentence, model, dialog_act, act_string, history, semantic_threshold, utterances_to_keep, utterance_offset=0):
        contains_question = False
        for regex in robust_question_regex.keys():
            usr_utt = sentence.lower().lstrip().rstrip()
            match = re.match(robust_question_regex[regex], usr_utt)
        # print(match)
            if "?" in usr_utt or bool(match):
                contains_question = True
        if contains_question and 'closing' in dialog_act.lower():
            return "Sorry, I have to go now, but it’s been great talking to you! Thanks for your time!"
        model_input = ' '.join(
            history.all_utt[-(utterances_to_keep+utterance_offset):]) + ' '
        if sentence.lstrip().rstrip() not in history.all_utt:    
            model_input = model_input + sentence
        model_input = model_input.lower() + act_string
        model_inputs = self.tokenizer(model_input)        
        print("Input:", model_input)
        if len(model_inputs['input_ids']) > 1024:
            model_inputs['input_ids'] = model_inputs['input_ids'][-1024:]
            model_inputs['attention_mask'] = model_inputs['attention_mask'][-1024:]
        history_embedding = self.sentence_transformer.encode(
            [entire_utterance.split(':') for entire_utterance in history.all_utt[-utterances_to_keep:]])
        done_generating = False
        failed_too_many_times = False
        attempt_counter = 0
        while not done_generating and not failed_too_many_times:
            done_generating = True
            gen = model.generate(torch.Tensor([model_inputs['input_ids']]).int().to(
                self.device), no_repeat_ngram_size=4, encoder_no_repeat_ngram_size=3, top_p=0.99, max_length=100, do_sample=True, num_beams=4)
            gen_sentences = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            need_to_regenerate = False
            if semantic_threshold != 0:
                for sentence in gen_sentences:
                    for i, embedding in enumerate(history_embedding):
                        sentence_embedding = self.sentence_transformer.encode(
                            sentence)
                        dist = cosine(sentence_embedding, embedding)
                        if dist <= semantic_threshold:
                            need_to_regenerate = True

                        # Do the same, but for each pairing of sentences within the generated sentence.
                        sentences = nltk.sent_tokenize(sentence)
                        if len(sentences) > 1:
                            sentence_pairs = list(
                                itertools.combinations(sentences, 2))
                            for (s1, s2) in sentence_pairs:
                                s1_embedding = self.sentence_transformer.encode(
                                    s1)
                                s2_embedding = self.sentence_transformer.encode(
                                    s2)
                                dist = cosine(s1_embedding, s2_embedding)
                                if dist <= semantic_threshold:
                                    need_to_regenerate = True

                if need_to_regenerate:
                    attempt_counter += 1
                    done_generating = False
                    if attempt_counter >= 5:
                        failed_too_many_times = True
        if failed_too_many_times:
            if dialog_act in SYS_TEMPLATE_ACT_DIC.keys():
                return random.choice(SYS_TEMPLATE_ACT_DIC[dialog_act])
            else:
                #If it's not part of the agenda pushing model we'll just ignore it.
                return ""
                
        return truecase.get_true_case(gen_sentences[0])

    def load_model(self, model_path):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model = self.model.to(self.device)

    def generate(self, dialog_act_list, history):
        # act_string = STRATEGY_TO_ACT_DICT[dialog_act]
        # When we want separation like: Utt A, Utt B, Utt C, Act A, Act B, Act C, etc. uncomment below
        if history.usr_utt and generation_config.extra_utterance_for_questions:            
            contains_question = False
            for regex in robust_question_regex.keys():
                # print(history.usr_utt[-1].lower())
                usr_utt = history.usr_utt[-1].lower().split(':')[-1].lstrip().rstrip()
                match = re.match(robust_question_regex[regex], usr_utt)
                # print(match)
                if "?" in usr_utt or bool(match):
                    contains_question = True
            if contains_question:
                #If it is a question, then insert the respond to user token as an extra model input
                dialog_act_list.insert(0, generation_config.respond_to_user_token)
        def generate_for_one_act(dialog_act, added_utterance_offset=0):
            dialog_act = STRATEGY_TO_ACT_DICT[dialog_act]
            if generation_config.dialog_acts_to_keep == 0:
                act_hist = ''
            else:
                # print("here", dialog_act)
                mapped_act_history = [STRATEGY_TO_ACT_DICT[act]
                                      for act in history.sys_act[-generation_config.dialog_acts_to_keep:]]
                # mapped_act_history = history.sys_act[-generation_config.dialog_acts_to_keep:]
                if generation_config.collapse_inquiries:
                    for act_i in range(len(mapped_act_history)):
                        if 'inquiry' in mapped_act_history[act_i]:
                            mapped_act_history[act_i] = 'inquiry'
                    if 'inquiry' in dialog_act:
                        dialog_act = 'inquiry'
                act_hist = generation_config.act_history_tag + ' ' + \
                    ' '.join(mapped_act_history)
            #If we need to just respond to the user, then we don't incorporate a dialog act on this utterance.
            if dialog_act == generation_config.respond_to_user_token:
                act_string = ' ' + act_hist + ' ' + dialog_act
            else:
                act_string = ' ' + act_hist + ' ' + generation_config.planned_act_tag + \
                    ' ' + dialog_act
            try:
                model_input = ' '.join(
                    history.all_utt[-(generation_config.utterances_to_keep+added_utterance_offset):]) + act_string
            except:
                try:
                    model_input = ' '.join(history) + ' ' + act_string
                except:
                    raise Exception("Invalid model input given.")
            model_inputs = self.tokenizer(model_input.lower())
            print("Input:", model_input.lower())
            if len(model_inputs['input_ids']) > 1024:
                model_inputs['input_ids'] = model_inputs['input_ids'][-1024:]
                model_inputs['attention_mask'] = model_inputs['attention_mask'][-1024:]

            history_embedding = self.sentence_transformer.encode(
                history.all_utt[-generation_config.utterances_to_keep:])

            done_generating = False
            while not done_generating:
                done_generating = True
                gen = self.model.generate(torch.Tensor([model_inputs['input_ids']]).int().to(
                    self.device), no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3, top_p=0.99, max_length=100, do_sample=True, num_beams=4)
                gen_sentences = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
                need_to_regenerate = False
                if generation_config.semantic_threshold != 0:
                    for sentence in gen_sentences:
                        for embedding in history_embedding:
                            sentence_embedding = self.sentence_transformer.encode(
                                sentence)
                            dist = cosine(sentence_embedding, embedding)
                            if dist <= generation_config.semantic_threshold:
                                need_to_regenerate = True

                            # Do the same, but for each pairing of sentences within the generated sentence.
                            sentences = nltk.sent_tokenize(sentence)
                            if len(sentences) > 1:
                                sentence_pairs = list(
                                    itertools.combinations(sentences, 2))
                                for (s1, s2) in sentence_pairs:
                                    s1_embedding = self.sentence_transformer.encode(
                                        s1)
                                    s2_embedding = self.sentence_transformer.encode(
                                        s2)
                                    dist = cosine(s1_embedding, s2_embedding)
                                    if dist <= generation_config.semantic_threshold:
                                        need_to_regenerate = True

                    if need_to_regenerate:
                        done_generating = False
            return gen_sentences[0]
        utts = []
        keep_generating = True
        while keep_generating:
            #Add utterances to utts
            utterances_added = 0
            tuple_hist = reconstruct_history(history)
            for dialog_act in dialog_act_list:
                utt = generate_for_one_act(dialog_act, utterances_added)
                if generation_config.regenerate_until_dialog_act:
                    classified_act = self.il_classifier.predict_dialog_act(
                        context=tuple_hist, sent=utt, role='A')
                    for i in range(len(classified_act)):
                        act = classified_act[i]
                        act = act.replace('_', '-')
                        act = STRATEGY_TO_ACT_DICT[ACT_TO_STRATEGY_DICT[act]]
                        classified_act[i] = act
                    trial_utt = utt
                    while len(set(classified_act) & set([dialog_act[1]])) == 0:
                        trial_utt = utt + ' ' + generate_for_one_act(dialog_act)
                        classified_act = self.il_classifier.predict_dialog_act(
                            context=tuple_hist, sent=trial_utt, role='A')
                        for i in range(len(classified_act)):
                            act = classified_act[i]
                            act = act.replace('_', '-')
                            act = STRATEGY_TO_ACT_DICT[ACT_TO_STRATEGY_DICT[act]]
                            classified_act[i] = act
                    utt = trial_utt
                utts.append(utt)
                if not generation_config.require_no_sentences_from_same_turn_in_model_context:
                    history.all_utt.append('A: ' + utt)
                    utterances_added += 1
            if len(utts) == 1:
                keep_generating = False
            else:
                utt_pairs = list(itertools.combinations(utts, 2))
                keep_generating = False
                for (u1, u2) in utt_pairs:
                    u1_embedding = self.sentence_transformer.encode(u1)
                    u2_embedding = self.sentence_transformer.encode(u2)
                    dist = cosine(u1_embedding, u2_embedding)
                    if dist <= generation_config.semantic_threshold:
                        keep_generating = True
                        print("Resetting")
                        utts = []
                        while utterances_added > 0:
                            history.all_utt.pop()
                            utterances_added -= 1
                        break
                    else:
                        continue
        for i, utterance in enumerate(utts):
            utts[i] = truecase.get_true_case(postprocessing(utterance))
            generated_utterance = "\n".join(utts)
            generated_utterance = generated_utterance.replace('URL', 'https://www.savethechildren.org/')
            generated_utterance = generated_utterance.replace('Url', 'https://www.savethechildren.org/')
            generated_utterance = generated_utterance.replace('url', 'https://www.savethechildren.org/')
        return generated_utterance                

class DualConditionalGenerator(ConditionalGenerator):
    def __init__(self, strategy_model_path, device, response_model_path, qa_path, question_representation_path):
        super().__init__(strategy_model_path, device)
        self.is_blenderbot = False
        self.used_blenderbot_already = False #We only want to generate using Blenderbot once per turn, since it's not based on dialog acts.
        if 'blenderbot' in response_model_path and '.json' in response_model_path:
            import json
            from blenderbot2.blenderbot2.blenderbot2 import BlenderBot2FidAgent
            with open(response_model_path) as f:
                opt = json.load(f)            
            start = time.time()
            self.response_model = BlenderBot2FidAgent(opt)
            end = time.time()
            print("Blenderbot2 loaded in {} seconds".format(end - start))
            self.is_blenderbot = True
            try:
                with open('/local-scratch1/data/shared/qywu/ParlAI/data/models/blenderbot2/blenderbot2_3B/model.opt', 'r') as f:
                    tmp = json.load(f)
                    self.DELIMITER = tmp['delimiter']
            except:
                self.DELIMITER = '  '
        else:
            self.load_response_model(response_model_path)            
        
        self.qa = self.load_qa(qa_path)
        self.question_representation = self.load_qa_representation(question_representation_path)
        self.method = "dual-conditional-based"
    def _build_up_model_input(self, history_list):
        assert self.is_blenderbot
        text = self.DELIMITER.join(history_list)# + [user_text])
        text = text.lower()
        return text
    def generate_personal_response(self, sentence, dialog_act, act_string, history, semantic_threshold, utterances_to_keep, utterance_offset=0, failed_tries=0):
        if self.is_blenderbot:
            if not self.used_blenderbot_already and failed_tries < 8:
                self.response_model.reset()
                start = time.time()
                blender = self.response_model.clone()
                print("Blender cloned in {} seconds".format(time.time() - start))
                history_list = [utt.split(':')[-1] for utt in history.all_utt]
                obs = self._build_up_model_input(history_list)
                try:
                    # self.response_model.observe({'text':obs, 'episode_done':True})
                    # agent_reply = self.response_model.act()
                    blender.observe({'text':obs, 'episode_done':True})
                    agent_reply = blender.act()
                    # print(agent_reply)
                    agent_output = agent_reply['text']
                    
                    # agent_output = blender.respond(obs)
                except Exception as e:
                    print("Error: ", e)                
                    print("Failed. Resetting after 0.5 second delay.")
                    time.sleep(0.5)
                    # import pdb
                    # pdb.set_trace()
                    return self.generate_personal_response(sentence, dialog_act, act_string, history, semantic_threshold, utterances_to_keep, utterance_offset, failed_tries + 1)
                
                # self.response_model.observe({"text":"<Start>", "id":"B", "episode_done":False})
                # for utterance in history.all_utt:
                #     speaker = utterance.split(':')[0]
                #     text = utterance.split(':')[-1]
                #     episode_done = False
                #     if speaker == 'A':
                #         self.response_model.self_observe({"text":text, "id":speaker, "episode_done":False})
                #     else:
                #         self.response_model.observe({"text":text, "id":speaker, "episode_done":False})
                # print(self.response_model.observation)
                # try:
                #     agent_reply = self.response_model.batch_act([self.response_model.observation])
                #     self.response_model.self_observe(agent_reply)
                # except:
                #     print("Failed. Resetting.")
                #     return self.generate_personal_response(sentence, dialog_act, act_string, history, semantic_threshold, utterances_to_keep, utterance_offset, failed_tries + 1)
                # agent_output = ' '.join([r.get('text', 'dummy') for r in agent_reply])

                print("Blenderbot says {}".format(agent_output))
                if '_potentially_unsafe_' in agent_output.lower():
                    print("Low confidence output; return nothing")
                    return ''
                else:
                    self.used_blenderbot_already = True
                    sentences = nltk.sent_tokenize(agent_output)
                    candidate = sentences[0]
                    if len(sentences) > 1:
                        print("Only keeping first sentence: {}".format(candidate))
                    if generation_config.skip_blender_questions and str_contains_question(candidate): 
                        print("Asked a question; skipping {}".format(candidate))
                        return ''
                    if generation_config.check_questions_in_domain and str_contains_question(candidate):
                        user_utterance = history.usr_utt[-1]                        
                        rep = self.sentence_transformer.encode(user_utterance)
                        CLOSEST_DIST = 10000
                        CLOSEST = None
                        for q in self.qa.keys():
                            q_rep = self.question_representation[q]
                            dist = cosine(rep, q_rep)
                            CLOSEST_DIST = min(dist, CLOSEST_DIST)
                            CLOSEST = q if CLOSEST_DIST == dist else CLOSEST
                        if CLOSEST and CLOSEST_DIST < 0.65:
                            #Possibly explore a minmum jaccard threshold, so AND (CLOSEST_DIST < X OR JACCARD_SIM > Y)
                            print("There is an in-domain question similar question:", CLOSEST)
                            return candidate
                        else:
                            print("Question out of domain. {}".format(candidate))
                            return ''
                    
                    return candidate #agent_output
            else:
                #We already generated an utterance using blenderbot this turn, or we have failed too many times.
                print("Number of failed tries: {}".format(failed_tries))
                return ''
        else:
            return self.neural_generation(sentence, self.model, dialog_act, act_string, history, generation_config.semantic_threshold, generation_config.dual_model_utterances_to_keep, utterance_offset)#(self.response_model, dialog_act, act_string, history, semantic_threshold, utterances_to_keep, utterance_offset)

    def load_response_model(self, path):
        self.response_model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self.response_model = self.response_model.to(self.device)
    
    def load_qa(self, path):
        with open(path, 'r') as f:
            return json.load(f)
    
    def load_qa_representation(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


    def generate(self, dialog_act_list, history):
        failed_too_many_times = False
        attempt_counter = 0
        # act_string = STRATEGY_TO_ACT_DICT[dialog_act]
        # When we want separation like: Utt A, Utt B, Utt C, Act A, Act B, Act C, etc. uncomment below
        
        def generate_for_one_act(dialog_tuple, added_utterance_offset=0):
            sentence = dialog_tuple[0]
            dialog_act = dialog_tuple[1]
            mapped_act_history = [STRATEGY_TO_ACT_DICT[act[1]] if act[1] != 'DBCALL' else act[1]
                                    for act in history.sys_act[-generation_config.dual_model_dialog_acts_to_keep:]]
            if generation_config.collapse_inquiries:
                for act_i in range(len(mapped_act_history)):
                    if 'inquiry' in mapped_act_history[act_i]:
                        mapped_act_history[act_i] = 'inquiry'
                if 'inquiry' in dialog_act:
                    dialog_act = 'inquiry'
            act_hist = generation_config.act_history_tag + ' ' + \
                ' '.join(mapped_act_history)            
            act_string = ' ' + act_hist + ' ' + generation_config.planned_act_tag + \
                ' ' + dialog_act
            if generation_config.dialog_acts_to_keep == 0:
                act_hist = ''
            elif dialog_act == 'DBCALL':
                user_utterance = history.usr_utt[-1]
                if generation_config.end_to_end_question_model:
                    gen = self.model.generate(torch.Tensor([self.tokenizer(user_utterance)['input_ids']]).int().to(
                        self.device), no_repeat_ngram_size=3, encoder_no_repeat_ngram_size=3, top_p=0.99, max_length=100, do_sample=True, num_beams=4)
                    gen_sentences = ' '.join(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
                    return gen_sentences
                rep = self.sentence_transformer.encode(user_utterance)
                CLOSEST_DIST = 10000
                CLOSEST = None
                for q in self.qa.keys():
                    q_rep = self.question_representation[q]
                    dist = cosine(rep, q_rep)
                    CLOSEST_DIST = min(dist, CLOSEST_DIST)
                    CLOSEST = q if CLOSEST_DIST == dist else CLOSEST
                if CLOSEST and CLOSEST_DIST < 0.6:
                    #Possibly explore a minmum jaccard threshold, so AND (CLOSEST_DIST < X OR JACCARD_SIM > Y)
                    print("Most similar question:", CLOSEST)
                    return random.choice(self.qa[CLOSEST])
                else:
                    print("Not returning. Most similar question:", CLOSEST)
                    print(CLOSEST_DIST)
                    if self.is_blenderbot:
                        return self.generate_personal_response(sentence, dialog_act, act_string, history, generation_config.semantic_threshold, generation_config.dual_model_utterances_to_keep, added_utterance_offset)
                    else:
                        return "I'm not sure how to answer your question."
            
            else:            
                history_embedding = self.sentence_transformer.encode(
                    [entire_utterance.split(':') for entire_utterance in history.all_utt[-generation_config.utterances_to_keep:]])
                if dialog_act in ['acknowledgement', 'inquiry-response', 'commentary', 'thank', 'you-are-welcome', 'ask-not-donate-reason', 'comment-partner', 'praise-user',  'donation-information']:
                    model = self.response_model
                    if self.is_blenderbot and dialog_act in ['thank']:
                        return ''
                    else:
                        return self.generate_personal_response(sentence, dialog_act, act_string, history, generation_config.semantic_threshold, generation_config.dual_model_utterances_to_keep, added_utterance_offset)
                else:
                    model = self.model
                    self.used_blenderbot_already = False #neural generation comes at end of turn; we can use blenderbot again
                    return self.neural_generation(sentence, model, dialog_act, act_string, history, generation_config.semantic_threshold, generation_config.dual_model_utterances_to_keep, added_utterance_offset) 
        utts = []
        keep_generating = True
        while keep_generating and not failed_too_many_times:                                                 
            #Add utterances to utts
            utterances_added = 0
            tuple_hist = reconstruct_history(history)
            for dialog_act in dialog_act_list:
                if not generation_config.require_no_sentences_from_same_turn_in_model_context:
                    utt = generate_for_one_act(dialog_act, 0)
                else:
                    utt = generate_for_one_act(dialog_act, utterances_added)
                if generation_config.regenerate_until_dialog_act and dialog_act[1] != 'DBCALL':
                    classified_act = self.il_classifier.predict_dialog_act(
                        context=tuple_hist, sent=utt, role='A')
                    for i in range(len(classified_act)):
                        act = classified_act[i]
                        act = act.replace('_', '-')
                        act = STRATEGY_TO_ACT_DICT[ACT_TO_STRATEGY_DICT[act]]
                        classified_act[i] = act
                    trial_utt = utt
                    while len(set(classified_act) & set([dialog_act[1]])) == 0:
                        trial_utt = utt + ' ' + generate_for_one_act(dialog_act)
                        classified_act = self.il_classifier.predict_dialog_act(
                            context=tuple_hist, sent=trial_utt, role='A')
                        for i in range(len(classified_act)):
                            act = classified_act[i]
                            act = act.replace('_', '-')
                            act = STRATEGY_TO_ACT_DICT[ACT_TO_STRATEGY_DICT[act]]
                            classified_act[i] = act
                    utt = trial_utt
                utts.append(utt)
                if not generation_config.require_no_sentences_from_same_turn_in_model_context:
                    history.all_utt.append('A: ' + utt)
                    # print(history.all_utt)
                    utterances_added += 1
            if len(utts) == 1:
                keep_generating = False
            else:
                utt_pairs = list(itertools.combinations(utts, 2))
                keep_generating = False
                for (u1, u2) in utt_pairs:
                    u1_embedding = self.sentence_transformer.encode(u1)
                    u2_embedding = self.sentence_transformer.encode(u2)
                    dist = cosine(u1_embedding, u2_embedding)
                    if dist <= generation_config.semantic_threshold:
                        keep_generating = True
                        print("Resetting")
                        attempt_counter += 1
                        if attempt_counter >= 5:
                            failed_too_many_times = True  
                            break      
                        utts = []
                        while utterances_added > 0:
                            history.all_utt.pop(-1)
                            utterances_added -= 1
                        break
                    else:
                        continue

        if failed_too_many_times:
            #Default to a template option if we fail too many times
            return random.choice(SYS_TEMPLATE_ACT_DIC[ACT_TO_STRATEGY_DICT[dialog_act_list[-1][-1].replace('_', '-')]])

        else:
            for i in range(utterances_added):
                history.all_utt.pop(-1)
            for i, utterance in enumerate(utts):
                utts[i] = truecase.get_true_case(postprocessing(utterance))
            generated_utterance = "\n".join(utts)
            generated_utterance = generated_utterance.replace('URL', 'https://www.savethechildren.org/')
            generated_utterance = generated_utterance.replace('Url', 'https://www.savethechildren.org/')
            generated_utterance = generated_utterance.replace('url', 'https://www.savethechildren.org/')
            return generated_utterance
        
    
def main():
    from dialog_history import DialogHistory
    generator = ConditionalGenerator(
        '/local-scratch1/data/maxlchen/persuasion_online/conditional_generation/dialog_lr2e-05_wd0.01_utterances4_acts1_unpreprocessed_cond_penalty0.8_sem_threshold0.2_sem_penalty0.5_dont_collapse_inquiries_one_act_during_training_dont_require_no_same_turn_sentences_in_context__optimal_strategies/checkpoint-9605', device="cuda:0")
    generator2 = ConditionalGenerator(
        '/local-scratch1/data/maxlchen/persuasion_online/conditional_generation/dialog_lr2e-05_wd0.01_utterances4_acts1_unpreprocessed_cond_penalty0.8_sem_threshold0.2_sem_penalty0.5_dont_collapse_inquiries_one_act_during_training_dont_require_no_same_turn_sentences_in_context__responses/checkpoint-9569', device="cuda:0")    
    print(generator.generate(['A: Hi, how are you today? B: Good, how are you? A: Very well. I\'m just up organizing info for my charity. Are you involved with any charities? B: Yes! I work with children who have terminal illnesses. What charity are you involved in? Past Act: greeting Act: acknowledgement'], DialogHistory()))
    print(generator2.generate(['A: Hi, how are you today? B: Good, how are you? A: Very well. I\'m just up organizing info for my charity. Are you involved with any charities? B: Yes! I work with children who have terminal illnesses. What charity are you involved in? Past Act: greeting Act: task-related-inquiry'], DialogHistory()))
    quit()


if __name__ == '__main__':
    main()
