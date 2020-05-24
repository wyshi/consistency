import nltk.corpus
import nltk.tokenize.punkt
import nltk.stem.snowball
import string
# import spacy
# spacy_en = spacy.load("en")
import config as cfg
# Get default English stopwords and extend with punctuation
stopwords = ['save_the_children']
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# logging.basicConfig(filename=cfg.repetition_log_file,level=logging.DEBUG)

# Create tokenizer and stemmer
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer

def is_ci_token_stopword_set_match(a, b, threshold=0.5):
    """Check if a and b are matches."""
    # text = [tok.text for tok in spacy_en.tokenizer(text) if tok.text != ' ']
    a = a.lower().replace("save the children", "save_the_children")
    b = b.lower().replace("save the children", "save_the_children")
    tokens_a_tmp = [token.lower().strip(string.punctuation) for token in word_tokenize(a) \
                    if token.lower().strip(string.punctuation) not in stopwords]
    tokens_b_tmp = [token.lower().strip(string.punctuation) for token in word_tokenize(b) \
                    if token.lower().strip(string.punctuation) not in stopwords]
    
    if len(tokens_a_tmp) == 0 and len(tokens_b_tmp) == 0:
        tokens_a = [token.lower().strip(string.punctuation) for token in word_tokenize(a)]
        tokens_b = [token.lower().strip(string.punctuation) for token in word_tokenize(b)]
    else:
        tokens_a = tokens_a_tmp
        tokens_b = tokens_b_tmp
    # Calculate Jaccard similarity
    try:
        ratio = len(set(tokens_a).intersection(tokens_b)) / float(len(set(tokens_a).union(tokens_b)))
    except:
        print("divided by zero!")
        print(a)
        print(b)
    # print(ratio)
    if len(tokens_a) > 2:
        return (ratio >= threshold), ratio
    else:
        return False, ratio

def is_repetition_with_context(sent, context_list, threshold=0.5):
    """
    check if the current sentence candidate has repetition with the current context
    one exception: "B: can you remind me how to donate? A: directly deducted from " is not a repetition
    """
    context_list = list(set(context_list)) # because one user_input_text can belong to multiple system_act in the user_profile
    max_ratio = -100
    for c_sent in context_list:
        is_match, ratio = is_ci_token_stopword_set_match(sent, c_sent, threshold=threshold)
        max_ratio = max(ratio, max_ratio)
        if is_match:
            if cfg.verbose:
                print("\n\n\n--- repetition occurs between these sents: ratio {} ---".format(ratio))
                print("|context: {}|\n|candidate: {}|".format(c_sent, sent))
                print("---------------------------------------------\n\n\n")

            logging.debug("--- repetition occurs between these sents: ratio {} ---".format(ratio))
            logging.debug("|context: {}|\n|candidate: {}|".format(c_sent, sent))
            logging.debug("---------------------------------------------")
            return True, max_ratio
    return False, max_ratio

def toNumReg(sent):
    sent = sent.lower()
    regInt=r'^0$|^[1-9]\d*$'
    regFloat=r'[-+]?\d*\.\d+|\d+'
    # regWord = r'(one)|(two)|(three)|(four)|(five)|(six)|(seven)|(eight)|(nine)|(zero)'


   
    regIntOrFloat=regInt+'|'+regFloat#float / int

    patternIntOrFloat=re.compile(regIntOrFloat)
    if len(re.findall(patternIntOrFloat,sent)) == 0:
        if "a dollar" in sent:
            return float(1)
        else:
            num = None
            if "one " in sent:
                num = float(1)
            elif "two " in sent:
                num = float(2)
            elif "three " in sent:
                num = float(3)
            elif "four " in sent:
                num = float(4)
            elif "five " in sent:
                num = float(5)
            elif "six " in sent:
                num = float(6)
            elif "seven " in sent:
                num = float(7)
            elif "eight " in sent:
                num = float(8)
            elif "nine " in sent:
                num = float(9)
            elif "zero " in sent:
                num = float(0)
            elif "fifty " in sent:
                num = float(50)
            else:
                num = None
            if " cents" in sent or " cent" in sent:
                if num is not None:
                    num = 0.01 * num
            else:
                num = num
            return num
    else:
        number = float(re.findall(patternIntOrFloat,sent)[0])
        if " cents" in sent or " cent" in sent:
            number = 0.01 * number

        return number
        # return float(re.findall(patternIntOrFloat,sent)[0])
        
    return None


def compare_two_sent_embeddings(sent1, sent2):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sents = [sent1, sent2]
    sent_embeddings = model.encode(sents)

    return cosine_similarity(sent_embeddings[0].T.reshape(1,-1), sent_embeddings[1].T.reshape(1, -1))

import pickle as pkl

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()



import nltk
import pdb

def filter_ngram(utt, n):
    words = utt.split()
    right = n
    repeat = False
    while right <= len(words):
        cur_grams = nltk.ngrams(words[:right], n)
        fdist = nltk.FreqDist(cur_grams)
        if any([v > 1 for v in fdist.values()]):
            repeat = True
            break
        # for k, v in fdist.items():
        #     if v > 1:
        #         break
        # pdb.set_trace()
        right += 1
    if repeat:
        right = right - n
    return " ".join(words[:right])

import collections
import nltk
from nltk import ngrams
import numpy as np
import tqdm
# automatic metrics
def compute_distinct(inputs, n=4):
    counter = collections.Counter()
    total_count = 0   
    for item in inputs:
        hyp = nltk.word_tokenize(item[1].lower())
        n_grams = list(ngrams(hyp, n=n))  
        counter.update(n_grams)
        total_count += len(n_grams)
    return len(counter) / total_count

def get_human_n_grams(inputs, n=4):
    human_n_grams = collections.Counter()
    for item in tqdm.tqdm(inputs):
        list_n_grams = ngrams(nltk.word_tokenize(item.lower()), n=n)
        human_n_grams.update(list_n_grams)       
    human_n_grams = {k:v for k,v in human_n_grams.items() if v > 1}
    return human_n_grams

def compute_sentence_repeat(inputs, human_n_grams, n=4):
    scores = []
    for item in inputs:
        count = 0
        tokens = nltk.word_tokenize(item[1].lower())
        n_grams = list(ngrams(tokens, n=n))
        for n_gram in n_grams:
            if n_gram in human_n_grams:
                count += 1
        if len(n_grams) == 0:
            scores.append(0)
        else:
            scores.append(count/len(n_grams))
    return np.mean(scores)

from nltk.translate.bleu_score import sentence_bleu
def compute_bleu(inputs, n=2):
    if n==3:
        weights=(0.333, 0.333, 0.333, 0)
    elif n==2:
        weights=(0.5, 0.5, 0.0, 0)
    elif n==4:
        weights=(0.25, 0.25, 0.25, 0.25)
    else:
        # assert False
        weights=(1, 0, 0, 0)
    scores = []   
    for item in inputs:
        ref = nltk.word_tokenize(item[0].lower())
        hyp = nltk.word_tokenize(item[1].lower())       
        score = sentence_bleu([ref], hyp, weights=weights)
        scores.append(score)
    return np.mean(scores)

def automatic_metric(final_targets_A, final_responses_A, final_targets_B, final_responses_B):
    # distinct
    A_distinct = compute_distinct(zip(final_targets_A, final_responses_A))
    B_distinct = compute_distinct(zip(final_targets_B, final_responses_B))
    all_distinct = compute_distinct(zip(final_targets_A+final_targets_B, final_responses_A+final_responses_B))

    # repeat
    human_n_grams_A = get_human_n_grams(final_targets_A)
    A_repeat = compute_sentence_repeat(zip(final_targets_A, final_responses_A), human_n_grams_A)
    human_n_grams_B = get_human_n_grams(final_targets_B)
    B_repeat = compute_sentence_repeat(zip(final_targets_B, final_responses_B), human_n_grams_B)
    all_human_n_grams = get_human_n_grams(final_targets_A+final_targets_B)
    all_repeat = compute_sentence_repeat(zip(final_targets_A+final_targets_B, final_responses_A+final_responses_B), all_human_n_grams)

    # bleu 2
    A_bleu_2 = compute_bleu(zip(final_targets_A, final_responses_A), n=2)
    B_bleu_2 = compute_bleu(zip(final_targets_B, final_responses_B), n=2)
    all_bleu_2 = compute_bleu(zip(final_targets_A+final_targets_B, final_responses_A+final_responses_B), n=2)

    A_bleu_1 = compute_bleu(zip(final_targets_A, final_responses_A), n=1)
    B_bleu_1 = compute_bleu(zip(final_targets_B, final_responses_B), n=1)
    all_bleu_1 = compute_bleu(zip(final_targets_A+final_targets_B, final_responses_A+final_responses_B), n=1)

    with open("Eval/simulated_dialogs.txt", "a") as fh:
        print(f"distinct: A: {A_distinct}, B: {B_distinct}, all: {all_distinct}")
        print(f"repeat: A: {A_repeat}, B: {B_repeat}, all: {all_repeat}")
        print(f"bleu-2: A: {A_bleu_2}, B: {B_bleu_2}, all: {all_bleu_2}")
        print(f"bleu-1: A: {A_bleu_1}, B: {B_bleu_1}, all: {all_bleu_1}")

        fh.write(f"distinct: A: {A_distinct}, B: {B_distinct}, all: {all_distinct}")
        fh.write(f"repeat: A: {A_repeat}, B: {B_repeat}, all: {all_repeat}")
        fh.write(f"bleu-2: A: {A_bleu_2}, B: {B_bleu_2}, all: {all_bleu_2}")
        fh.write(f"bleu-1: A: {A_bleu_1}, B: {B_bleu_1}, all: {all_bleu_1}")

