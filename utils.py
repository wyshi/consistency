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
    return (ratio >= threshold), ratio

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
    if len(re.findall(patternIntOrFloat,sent)) ==0:
        if "a dollar" in sent:
            return float(1)
        elif "one " in sent:
            return float(1)
        elif "two " in sent:
            return float(2)
        elif "three " in sent:
            return float(3)
        elif "four " in sent:
            return float(4)
        elif "five " in sent:
            return float(5)
        elif "six " in sent:
            return float(6)
        elif "seven " in sent:
            return float(7)
        elif "eight " in sent:
            return float(8)
        elif "nine " in sent:
            return float(9)
        elif "zero " in sent:
            return float(0)
        else:
            return None
    else:
        number = float(re.findall(patternIntOrFloat,sent)[0])
        if "cents" in sent or "cent" in sent:
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



