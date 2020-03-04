import nltk.corpus
import nltk.tokenize.punkt
import nltk.stem.snowball
import string
# import spacy
# spacy_en = spacy.load("en")
import config as cfg
# Get default English stopwords and extend with punctuation
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')

# Create tokenizer and stemmer
from nltk import word_tokenize

def is_ci_token_stopword_set_match(a, b, threshold=0.5):
    """Check if a and b are matches."""
    # text = [tok.text for tok in spacy_en.tokenizer(text) if tok.text != ' ']
    tokens_a = [token.lower().strip(string.punctuation) for token in word_tokenize(a) \
                    if token.lower().strip(string.punctuation) not in stopwords]
    tokens_b = [token.lower().strip(string.punctuation) for token in word_tokenize(b) \
                    if token.lower().strip(string.punctuation) not in stopwords]

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
            if cfg.debug:
                print("--- repetition occurs between these sents ---")
                print("|{}|\n|{}|".format(c_sent, sent))
                print("---------")
            return True, max_ratio
    return False, max_ratio
