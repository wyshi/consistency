from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('bert-base-nli-mean-tokens')#('roberta-large-nli-stsb-mean-tokens')

while True:
    sent1 = input("sent1: ")
    sent2 = input("sent2: ")
    sents = [sent1, sent2]
    sent_embeddings = model.encode(sents)
    print("score: {}".format(cosine_similarity(sent_embeddings[0].reshape(1, -1),
                                               sent_embeddings[1].reshape(1, -1))))