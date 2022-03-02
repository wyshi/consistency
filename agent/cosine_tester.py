from sentence_transformers import SentenceTransformer
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
from scipy.spatial.distance import cosine
sentence1 = input("Sentence 1:")
sentence_embedding1 = sentence_transformer.encode(sentence1)
sentence2 = input("Sentence 2:")
sentence_embedding2 = sentence_transformer.encode(sentence2)

dist = cosine(sentence_embedding1, sentence_embedding2)
print("Cosine distance: ", dist)
