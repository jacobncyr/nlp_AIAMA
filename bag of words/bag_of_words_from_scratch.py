from collections import defaultdict
import string

#sample text ddata: sentences
corpus = [
    "dogs are the best becuase they are loyal.",
    "to be prisoned like an animal and treated like a dog.",
    "HOt dogs are my favourite food to eat in the summer.",
]

#function to preprocess text
def preprocess(text):
    #convert to lowercase
    text = text.lower()
    #remove punctuation
    text = text.translate(str.maketrans("","",string.punctuation))
    #tokenize
    tokens = text.split()
    return tokens

#spply preprocessing to sample corpus
preprocessed_corpus = [preprocess(sentence) for sentence in corpus]
print("preprocessed corpus\n",preprocessed_corpus)

#initialize an empty set for the vocabulary
vocabulary = set()

#build the vocabulary
for sentence in preprocessed_corpus:
    vocabulary.update(sentence)
#convert to a sorted list
vocabulary = sorted(list(vocabulary))
print("Vocabulary:",vocabulary)

#calculate the word frequency and vectorize
def create_bow_vector(sentence, vocab):
    vector = [0] * len(vocab)
    for word in sentence:
        if word in vocab:
            idx = vocab.index(word) #find index of the word in the vocabulary
            vector[idx] += 1
    return vector

#apply the function to our sample data
bow_vectors = [create_bow_vector(sentence,vocabulary) for sentence in preprocessed_corpus]
print("BAg of Words Vectors:")
for vector in bow_vectors:
    print(vector)



