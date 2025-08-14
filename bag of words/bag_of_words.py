#bag of words, why?
#text classification and sentiment analysys

#find the vocabulary iun the corpus and measure the occurence of each word
#order and structre is not important

#first we will tokenize a sentence

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#download the stopwords and tokenizer
nltk.download("punkt_tab")
nltk.download("stopwords")

#example sentence
sentence = "this is an example sentence showing how to remove stopwords from a sentence."

#tokenize the sentence into words
words = word_tokenize(sentence)

#get and print the list of stop words in english
stop_words = set(stopwords.words("english"))
print(f"stopwords : \n")
print(stop_words)

#remove stopwords from the sentence
filtered_sentence = [word for word in words if word.lower() not in stop_words]

#join the words back into a sentence
filtered_sentence = " ".join(filtered_sentence)
print(filtered_sentence)


#now we want to build a vocabulary
import re
# Import the regular expressions module to help with text processing
from collections import (
    defaultdict,
)
#import defaultdict to easily handle word frequency counting
#sample corpus of text - a small dataset of sentences to analyze
corpus = [
"my name is jacob and im here to do some computer science.",
    "python is my version of having a glass that i can use to peer into the workings of the world around me.",
    "mathematics is my favourite hobby in the world.",
]

#initialize a defaultdict with integer values to store word frequencies
#defaultdict(int) initialized each new key with a default integer value of 0
vocab = defaultdict(int)

#loop through each sentence in the corpus to tokenize and normalize
for sentence in corpus:
    #convert the sentence to lowercase to ensure consistency in counting
    #use regex to find words composed of alphanumeric characters
    words = re.findall(r"\b\w+\b",sentence.lower())
    for word in words:
        vocab[word] += 1
#convert the defaultdict vocab to a regular dictionary for easier handling and sorting
sorted_vocab = dict(sorted(vocab.items(),key = lambda x: x[1], reverse=True))

#display the sorted vocabulary with each word and its frequency count
print("Vocabulary with Frequencies:",sorted_vocab)