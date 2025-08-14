from nltk import ngrams
sentence = input("enter the sentence: ")
n = int(input("enter the value of n: "))
n_grams = ngrams(sentence.split(),n)
for grams in n_grams:
    print(grams)