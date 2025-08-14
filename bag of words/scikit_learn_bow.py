from sklearn.feature_extraction.text import CountVectorizer

#sample text ddata: sentences
corpus = [
    "dogs are the best becuase they are loyal.",
    "to be prisoned like an animal and treated like a dog.",
    "HOt dogs are my favourite food to eat in the summer.",
]

#create a CountVectorizer Object
vectorizer = CountVectorizer()
#fit and transform the corpus
X = vectorizer.fit_transform(corpus)
#print the generated vocabulary
print("vocabulary:",vectorizer.get_feature_names_out())
#print the bag of words matrix
print("BoW Representaion:")
print(X.toarray())