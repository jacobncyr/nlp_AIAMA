from sklearn.feature_extraction.text import CountVectorizer

# Sample corpus of movie reviews
corpus = [
    "I loved the movie, it was fantastic!",
    "The movie was okay, but not great.",
    "I hated the movie, it was terrible.",
]

#initialize the CountVectorizer
vectorizer = CountVectorizer()

#fit and transform the corpus to a document-term matrix
X = vectorizer.fit_transform(corpus)

#convert the document-term matrix into a dense format(for visualiation)
X_dense = X.toarray()

#get the vocabulary(mapping of words to index postiions)
vocab = vectorizer.get_feature_names_out()

#print the vocabulary and document-term matrix
print("vocabulary:",vocab)
print("Document-term Matrix:\n",X_dense)