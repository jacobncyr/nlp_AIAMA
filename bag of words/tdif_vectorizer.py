from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus of movie reviews
corpus = [
    "I loved the movie, it was fantastic!",
    "The movie was okay, but not great.",
    "I hated the movie, it was terrible.",
]

# Create the Tf-idf vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the corpus
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

# Show the Vocabulary
print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())

# Show the TF-IDF Matrix
print("TF-IDF Representation:")
print(X_tfidf.toarray())