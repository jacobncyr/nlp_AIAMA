# Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (expand with more examples)
texts = [
    "nimitho-kiskisin kā-kī-pī-awāsisīwiyāhk nītha ikwa nītisānak onikāyāmihk ninīkihikonānak, nitōsisinānak, nohkomisinānak, nimosōminānak, nōhkominānak ikwa nītisānak asici. kahkithaw mīna nikī-nīhithawānān.",
    "I have good memories of us children growing up at Uskik Lake with our parents, aunts, uncles, grandparents and brothers and sisters (cousins). We all spoke Cree too",
    "onikāyāmihk anima pikwācāyihk astīw anima sākahikan ikota māhtāwi-sīpiy kā-pimiciwahk kisiwāk askihk-pāwistikohk.  niyānan onikāhpa poko ka-miyāskaman, pāwistikwa, kayāsi asinīyāpiskāw, asīniyāpiyak mīna ikota apiwak ikwa misāci nanātohk pisiskiwak ayāwak. ikota kā-kī-wīkicik kitāniskocāpaninawak ikwa iyako kā-kī-nakatamākoyāhkwāw",
    "Uskik Lake is an isolated lake located along the Churchill River near Kettle Falls. There are five portages to get through, rapids, precambrian shield of rock with some areas covered with rock paintings and there is an abundance of wildlife. It was the home of our ancestors and is our inheritance",
]
labels = ["cree", "english", "cree", "english"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), lowercase=False)
X = vectorizer.fit_transform(texts)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=2, random_state=42,stratify=labels)

# Train Classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["cree", "english"])
sns.heatmap(cm, annot=True, fmt='d', xticklabels=["Cree", "English"], yticklabels=["Cree", "English"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Prediction Function
def predict_language(text):
    X_new = vectorizer.transform([text])
    prediction = model.predict(X_new)[0]
    confidence = model.predict_proba(X_new).max()
    return prediction, confidence

# Example Usage
sample_text = "i got pac man fever im going out of my mind"
lang, score = predict_language(sample_text)
print(f"Predicted Language: {lang} (Confidence: {score:.2f})")
