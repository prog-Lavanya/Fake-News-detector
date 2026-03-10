import pandas as pd
import re
import pickle
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load Dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

data["content"] = data["title"] + " " + data["text"]

# Text Cleaning
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    return text

data["content"] = data["content"].apply(clean_text)

X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#Train Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

#Prediction
y_pred = model.predict(X_test_tfidf)
nb_pred = nb_model.predict(X_test_tfidf)

#Evaluation

accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy)
nb_accuracy = accuracy_score(y_test, nb_pred)
print("Naive Bayes Accuracy:", nb_accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

#Save Model
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# 9 Test Custom News

news = [input("\nEnter news text: ")]
news_vector = vectorizer.transform(news)

prediction = model.predict(news_vector)

if prediction[0] == 0:
    print("\nPrediction: Fake News")
else:
    print("\nPrediction: Real News")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


#GRAPH
# sns.heatmap(cm, annot=True, fmt='d')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# Indicators 

feature_names = vectorizer.get_feature_names_out()

coefficients = model.coef_[0]

fake_words = sorted(zip(coefficients, feature_names))[:20]
print("\nTop words indicating FAKE news:\n")
for coef, word in fake_words:
    print(word)

real_words = sorted(zip(coefficients, feature_names), reverse=True)[:20]
print("\nTop words indicating REAL news:\n")
for coef, word in real_words:
    print(word)

misclassified = X_test[y_test != y_pred]

print("\nSome misclassified news examples:\n")

for i in misclassified[:5]:
    print(i)
    print("------")