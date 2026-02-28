import os
import nltk
from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

def download_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)

download_nltk_resources()


ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stop_words and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]

    transformed_sms = transform_text(message)
    vector_input = vectorizer.transform([transformed_sms])
    prediction = model.predict(vector_input)[0]

    result = "Spam 🚨" if prediction == 1 else "Not Spam ✅"

    return render_template("index.html", prediction=result, message=message)

# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)