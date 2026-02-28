
import os
import nltk
from flask import Flask, render_template, request
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

ps = PorterStemmer()

# Load trained model & vectorizer

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# loaded_model = pickle.load(open("model.pkl","rb"))
# print(hasattr(loaded_model, "classes_"))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

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

    return render_template("index.html",prediction=result, message=message)


if __name__ == "__main__":
    app.run(debug=True)