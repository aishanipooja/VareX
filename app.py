import pickle

import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

df = pd.read_csv("news.csv")

x = df["text"]
y = df["label"]

x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


def detection(news):
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)
    ip = [news]
    vect_ip = tfidf_vectorizer.transform(ip)
    pred = model.predict(vect_ip)
    return pred




@app.route('/')
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        prediction = detection(message)
        print(prediction)
        return render_template('index.html', pred=prediction)
    else:
        return render_template('index.html', pred="Something went wrong")

#just checking


# main driver function
if __name__ == '__main__':
    app.run(debug=True)
