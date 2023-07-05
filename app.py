from flask import Flask, render_template, request
from helpers import predict_sentiment

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    print("Request method: ", request.method)

    if request.method == "GET":
        return render_template("home.html")

    sentence = request.form.get("sentence")
    
    sentiment = predict_sentiment(sentence)
    
    return render_template("home.html", sentiment=sentiment)


if __name__ == "__main__":
    app.run(debug=True)
