from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("spam_detector.pkl", "rb") as file:
    model = pickle.load(file)
    
    
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None

    if request.method == "POST":

        message = request.form["message"]

        prediction = model.predict([message])[0]
        probability = model.predict_proba([message])[0][1] * 100

    return render_template(
        "index.html",
        prediction=prediction,
        probability=round(probability, 2) if probability else None
    )


if __name__ == "__main__":
    app.run(debug=True)