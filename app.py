from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

model = joblib.load("kmeans_model.pkl")

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Form prediction
@app.route("/predict_form", methods=["POST"])
def predict_form():
    income = float(request.form["income"])
    score = float(request.form["score"])

    prediction = model.predict([[income, score]])

    print(prediction)s

    return render_template("index.html", prediction=int(prediction[0]))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
