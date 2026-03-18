from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

model = joblib.load("kmeans_model.pkl")
@app.route("/test")
def test():
    pred = model.predict([[60, 70]])
    return {"cluster": int(pred[0])}

@app.route("/")
def home():
    return "Mall Customer KMeans API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["input"]
    prediction = model.predict([data])
    return jsonify({"cluster": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))