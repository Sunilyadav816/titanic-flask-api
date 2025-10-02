from flask import Flask, request, jsonify
import logging
from model import predict_survival

app = Flask(__name__)

@app.route("/")
def home():
    return "Titanic ML API with XGBoost - Day 7"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    logging.info(f"Received data: {data}")

    if not data:
        return jsonify({"error": "Please provide passenger data"}), 400

    required_fields = ["pclass", "sex", "age", "fare"]
    for field in required_fields:
        if field not in data:
            logging.error(f"Missing field: {field}")
            return jsonify({"error": f"Missing field: {field}"}), 400

    if data["sex"] not in ["male", "female"]:
        return jsonify({"error": "Invalid value for sex"}), 400
    if data["age"] <= 0:
        return jsonify({"error": "Age must be positive"}), 400
    if data["fare"] <= 0:
        return jsonify({"error": "Fare must be positive"}), 400

    survived = predict_survival(data)
    return jsonify({"prediction": survived, "input": data})

if __name__ == "__main__":
    app.run(debug=True)
