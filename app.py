from flask import Flask, render_template, request, jsonify
from utils.predictor import predict_sentiment
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    drug_name = data.get("drug_name", "").strip()

    if not drug_name:
        return jsonify({"error":"Drug name is required."}),400

    try:
        result = predict_sentiment(drug_name)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error":str(e)}),500


if __name__ == "__main__":
    port = int(os.environ.get("PORT",7860))
    app.run(host="0.0.0.0", port=port)