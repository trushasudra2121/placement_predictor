from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return 'placement prediction API running'

@app.route('/predict', methods = ["POST"])
def predict():
    data = request.json
    features = [[data['cgpa'], data['aptitude'], data['communication'], data["projects"]]]
    
    prediction = model.predict(features)[0]

    return jsonify({'placement_prediction': int(predict)})
import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)