from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the best model and scaler
best_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json
    df = pd.DataFrame([data])

    # Preprocess data
    df_scaled = scaler.transform(df)

    # Make prediction with the best model
    prediction = best_model.predict(df_scaled)[0]

    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
