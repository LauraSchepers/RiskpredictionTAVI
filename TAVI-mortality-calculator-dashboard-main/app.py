from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained logistic regression model
model = joblib.load('trained_final.pkl')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json

        # Process data and make prediction using the loaded model
        X = [[ 
            data['age'], 
            data['lengthCM'], 
            data['weight'], 
            data['BL_AF'], 
            data['HistoryAorticValveIntervention'], 
            data['NYHA'], 
            data['eGFR'], 
            data['LVEF'], 
            data['AorticValveArea']
        ]]

        # Make prediction
        prediction = model.predict(X)
        probability = model.predict_proba(X)

        # Return prediction result as JSON
        return jsonify({
            'mortality_chance': int(prediction[0]),
            'probability': float(probability[0][1])  # Assuming the second column is the probability of mortality
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
