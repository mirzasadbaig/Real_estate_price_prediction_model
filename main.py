import pandas as pd
import pickle
import numpy as np
from flask import Flask, render_template, request

# Load data and model
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    Address = sorted(data['Address'].unique())
    return render_template('index.html', Address=Address)


@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get data (and handle case sensitivity)
    location = request.form.get('Address')
    bhk = request.form.get('Bedrooms')
    bath = request.form.get('Bathrooms')
    sqft = request.form.get('area')  # Ensure this matches your HTML name attribute exactly

    print(location, bhk, bath, sqft)  # Debugging print

    # 2. Create a DataFrame with the SAME column names used during training
    # (Assuming training columns were: Address, Bedrooms, Bathrooms, Area)
    input_data = pd.DataFrame([[location, bhk, bath, sqft]],
                              columns=['Address', 'Bedrooms', 'Bathrooms', 'area'])

    # 3. Convert types from Strings to Numbers
    input_data['Bedrooms'] = pd.to_numeric(input_data['Bedrooms'])
    input_data['Bathrooms'] = pd.to_numeric(input_data['Bathrooms'])
    input_data['area'] = pd.to_numeric(input_data['area'])

    # 4. Predict
    prediction = pipe.predict(input_data)[0]

    # 5. Return the price as a string (Rounded to 2 decimals)
    return str(np.round(prediction, 2))


if __name__ == '__main__':
    app.run(debug=True, port=5001)