# Delhi Real Estate Price Prediction

A full-stack machine learning application designed to estimate residential property prices in Delhi, India. This project integrates a Ridge Regression model with a Flask web interface to provide real-time market value predictions based on property attributes such as location, square footage, and layout.

## Project Overview

The core of this system is a regression pipeline built using Scikit-Learn. The model addresses common challenges in real estate data, specifically high cardinality in categorical features (locations) and multicollinearity among predictors.

**Key Technical Features:**

* **Dimensionality Reduction:** Implements logic to handle high-cardinality location data by grouping rare occurrences (less than 10 data points) into an 'Other' category.
* **Pipeline Architecture:** Utilizes `sklearn.pipeline.Pipeline` to ensure preprocessing steps (OneHotEncoding and Scaling) are consistently applied to both training data and inference inputs.
* **Regularization:** employs Ridge Regression to penalize large coefficients, reducing overfitting and improving generalization on unseen data.

## Technology Stack

* **Backend:** Python 3.13, Flask
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Frontend:** HTML, CSS (Bootstrap)
* **Serialization:** Pickle

## Installation and Usage

To run this project locally, follow these steps:

**1. Clone the repository**

```bash
git clone https://github.com/mirzasadbaig/Real_estate_price_prediction_model.git
cd Real_estate_price_prediction_model

```

**2. Set up a virtual environment**

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

```

**3. Install dependencies**

```bash
pip install -r requirements.txt

```

**4. Run the application**

```bash
python main.py

```

The application will launch at `http://127.0.0.1:5001/`.

## Data Pipeline

The data processing workflow is automated within the `train.py` script (or notebook) and adheres to the following logic:

1. **Data Cleaning:** Removal of non-essential features (e.g., 'Landmarks', 'Balcony', 'Status') and null value handling.
2. **Feature Engineering:** Transformation of the 'Address' column. Locations appearing fewer than 10 times are remapped to 'other' to prevent the OneHotEncoder from generating an excessive number of columns (curse of dimensionality).
3. **Transformation:**
* **Categorical:** OneHotEncoder (sparse_output=False)
* **Numerical:** StandardScaler (with_mean=False)


4. **Modeling:** The transformed data is passed to a Ridge Regression estimator.

## Model Performance

During development, three linear models were evaluated:

* Linear Regression
* Lasso Regression (L1)
* Ridge Regression (L2)

Ridge Regression was selected for the final deployment due to its superior performance metrics (`r2_score`) and stability when dealing with the feature set derived from the Delhi housing dataset.

## Repository Structure

```text
Real_estate_price_prediction_model/
├── templates/          # HTML templates for the frontend
├── Cleaned_data.csv    # Preprocessed dataset used for training
├── Delhi_v2.csv        # Raw initial dataset
├── main.py             # Application entry point (Flask)
├── RidgeModel.pkl      # Serialized model pipeline
├── requirements.txt    # Project dependencies
└── README.md           # Documentation

```

## License

This project is open source and available under the MIT License.
