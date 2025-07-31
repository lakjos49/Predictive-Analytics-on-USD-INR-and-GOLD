import pandas as pd
import joblib
import datetime
import os
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns 
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the paths to the model files relative to the script's directory
gold_model_path = os.path.join(BASE_DIR, 'gold_prediction_model.pkl')
inr_usd_model_path = os.path.join(BASE_DIR, 'inr_usd_prediction_model.pkl')

# Load the trained models once when the application starts
try:
    gold_model = joblib.load(gold_model_path)
    inr_usd_model = joblib.load(inr_usd_model_path)
    print("Models loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: A model file was not found. Please ensure 'gold_prediction_model.pkl' and 'inr_usd_prediction_model.pkl' are in the same directory as this script.")
    print(f"Details: {e}")
    gold_model = None
    inr_usd_model = None
except Exception as e:
    print(f"An error occurred while loading the models: {e}")
    gold_model = None
    inr_usd_model = None


# New route to handle requests to the base URL
@app.route('/')
def home():
    """
    Returns a simple message to confirm the server is running.
    """
    return jsonify({"message": "API server is running. Use /predict_point or /predict_range endpoints."})


def get_prediction_features(date_str):
    """
    Takes a date string and returns the features needed for the models.
    """
    try:
        custom_date = pd.to_datetime(date_str)
        date_ordinal = custom_date.toordinal()
        month = custom_date.month
        day_of_year = custom_date.dayofyear
        return pd.DataFrame([[date_ordinal, month, day_of_year]], 
                            columns=['Date_Ordinal', 'Month', 'Day_of_Year'])
    except ValueError:
        return None

@app.route('/predict_point', methods=['POST'])
def predict_point():
    """
    API endpoint for a single-date prediction.
    Expects a JSON payload like: {"date": "YYYY-MM-DD"}
    """
    if not gold_model or not inr_usd_model:
        return jsonify({"error": "Prediction models not loaded. Please check the server logs."}), 500
    
    data = request.get_json()
    date_str = data.get('date')

    if not date_str:
        return jsonify({"error": "Missing 'date' parameter in request body."}), 400

    input_features = get_prediction_features(date_str)
    if input_features is None:
        return jsonify({"error": "Invalid date format. Please use YYYY-MM-DD."}), 400

    try:
        predicted_gold_rate = gold_model.predict(input_features)[0]
        predicted_inr_usd_rate = inr_usd_model.predict(input_features)[0]

        response = {
            "gold_rate": round(predicted_gold_rate, 2),
            "inr_usd_rate": round(predicted_inr_usd_rate, 4),
            "date": date_str
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500


@app.route('/predict_range', methods=['POST'])
def predict_range():
    """
    API endpoint for a date-range prediction.
    Expects a JSON payload like: {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}
    """
    if not gold_model or not inr_usd_model:
        return jsonify({"error": "Prediction models not loaded. Please check the server logs."}), 500
    
    data = request.get_json()
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')

    if not start_date_str or not end_date_str:
        return jsonify({"error": "Missing 'start_date' or 'end_date' parameter."}), 400

    try:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        if start_date > end_date:
            return jsonify({"error": "Start date cannot be after end date."}), 400

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        future_df = pd.DataFrame({'Date': date_range})
        future_df['Date_Ordinal'] = future_df['Date'].apply(lambda date: date.toordinal())
        future_df['Month'] = future_df['Date'].dt.month
        future_df['Day_of_Year'] = future_df['Date'].dt.dayofyear

        future_df['Predicted_Gold_Price'] = gold_model.predict(future_df[['Date_Ordinal', 'Month', 'Day_of_Year']])
        future_df['Predicted_INR_USD_Rate'] = inr_usd_model.predict(future_df[['Date_Ordinal', 'Month', 'Day_of_Year']])

        # --- Plotting logic using matplotlib ---
        sns.set_style("darkgrid")
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        sns.lineplot(x='Date', y='Predicted_Gold_Price', data=future_df, ax=axes[0], color='goldenrod', label='Predicted 24K Gold Price')
        axes[0].set_title(f'Predicted 24K Gold Rate (INR per 10 grams) from {start_date_str} to {end_date_str}', color='white')
        axes[0].set_ylabel('Price (INR)', color='white')
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend()
        axes[0].tick_params(colors='white')
        axes[0].spines['left'].set_color('white')
        axes[0].spines['bottom'].set_color('white')

        sns.lineplot(x='Date', y='Predicted_INR_USD_Rate', data=future_df, ax=axes[1], color='forestgreen', label='Predicted USD/INR Rate')
        axes[1].set_title(f'Predicted USD/INR Rate from {start_date_str} to {end_date_str}', color='white')
        axes[1].set_xlabel('Date', color='white')
        axes[1].set_ylabel('INR per USD', color='white')
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].legend()
        axes[1].tick_params(colors='white')
        axes[1].spines['left'].set_color('white')
        axes[1].spines['bottom'].set_color('white')
        
        plt.tight_layout()
        
        # Save the plot to an in-memory buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)

        # Encode the image to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        response = {
            "plot_image": image_base64
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"An error occurred during range prediction: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
