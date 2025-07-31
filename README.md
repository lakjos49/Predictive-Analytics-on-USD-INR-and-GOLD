# Predictive-Analytics-on-USD-INR-and-GOLD

Predictive Analysis Web App
This is a full-stack web application that predicts future rates for 24K Gold (in INR) and the USD/INR exchange rate. It uses machine learning models trained on historical data, which are served via a Python Flask API. The frontend is a responsive web dashboard built with HTML, Tailwind CSS, and vanilla JavaScript for a clean, modern user experience.

The application is composed of two main parts: a backend API that performs the heavy lifting of model inference and plot generation, and a frontend user interface that interacts with the API to display the results.

# Features
1. Point Predictions: Get a single prediction for both Gold and USD/INR rates for a specific date.

2. Future Trend Plots: Generate and visualize a time-series plot of predicted rates over a custom date range.

3. Modular Architecture: The project is cleanly separated into a Python backend and a web frontend, allowing for easier development and maintenance.

4. Stylized Frontend: The user interface is built with HTML and Tailwind CSS for a modern, dark-themed, and fully responsive design.


# Setup and Installation
Prerequisites
You need to have Python 3 and pip installed on your machine.

Backend Setup
Navigate to your project directory:

cd /path/to/your_project

Install the required Python libraries:

pip install Flask Flask-Cors pandas joblib matplotlib seaborn

Place the model files:
Ensure your gold_prediction_model.pkl and inr_usd_prediction_model.pkl files are in the same directory as app.py.

Frontend Setup
The frontend only requires a modern web browser. The necessary libraries (Tailwind CSS) are loaded via a CDN in the HTML file, so no additional installation is needed.

How to Run
1. Start the Backend API
Open your terminal or command prompt and run the Flask application:

python app.py

You should see a message in the console indicating that the server is running, for example:

Models loaded successfully!
 * Serving Flask app 'app'
 * Debug mode: on
...
 * Running on http://127.0.0.1:5000

Keep this terminal window open and running.

2. Open the Frontend
Open the index.html file in your web browser. You can do this by double-clicking the file or by dragging it into the browser window.

The frontend will automatically connect to the backend running at http://127.0.0.1:5000 to get predictions and generate plots.

API Endpoints
The backend provides two API endpoints for the frontend to consume:

POST /predict_point
Description: Makes a prediction for a single specified date.

Request Body (JSON):

{
  "date": "YYYY-MM-DD"
}

Success Response (JSON):

{
  "date": "2025-12-15",
  "gold_rate": 65432.10,
  "inr_usd_rate": 84.7567
}

Error Response (JSON):

{
  "error": "Invalid date format. Please use YYYY-MM-DD."
}

POST /predict_range
Description: Generates a trend plot for a range of dates.

Request Body (JSON):

{
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD"
}

Success Response (JSON):

{
  "plot_image": "base64_encoded_png_image_string"
}

Error Response (JSON):

{
  "error": "Start date cannot be after end date."
}

Troubleshooting
FileNotFoundError: [Errno 2] No such file or directory: 'gold_prediction_model.pkl'

Reason: The Flask app cannot find your .pkl model files.

Solution: Ensure that app.py, gold_prediction_model.pkl, and inr_usd_prediction_model.pkl are all located in the exact same directory.

Not Found error when visiting http://127.0.0.1:5000/

Reason: This is expected. The Flask API is not designed to serve a web page at the base URL. It only responds to the /predict_point and /predict_range endpoints. The index.html file is a separate static file that you open in your browser.

Solution: Simply open index.html directly in your browser.

An error occurred. Check the console for details. on the frontend

Reason: This is a generic error message from the frontend's try/catch block, indicating a problem during the network request.

Solution: Open your browser's developer console (press F12) and look for a more specific error message. Common issues include:

CORS errors: Make sure Flask-CORS is installed and correctly configured in your app.py.

Network unreachable: Verify that your Flask server is running and that your browser can connect to http://127.0.0.1:5000.

Backend error: Check the terminal where your app.py is running for any Python error logs, as a crash on the backend will also cause this error.

By following this documentation, you should be able to set up and run the full application and resolve common issues.
