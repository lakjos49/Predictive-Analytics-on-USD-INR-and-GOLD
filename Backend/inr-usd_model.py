import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib 

def load_and_preprocess_inr_usd_data(file_path):
    """
    Loads USD/INR conversion rate data, converts 'Date' to datetime,
    and extracts numerical features for the model.
    """
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])

        # Drop rows with any missing values
        df.dropna(inplace=True)

        # Feature Engineering: Convert date to ordinal for linear regression
        df['Date_Ordinal'] = df['Date'].apply(lambda date: date.toordinal())
        df['Month'] = df['Date'].dt.month
        df['Day_of_Year'] = df['Date'].dt.dayofyear

        # Select features (X) and target (y)
        X = df[['Date_Ordinal', 'Month', 'Day_of_Year']]
        y = df['INR_per_USD'] # Predicting INR per USD

        print(f"USD/INR data loaded and preprocessed. Shape: {df.shape}")
        return X, y
    except FileNotFoundError:
        print(f"Error: USD/INR data file not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error processing USD/INR data: {e}")
        return None, None

def train_inr_usd_model(X, y):
    """
    Trains a Linear Regression model for USD/INR rates and evaluates its performance.
    """
    if X is None or y is None:
        print("Cannot train USD/INR model due to missing data.")
        return None

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"USD/INR Rate Linear Regression Mean Squared Error: {mse:.4f}")

    return model

if __name__ == "__main__":
    inr_usd_file = 'inr_usd_conversion_rates_past_2_years.csv'
    inr_usd_model_filename = 'inr_usd_prediction_model.pkl'

    print("--- Starting USD/INR Model Training ---")
    X_inr_usd, y_inr_usd = load_and_preprocess_inr_usd_data(inr_usd_file)

    if X_inr_usd is not None and y_inr_usd is not None:
        inr_usd_model = train_inr_usd_model(X_inr_usd, y_inr_usd)
        if inr_usd_model:
            # Save the trained model to a .pkl file
            joblib.dump(inr_usd_model, inr_usd_model_filename)
            print(f"USD/INR prediction model saved as {inr_usd_model_filename}")
        else:
            print("USD/INR model training failed.")
    else:
        print("USD/INR data could not be loaded or preprocessed.")
    print("--- USD/INR Model Training Finished ---")

