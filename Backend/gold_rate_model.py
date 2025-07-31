import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib # Using joblib for saving/loading scikit-learn models

def load_and_preprocess_gold_data(file_path):
    """
    Loads gold rate data, converts 'Year' and 'Month' to a datetime object,
    and extracts numerical features for the model.
    """
    try:
        df = pd.read_csv(file_path)
        # Create a 'Date' column from 'Year' and 'Month'.
        # Assuming the day is the 1st of the month for simplicity.
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%B')
        
        # Drop rows with any missing values
        df.dropna(inplace=True)

        # Feature Engineering: Convert date to ordinal for linear regression
        # Also add month and day of year to capture seasonality
        df['Date_Ordinal'] = df['Date'].apply(lambda date: date.toordinal())
        df['Month'] = df['Date'].dt.month
        df['Day_of_Year'] = df['Date'].dt.dayofyear
        
        # Select features (X) and target (y)
        X = df[['Date_Ordinal', 'Month', 'Day_of_Year']]
        y = df['Avg_24K_Price'] # Predicting average 24K gold price
        
        print(f"Gold data loaded and preprocessed. Shape: {df.shape}")
        return X, y
    except FileNotFoundError:
        print(f"Error: Gold data file not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error processing gold data: {e}")
        return None, None

def train_gold_model(X, y):
    """
    Trains a Linear Regression model for gold rates and evaluates its performance.
    """
    if X is None or y is None:
        print("Cannot train gold model due to missing data.")
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
    print(f"Gold Rate Linear Regression Mean Squared Error: {mse:.2f}")

    return model

if __name__ == "__main__":
    gold_file = 'Gold_Rates_New_Delhi_2020_2025.csv'
    gold_model_filename = 'gold_prediction_model.pkl'

    print("--- Starting Gold Model Training ---")
    X_gold, y_gold = load_and_preprocess_gold_data(gold_file)
    
    if X_gold is not None and y_gold is not None:
        gold_model = train_gold_model(X_gold, y_gold)
        if gold_model:
            # Save the trained model to a .pkl file
            joblib.dump(gold_model, gold_model_filename)
            print(f"Gold prediction model saved as {gold_model_filename}")
        else:
            print("Gold model training failed.")
    else:
        print("Gold data could not be loaded or preprocessed.")
    print("--- Gold Model Training Finished ---")

