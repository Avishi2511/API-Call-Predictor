import pandas as pd
import os
import numpy as np
from collections import Counter
import pickle

# Step 1: Identify top 3 most called APIs by frequency
def find_top_apis(data, api_column='API Code'):
    api_counts = Counter(data[api_column])
    top_apis = [api for api, count in api_counts.most_common(3)]
    return top_apis

# Step 2: Filter data by APIs and save each to a separate CSV file
def filter_and_save_by_api(data, top_apis, api_column='API Code'):
    os.makedirs("api_files", exist_ok=True)
    api_files = {}
    for api in top_apis:
        api_data = data[data[api_column] == api]
        file_path = f"api_files/{api}.csv"
        api_data.to_csv(file_path, index=False)
        api_files[api] = file_path
        print(f"File saved for {api} at {file_path} with {len(api_data)} entries.")  # Debugging
    return api_files

# Step 3: Custom model training and evaluation
def train_custom_model(file_path):
    # Load data and print column names for verification
    print(f"Training model for file: {file_path}")  # Debugging
    data = pd.read_csv(file_path)
    
    print("Loaded columns:", data.columns.tolist())  # Check column names
    if 'Time of Call' not in data.columns:
        print("Error: 'Time of Call' column not found. Check for exact spelling or extra spaces.")
        return None, None, None

    # Convert 'Time of Call' to datetime, setting dayfirst=True
    data['Time of Call'] = pd.to_datetime(data['Time of Call'], dayfirst=True, errors='coerce')

    # Identify and print rows with invalid datetime entries
    invalid_dates = data[data['Time of Call'].isnull()]
    if not invalid_dates.empty:
        print("Warning: Some entries in 'Time of Call' could not be parsed.")
        print("Invalid entries:\n", invalid_dates)
        
        # Drop rows with NaT in 'Time of Call'
        data = data.dropna(subset=['Time of Call'])

    # Proceed with feature extraction if there are valid rows left
    if data.empty:
        print(f"Error: After dropping invalid dates, {file_path} is empty.")
        return None, None, None
    
    data['Hour'] = data['Time of Call'].dt.hour
    data['DayOfWeek'] = data['Time of Call'].dt.dayofweek
    
    # Drop 'Time of Call' column as we have extracted relevant features
    X = data[['Hour', 'DayOfWeek']].values
    y = data['API Code'].values  # Using `API Code` as a placeholder target for demonstration
    
    # Split the data manually (80-20 split)
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Custom simple model: Average-Based Predictor
    avg_hour = int(np.mean(X_train[:, 0]))  # Mean of the 'Hour' feature
    avg_day = int(np.mean(X_train[:, 1]))  # Mean of the 'DayOfWeek' feature
    
    # Model evaluation based on simple matching criteria
    def evaluate_model(X_test, avg_hour, avg_day):
        correct_predictions = 0
        for i in range(len(X_test)):
            hour, day = X_test[i]
            if abs(hour - avg_hour) < 2 and day == avg_day:
                correct_predictions += 1
        accuracy = correct_predictions / len(X_test)
        return accuracy
    
    # Evaluate the model based on accuracy
    model_accuracy = evaluate_model(X_test, avg_hour, avg_day)
    
    # Save the model using pickle
    model_path = file_path.replace(".csv", "_custom_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({'avg_hour': avg_hour, 'avg_day': avg_day}, f)
    
    return "CustomAvgModel", model_accuracy, model_path

# Step 4: Main function to execute the workflow
def main(csv_path, api_column='API Code', time_column='Time of Call'):
    # Load the data from CSV
    data = pd.read_csv(csv_path)
    
    # Step 1: Find the top 3 APIs by frequency
    top_apis = find_top_apis(data, api_column)
    print(f"Top 3 APIs: {top_apis}")
    
    # Step 2: Filter and save data for each top API
    api_files = filter_and_save_by_api(data, top_apis, api_column)
    
    # Step 3: Train and find the best model for each API file
    for api, file_path in api_files.items():
        best_model_name, accuracy, model_path = train_custom_model(file_path)
        if best_model_name:
            print(f"Best model for {api}: {best_model_name} with accuracy {accuracy:.2f} (saved at {model_path})")

# Example usage
# Replace 'your_data.csv' with the path to your actual CSV file
main('C:/Users/hp/Documents/GitHub/API-Call-Predictor/API Call Dataset.csv', api_column='API Code', time_column='Time of Call')
