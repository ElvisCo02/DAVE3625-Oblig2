import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score #gives a score for performance of algorythm

#Load the data
data = pd.read_csv('TSLA.csv')  

#Converts 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

#Sorts the data by date (dont know if it is sorted or not)
data = data.sort_values('Date')

#Function to predict stock price for a specific date
def predict_stock_price(input_date):
    # Extract features and target variable which in this case is date and close
    X = data['Date'].dt.strftime('%Y%m%d').astype(int).values.reshape(-1, 1)
    y = data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Convert input date to the format used in the model
    input_date_formatted = pd.to_datetime(input_date).strftime('%Y%m%d')
    input_date_encoded = int(input_date_formatted)

    # Predict for the input date
    predicted_price = model.predict([[input_date_encoded]])

    # Calculate the R-squared score
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    return predicted_price[0], score

#Example usage 1:
input_date = '2023-11-01'  
predicted_price, score = predict_stock_price(input_date)
print(f"Predicted stock price for {input_date}: ${predicted_price:.2f}")
print(f"Prediction percentage score (R-squared): {score:.4f}")

#Example usage 2:
input_date = '2024-11-01'  
predicted_price, score = predict_stock_price(input_date)
print(f"Predicted stock price for {input_date}: ${predicted_price:.2f}")
print(f"Prediction percentage score (R-squared): {score:.4f}")

#Example usage 3:
input_date = '2025-11-01'  
predicted_price, score = predict_stock_price(input_date)
print(f"Predicted stock price for {input_date}: ${predicted_price:.2f}")
print(f"Prediction percentage score (R-squared): {score:.4f}")


# output: 
Predicted stock price for 2023-11-01: $213.79
Prediction percentage score (R-squared): 0.1485
Predicted stock price for 2024-11-01: $259.80
Prediction percentage score (R-squared): 0.1485
Predicted stock price for 2025-11-01: $305.81
Prediction percentage score (R-squared): 0.1485