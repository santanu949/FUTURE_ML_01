import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import os

def run_forecasting():
    print("Starting Future Sales Forecasting...")
    
    # 1. Generate Synthetic Sales Data
    print("Generating sample sales data...")
    dates = pd.date_range(start='2025-07-01', periods=60)
    sales = np.random.randint(50, 250, size=60)
    data = pd.DataFrame({'ds': dates, 'y': sales})
    
    # Save raw data
    data.to_csv('sales_data.csv', index=False)
    print("Sample data saved to 'sales_data.csv'.")
    
    # 2. Initialize and Train Prophet Model
    print("Training the Prophet model...")
    model = Prophet(yearly_seasonality=False, daily_seasonality=False)
    model.fit(data)
    
    # 3. Create Future Dataframe for Prediction
    print("Generating predictions for the next 30 days...")
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # 4. Display Results
    print("\nForecast Results (Last 5 days):")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    # 5. Visualize and Save Plots
    print("\nGenerating visualization plots...")
    
    # Forecast Plot
    fig1 = model.plot(forecast)
    plt.title("Sales Forecast Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    fig1.savefig('forecast_plot.png')
    
    # Components Plot (Trend, Weekly)
    fig2 = model.plot_components(forecast)
    fig2.savefig('forecast_components.png')
    
    print("Plots saved as 'forecast_plot.png' and 'forecast_components.png'.")
    print("\nForecasting process completed successfully!")

if __name__ == "__main__":
    run_forecasting()
