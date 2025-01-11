import os
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from openai import OpenAI

###############################################################################
#                           OpenAI Client Setup                               #
###############################################################################

# Explicitly load the API key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("\u26a0\ufe0f OpenAI API key not found. Ensure it is set in your environment.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

def generate_recommendation_with_openai(prompt):
    """
    Sends a prompt to OpenAI Chat Completions API (GPT-4 or similar) 
    and returns the model's response.
    """
    try:
        response = client.chat.completions.create(model="gpt-4",  # or "gpt-3.5-turbo", etc.
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial analyst. Provide clear and concise "
                    "investment advice for a non-expert user."
                )
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=400)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error fetching recommendation from OpenAI API: {e}"

###############################################################################
#                      Data Fetching and Visualization                        #
###############################################################################

def fetch_stock_data(stock_ticker, months=12):
    """
    Fetches the last `months` months of data for the provided `stock_ticker`.
    Returns a pandas DataFrame with the fetched data or an empty DataFrame if no data is found.
    """
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30 * months)  # ~12 months
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date, progress=False)
    return stock_data

def plot_stock_data(stock_ticker, stock_data):
    """
    Plots the closing price of the stock over time for better visualization.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data['Close'], label=f'{stock_ticker} Close Price', color='blue', linewidth=2)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title(f'{stock_ticker} - Last 12 Months Closing Prices', fontsize=14, fontweight='bold')
    
    # Find max & min
    max_idx = stock_data['Close'].idxmax()
    min_idx = stock_data['Close'].idxmin()
    max_price = float(stock_data['Close'].max())  # Convert to float
    min_price = float(stock_data['Close'].min())  # Convert to float

    ax.scatter(max_idx, max_price, color='green', marker='^', s=100, label=f'Highest ${max_price:.2f}')
    ax.scatter(min_idx, min_price, color='red', marker='v', s=100, label=f'Lowest ${min_price:.2f}')

    ax.legend()
    st.pyplot(fig)

###############################################################################
#                           Forecasting Logic                                 #
###############################################################################

def forecast_next_weeks(stock_data, weeks=4):
    """
    Uses a simple Linear Regression model to predict the stock's closing price 
    for the next `weeks` weeks.
    """
    y = stock_data['Close'].values
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)

    predictions = {}
    last_index = len(X) - 1

    for w in range(1, weeks + 1):
        future_index = last_index + (w * 7)
        predicted_price = float(model.predict([[future_index]])[0])  # Convert to float
        predictions[f"Week {w}"] = predicted_price

    return predictions

###############################################################################
#                          Streamlit App Interface                            #
###############################################################################

st.title("📈 Stock Analysis & Prediction App")

# User input for stock ticker
stock_ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, GOOG):").strip().upper()

if stock_ticker:
    st.subheader(f"Analyzing {stock_ticker}...")
    stock_data = fetch_stock_data(stock_ticker, months=12)

    if stock_data.empty:
        st.error(f"⚠️ No data found for '{stock_ticker}'. Please check the ticker symbol.")
    else:
        st.success("✅ Data fetched successfully!")
        st.write(stock_data.tail())

        # Plot stock data
        plot_stock_data(stock_ticker, stock_data)

        # Forecasting
        predictions = forecast_next_weeks(stock_data, weeks=4)
        st.subheader("📊 Forecast for the Next 4 Weeks")
        for week, price in predictions.items():
            st.write(f"{week}: **${float(price):.2f}**")  # Ensure float conversion

        # Generate AI-based recommendation
        prompt = f"Stock ticker: {stock_ticker}\nPredicted prices: {predictions}\nProvide a simple investment recommendation."
        recommendation = generate_recommendation_with_openai(prompt)
        st.subheader("💡 AI-Generated Investment Recommendation")
        st.write(recommendation)

