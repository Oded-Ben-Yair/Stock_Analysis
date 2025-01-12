**7-Day Stock Predictions app**

Are you curious about how a specific stock might behave over the next few trading days? This repository provides an application that looks at a chosen ticker (for instance, AAPL, TSLA, or GOOG) and offers a projected closing price for the upcoming seven days. The primary intention is a technical demonstration and educational insight, it's not a stand-in for professional advice!

## Key Highlights

- **Custom Ticker Entry:** Enter a known stock symbol, such as NVDA or INTC.
- **Data Collection:** Retrieves approximately a year's daily stock data using [yfinance](https://pypi.org/project/yfinance).
- **Forecasting:** Implements an ARIMA model to predict stock trends for the next 7 days.
- **Visual Chart:** Generates an interactive line chart with Plotly to showcase the projected trends.
- **News Sentiment Analysis:** Utilizes [NewsAPI](https://newsapi.org) to evaluate whether news headlines convey a positive or negative tone.
- **Insight Summary:** Provides a concise overview of the forecast and sentiment analysis, summarized with the assistance of GPT-4 technology.

## Setup Instructions

1. **Clone** this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/stockapp.git
   ```
2. **Navigate** inside the `stockapp` folder:
   ```bash
   cd stockapp
   ```
3. **Install** the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Add** environment variables to your system:
   - `OPENAI_API_KEY` for your OpenAI key  
   - `YOUR_NEWS_API_KEY` for your NewsAPI key

## Running the App

Type:
```bash
streamlit run Stock_Analysis.py
```
Afterward, open your browser at the address shown in your terminal (usually `http://localhost:8501`). You can provide any valid stock ticker (e.g., `TSLA` or `MSFT`) there. Based on these insights, the app will produce a 7-day prediction, a line graph of the short-term outlook, and a recommendation from GPT-4.

## File Layout

```
stockapp/
├─ .streamlit/         # (Optional) Streamlit configuration
├─ requirements.txt    # Dependency requirements
├─ Stock_Analysis.py   # Main code for the tool
└─ README.md           # This README
```

## How Does It Work?

1. **Data Retrieval:** Downloads roughly the past year of daily quotes from yfinance.  
2. **Modeling:** Fits an ARIMA(5,1,0) framework, predicting a 7-day outlook.  
3. **Charting:** Uses Plotly to show a line graph of the potential price trends for that window.  
4. **Market Sentiment:** Examines the latest headlines via NewsAPI for a bullish or bearish stance.  
5. **OpenAI Advisory:** GPT-4 compiles a short, easy-to-read note factoring in the model's forecast and sentiment.

## Disclaimer

> **Important:**  
> This application is intended **solely** for demonstration and educational purposes.  
> It should not be considered investment or financial advice.  
> You are responsible for your decisions—always consult an authorized financial professional when in doubt.

## License

Offered under the [MIT License](https://opensource.org/licenses/MIT). Feel free to clone or adapt it for your experiments.

---

## Feedback

Do you have feedback, issues, or ideas? Please open an issue or create a pull request. I welcome any suggestions to keep improving!

