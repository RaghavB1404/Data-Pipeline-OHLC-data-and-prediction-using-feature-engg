Data Pipeline: OHLC Data and Prediction Using Feature Engineering
This Python script provides functionalities for fetching OHLC (Open, High, Low, Close) stock data from Quandl, performing technical analysis, generating features, training machine learning models, and saving data to a SQLite database or Parquet files.

Requirements
Python 3
Required Python packages: requests, pandas, numpy, talib, scikit-learn, matplotlib, sqlite3, pyarrow
Installation
Clone the repository: git clone https://github.com/RaghavB1404/data-pipeline-ohlc-data-and-prediction-using-feature-engg.git
Install the required packages: pip install -r requirements.txt
Usage
Run the main.py script.
Follow the prompts to enter the stock symbols and choose indicators for analysis.
The script fetches data, performs analysis, trains models, and saves results to a SQLite database or Parquet files.
Features
Fetch historical OHLC stock data from Quandl.
Perform technical analysis using indicators such as RSI, OBV, Parabolic SAR, and SMA.
Generate features based on selected indicators.
Train RandomForestRegressor models to predict stock prices.
Visualize actual vs. predicted close prices.
Detect outliers and impute missing values using PCA.
Save data to a SQLite database or Parquet files for further analysis.
Contributions
Contributions are welcome! Feel free to submit issues or pull requests.

License
This project is licensed under the MIT License - see the LICENSE file for details.
