import pytest 
import pandas as pd 
from Ingestion import fetch_data_from_quandl, calculate_rolling_stats, mean_reversion_signal, resample_data

@pytest.fixture()
def sample_data():
    data = {
        'Date': pd.date_range(start='2022-01-01',end='2022-01-10'),
        'Close': [100, 102, 105, 103, 101, 99, 98, 96, 97, 95],
        'Volume': [100000, 120000, 110000, 95000, 105000, 98000, 102000, 97000, 105000, 110000]
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df 

def test_fetch_data_from_quandl():
    symbol_list = ['AAPL']
    data = fetch_data_from_quandl(symbol_list)
    assert isinstance(data, dict)
    assert 'AAPL' in data
    assert isinstance(data['AAPL'], pd.DataFrame)

def test_calculate_rolling_stats(sample_data):
    calculate_rolling_stats(sample_data)
    assert 'rolling_mean' in sample_data.columns
    assert 'rolling_std' in sample_data.columns

def test_mean_reversion_signal(sample_data):
    calculate_rolling_stats(sample_data)
    mean_reversion_signal(sample_data)
    assert 'z_score' in sample_data.columns
    assert 'mean_reversion_signal' in sample_data.columns

def test_resample_data(sample_data):
    resampled_data = resample_data(sample_data, frequency='W')
    assert len(resampled_data) == 2 


    