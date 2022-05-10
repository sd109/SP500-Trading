
# Add unit tests here 

import pathlib, pytest
from .data_wrangling import TickerData
import datetime as dt
import pandas as pd

DATA_DIR = pathlib.Path("/home/scottd/Dropbox/Other-Programming/SP500-Trading/data/")

@pytest.fixture(scope="module") #Base all tests in this file on a single instance of TickerData class by using scope="module"
def create_instance():
    return TickerData(DATA_DIR)



# ------------------------------------------------- Test data loading ------------------------------------------------ #

def test_raw_data_loading(create_instance):
    create_instance.load_data(raw=True)
    assert isinstance(create_instance.raw_df, pd.DataFrame) 
    assert create_instance.raw_df.columns.tolist() == ["Date", "Ticker", "Time", "c", "v"]


def test_cleaned_data_loading(create_instance):
    create_instance.load_data(raw=False)
    assert isinstance(create_instance.cleaned_df, pd.DataFrame) 
    assert create_instance.cleaned_df.columns.tolist() == ["Date", "Ticker", "Time", "c", "v"]


def test_augmented_data_loading(create_instance):
    create_instance.load_aug_df()
    assert isinstance(create_instance.aug_df, pd.DataFrame)
    assert create_instance.aug_df.columns.tolist() == ["Date", "Time", "c_avg","v_avg", "c_vwa"]


def test_time_range():
    time_range = TickerData.create_time_range()
    assert len(time_range) == 391 
    assert all(isinstance(t, dt.time) for t in time_range)

# @pytest.mark.skip
# def test_data_array_creation(create_instance):
#     arr = create_instance.create_data_array()
