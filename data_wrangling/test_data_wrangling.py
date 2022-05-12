
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

@pytest.mark.skip
def test_data_array_creation(create_instance):
    dates, data = create_instance.create_data_array()
    # dates, data = create_instance.create_data_array(save_name=create_instance.data_dir / "data-array_test")
    assert data.shape == (int(len(create_instance.cleaned_df) / 391), 5, 391)
    assert len(dates) == data.shape[0] #Check that we have a date for each chunk of data

def test_dataset_creation(create_instance):
    t0=dt.time(19, 0)
    cutoff_date=dt.date(2021, 4, 21)
    X_train, X_test, Y_train, Y_test, R_train, R_test = create_instance.create_dataset(t0, cutoff_date, pytorch=False)
    torch_train, torch_test = create_instance.create_dataset(t0, cutoff_date, pytorch=True)
    assert X_train.shape[0] == len(Y_train)
    assert X_train.shape[0] == len(R_train)
    assert X_test.shape[0] == len(Y_test)
    assert X_test.shape[0] == len(R_test)
    assert len(torch_train) == X_train.shape[0]
    assert len(torch_test) == X_test.shape[0]



# def test_torch_dataset(create_instance):