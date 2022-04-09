
import sys, os, pathlib
from importlib import reload
sys.path.append("/home/scottd/Dropbox/Other-Programming/SP500-Trading/data_wrangling")
from data_wrangling import *

DATA_DIR = pathlib.Path("/home/scottd/Dropbox/Other-Programming/SP500-Trading/data/")
data = TickerData(DATA_DIR)

data.convert_old_data("/home/scottd/SP500-Ticker-Data/full-data.csv.gz")