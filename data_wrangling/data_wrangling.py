
from pathlib import Path
import datetime as dt
import pandas as pd
from joblib import Parallel, delayed


class TickerData:

    def __init__(self, data_dir: Path) -> None:
        
        self.data_dir = data_dir
        self.csv_files = [data_dir / x for x in data_dir.iterdir() if x.is_file()]

        return

    def load_raw_data(self) -> pd.DataFrame:
        return pd.concat(map(pd.read_csv, self.csv_files)).reset_index(drop=True) #Reset then drop old out of order index


    # def clean_data(self) -> pd.DataFrame:


    # def fetch_new_data(self, save=False) -> pd.DataFrame:


    def convert_old_data(self, path) -> None:

        old_df = pd.read_csv(path)
        # old_df = old_df[:10000] #Uncomment for quicker testing
        old_df.rename({"Unnamed: 0" : "Date", "Unnamed: 1" : "Ticker", "Unnamed: 2" : "Time"}, axis=1, inplace=True)    
        groups = old_df.set_index("Date").groupby(lambda date_str: date_str[:-3]) #Group df entries by year and month
        for date_str, df in groups:
            df.to_csv(self.data_dir / (date_str + ".csv"))

        return