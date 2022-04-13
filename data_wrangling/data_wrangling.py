
import os
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


class TickerData:

    def __init__(self, data_dir: Path) -> None:
        
        #Filepath to data directory
        self.data_dir = data_dir
        #Empty attributes which will be loaded as needed
        self.raw_df = None
        self.cleaned_df = None
        self.avg_df = None

        return
        

    def load_data(self, raw=False) -> pd.DataFrame:

        if raw:
            df = pd.concat(map(pd.read_csv, (self.data_dir / "raw").iterdir())).reset_index(drop=True) #Reset then drop old out of order index
        else:
            df = pd.read_csv(self.data_dir / "cleaned-data.csv")

        # df = pd.concat(map(pd.read_csv, list((self.data_dir / "raw").iterdir())[:1])).reset_index(drop=True) #Use subset of data for fast testing
        #Convert time and date columns from strings to datetime objects
        df.Time = df.Time.map(dt.time.fromisoformat)
        df.Date = df.Date.map(dt.date.fromisoformat)

        #Set data as instance attribute too for convenience
        if raw:
            self.raw_df = df
        else:
            self.cleaned_df = df

        return df


    def load_avg_df(self):
        avg_df = pd.read_csv(self.data_dir / "avg-data.csv")
        avg_df.Time = avg_df.Time.map(dt.time.fromisoformat)
        avg_df.Date = avg_df.Date.map(dt.date.fromisoformat)
        self.avg_df = avg_df
        return avg_df


    def _clean_day_data(self, group, max_missing):
        (date, ticker), raw_df_subset = group #Unpack pandas groupby element
        if len(raw_df_subset) < (391 - max_missing): 
            return
        else:
            #Create multi-index for existing df
            raw_df_subset.set_index(pd.MultiIndex.from_product([[date], [ticker], raw_df_subset.Time]), inplace=True)
            #Create multi-index for empty df (multi-index is needed so that df.update works)
            multi_idx = pd.MultiIndex.from_product([[date], [ticker], self.time_range])
            df = pd.DataFrame(self.empty_day_df, index=multi_idx)
            #Make sure Date, Time and Ticker columns have no NaNs
            df.Date = date
            df.Time = self.time_range
            df.Ticker = ticker
            #Add existing data to empty day df
            df.update(raw_df_subset)
            #Interpolate missing values
            df.interpolate(axis=0, method="linear", limit_direction="both", inplace=True)
            return df


    @staticmethod
    def create_time_range(): #Useful for filtering various timeseries which don't have an attached time column
        return np.array([dt.time(14, m) for m in range(30, 60)] + [dt.time(h, m) for h in range(15, 21) for m in range(60)] + [dt.time(21, 0)])

    def clean_data(self, max_missing: int, save_name=None) -> pd.DataFrame:

        if self.raw_df is None:
            print("Loading raw data")
            raw_df = self.load_data(raw=True)

        #Save some attributes which will be used multiple times during cleaning loop
        self.time_range = TickerData.create_time_range() #[dt.time(14, m) for m in range(30, 60)] + [dt.time(h, m) for h in range(15, 21) for m in range(60)] + [dt.time(21, 0)]
        self.empty_day_df = pd.DataFrame(np.array([[np.nan for c in raw_df.columns] for t in self.time_range]), columns=raw_df.columns)

        groups = raw_df.groupby(["Date", "Ticker"])
        print("Cleaning data:")
        df_list = Parallel(n_jobs=os.cpu_count()-2)(delayed(lambda g: self._clean_day_data(g, max_missing))(g) for g in tqdm(groups))
        df_list = list(filter(lambda x: x is not None, df_list)) #Drop any day/ticker combos with too much missing data
        print("Joining all dataframes")
        full_df = pd.concat(df_list)

        if save_name is not None:
            print("Saving to file:", save_name)
            full_df.to_csv(save_name, index=False) #Use index=False to avoid saving multi-index (since columns already exist there)

        return full_df


    def construct_avg_df(self, save_name=None):

        if self.cleaned_df is None:
            print("Loading cleaned data")
            df = self.load_data(raw=False)

        groups = df.groupby(["Date", "Time"])
        calc_avg = lambda dt, df_subset: [*dt, *df_subset[["c", "v"]].mean().values] #Use splatting to unpack (date, time) tuple and average c, v values
        print("Calculating averages")
        averages_list = Parallel(n_jobs=os.cpu_count()-2)(delayed(calc_avg)(*g) for g in tqdm(groups)) #Parallelize averaging calc
        print("Constructing avg_df")
        self.avg_df = pd.DataFrame(averages_list, columns=["Date", "Time", "c_avg", "v_avg"]) #Convert to dataframe

        if save_name is not None:
            print("Saving to file:", save_name)
            self.avg_df.to_csv(save_name, index=False)
        
        return self.avg_df
        

    def create_data_array(self, save_name=None, dtype=np.float32):

        #Load data if not provided
        if self.cleaned_df is None:
            print("Loading cleaned data")
            df = self.load_data(raw=False)
        if self.avg_df is None:
            print("Loading averaged data")
            avg_df = self.load_avg_df()

        groups = df.groupby(["Date", "Ticker"])
        # groups = list(groups)[:10] #Use subset for quicker testing
        data_list = [[df_subset.c.values, df_subset.v.values, avg_df.c_avg[avg_df.Date == date].values] for (date, ticker), df_subset in tqdm(groups)] #Serial seems to be about 30% faster than parallel in this particular case
        data_array = np.array(data_list, dtype=dtype)

        if save_name is not None:
            print("Saving data array to:", save_name)
            np.save(self.data_dir / save_name, data_array)

        return data_array


    # def label_data(self, )



    # def fetch_new_data(self, save=False) -> pd.DataFrame:


    def convert_old_data(self, path) -> None:

        old_df = pd.read_csv(path)
        # old_df = old_df[:10000] #Uncomment for quicker testing
        old_df.rename({"Unnamed: 0" : "Date", "Unnamed: 1" : "Ticker", "Unnamed: 2" : "Time"}, axis=1, inplace=True)    
        groups = old_df.set_index("Date").groupby(lambda date_str: date_str[:-3]) #Group df entries by year and month
        for date_str, df in groups:
            df.to_csv(self.data_dir / (date_str + ".csv"))

        return


if __name__ == "__main__":

    import sys, os, pathlib

    DATA_DIR = pathlib.Path("/home/scottd/Dropbox/Other-Programming/SP500-Trading/data/")
    data = TickerData(DATA_DIR)

    print(TickerData.create_time_range())

    # data.convert_old_data("/home/scottd/SP500-Ticker-Data/full-data.csv.gz")
    # df = data.load_raw_data()
    # print(type(df.Time[0]))
    # print(type(df.Date[0]))

    # df = data.clean_data(10, save_name=data.data_dir / "cleaned-data.csv")
    # print(df.head())
    # print(df.index)
    # print(len(df) / 391)
    
    # print("Loading cleaned data")
    # data.load_data(raw=False)
    # print(len(df) / 391, len(df), df.count(), sep="\n")
    # avg_df = data.construct_avg_df(df=df, save_name=data.data_dir / "avg-data.csv")

    # print("Creating data array")
    # data.create_data_array(save_name="data-array.npy")
    