
import os, sys, time, finnhub
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

class TickerData:

    def __init__(self, data_dir: Path) -> None:
        
        #Filepath to data directory
        if type(data_dir) == str:
            data_dir = Path(data_dir)
        self.data_dir = data_dir
        #Empty attributes which will be loaded as needed
        self.raw_df = None
        self.cleaned_df = None
        self.aug_df = None

        return
        

    def load_data(self, raw=False) -> pd.DataFrame:

        """Loads minute by minute ticker data from csv files and saves resulting pd.DataFrame as an instance attribute."""

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


    def load_aug_df(self):

        """Loads augmented data from 'aug-data.csv' and saves pd.DataFrame as model attribute."""        
        
        aug_df = pd.read_csv(self.data_dir / "aug-data.csv")
        aug_df.Time = aug_df.Time.map(dt.time.fromisoformat)
        aug_df.Date = aug_df.Date.map(dt.date.fromisoformat)
        self.aug_df = aug_df
        return aug_df


    def _clean_day_data(self, group, max_missing):
        
        """Internal method to simplify parallelization of data cleaning"""

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
    def create_time_range():
        
        """Convenience method for constructing an associated list of datetime.time instances which span trading hours.
        (Useful for filtering various timeseries which don't have an attached time column.)"""

        return np.array([dt.time(14, m) for m in range(30, 60)] + [dt.time(h, m) for h in range(15, 21) for m in range(60)] + [dt.time(21, 0)])


    def clean_data(self, max_missing: int, save_name=None) -> pd.DataFrame:

        """Main method for converting raw ticker data to clean & useable model data.
        Cleaning process involves padding data missing timestamps followed by (linear) interpolation of missing values then joining of all data into a single pd.DataFrame.
        """

        if self.raw_df is None: #Avoid reloading data if already loaded
            print("Loading raw data")
            raw_df = self.load_data(raw=True)

        #Save some attributes which will be used multiple times during cleaning loop (must be instance attributes to be accessable from _clean_day_data method)
        self.time_range = TickerData.create_time_range()
        self.empty_day_df = pd.DataFrame(np.array([[np.nan for c in raw_df.columns] for t in self.time_range]), columns=raw_df.columns)

        groups = raw_df.groupby(["Date", "Ticker"])
        print("Cleaning data:")
        df_list = Parallel(n_jobs=os.cpu_count()-2)(delayed(lambda g: self._clean_day_data(g, max_missing))(g) for g in tqdm(groups)) #Parallelize main cleaning step
        df_list = list(filter(lambda x: x is not None, df_list)) #Drop any day/ticker combos with too much missing data
        print("Joining all dataframes")
        full_df = pd.concat(df_list)

        #Save result to disk if save_name given
        if save_name is not None:
            print("Saving to file:", save_name)
            full_df.to_csv(save_name, index=False) #Use index=False to avoid saving multi-index (since columns already exist there)

        return full_df


    def construct_augmented_df(self, save_name=None):

        """Calculates a number of different quantities (which may be useful for model predictions) from the interpolated ticker data.
        Currently includes:
            - average minute by minute (closing) price and traded volume across all tickers
            - volume weighted average of minute by minute closing price across all tickers
        """

        if self.cleaned_df is None:
            print("Loading cleaned data")
            df = self.load_data(raw=False)

        #Define small function for parallel mapping
        def calculate_quantities(group): 
            (date, time), df_subset = group
            vol_weighted_avg_close = (df_subset.c * df_subset.v).mean() #Volume weighted avg of prices (might be a useful data input to a model I guess...?)
            return [date, time, *df_subset[["c", "v"]].mean().values, vol_weighted_avg_close] #Use splatting to unpack average c, v values
        
        groups = df.groupby(["Date", "Time"])
        print("Calculating quantities")
        averages_list = Parallel(n_jobs=os.cpu_count()-2)(delayed(calculate_quantities)(g) for g in tqdm(groups)) #Parallelize averaging calc
        print("Constructing df")
        self.aug_df = pd.DataFrame(averages_list, columns=["Date", "Time", "c_avg", "v_avg", "c_vwa"]) #Convert to dataframe

        if save_name is not None:
            print("Saving to file:", save_name)
            self.aug_df.to_csv(save_name, index=False)
        
        return self.aug_df
        

    def create_data_array(self, save_name=None, dtype=np.float32):

        """Combines cleaned & augmented pd.DataFrames into a single numpy array for faster loading in model notebooks."""

        #Load data if not provided
        if self.cleaned_df is None:
            print("Loading cleaned data")
            df = self.load_data(raw=False)
        if self.aug_df is None:
            print("Loading averaged data")
            aug_df = self.load_aug_df()

        groups = df.groupby(["Date", "Ticker"])
        # groups = list(groups)[:10] #Uncomment to use subset for quicker testing
        data_list = []
        for (date, ticker), df_subset in tqdm(groups): #Serial seems to be about 30% faster than parallel in this particular case
            idxs = aug_df.Date == date
            data_list.append([df_subset.c.values, df_subset.v.values, aug_df.c_avg[idxs].values, aug_df.v_avg[idxs].values, aug_df.c_vwa[idxs].values])

        data_array = np.array(data_list, dtype=dtype)

        if save_name is not None:
            print("Saving data array to:", save_name)
            np.save(self.data_dir / save_name, data_array)

        return data_array


    def create_torch_dataset(self, t0, t1=dt.time(20, 55), N_classes=2):

        """Performs data labelling and converts to torch.Dataset for input into pytorch classifier models."""

        full_data = np.load(self.data_dir / "data-array.npy")
        time_range = TickerData.create_time_range()
        t0_idx = np.where(time_range == t0)
        t1_idx = np.where(time_range == t1)

        X = torch.tensor(full_data[:, :, time_range <= t0])
        ratios = torch.tensor(full_data[:, 0, t1_idx] / full_data[:, 0, t0_idx]).flatten() #0th element of dim 2 is 'close' price

        if N_classes == 2: #Binary profit/loss classification
            Y = ratios > 1.0
        else: #Or create discrete distribution over quantiles
            quantiles = ratios.quantile(torch.linspace(0, 1, N_classes+1))
            Y = torch.zeros(len(ratios))
            for edge in quantiles[1:-1]:
                Y += (ratios > edge)
                # print(edge, sum(ratios >= edge))
        
        # Check class labels were assigned evenly
        for y in Y.unique():
            print("Class value:", y, "\t Instance count:", sum(Y.flatten() == y))

        return TensorDataset(X, Y, ratios)



    def fetch_new_data(self, max_days=370, save=False) -> pd.DataFrame:

        """Method for fetching new data using the Finnhub API and adding it to existing raw csv data.
        (Includes various failsafe checks and network error catching functionality.)
        """

        #Fetch list of S&P 500 constituents from wiki
        TICKERS = pd.read_html("https://en.wikipedia.org/wiki/List_of_S&P_500_companies")[0].Symbol.to_list()[0:1]
        print("Tickers found from wiki:", TICKERS) #Print these to make sure wiki format hasn't changed

        start_date = dt.date.today()
        if dt.datetime.now().time() < dt.time(21, 0): #If we wont get a full trading day today, start yesterday instead
            start_date -= dt.timedelta(days=1)

        #Fetch data a month at a time until with find a month that we already have data for


        for T in TICKERS:

            print(f"\nStarting update process for ticker {T}\n----------------------------------------")
            client = finnhub.Client(api_key=sys.argv[1])
            # client.DEFAULT_TIMEOUT = 30 #Update network timeout
            

            fail_count = 0 # param which will be set True if we fail to get data for 10 concurrent trading says (signifying that we're too far back in time for the api to give any more data)
            count = 0
            while fail_count <= 10: # Quit searching if we fail to get data for 10 concurrent trading says (signifying that we're too far back in time for the api to give any more data)

                date = start_date - dt.timedelta(days=count)
                if date.weekday() == 5 or date.weekday() == 6: #Skip weekends
                    count += 1
                    continue

                filename = data_dir + f"{T}/{T}--{date}.csv"

                #Check for existing data in data_dir and if there's none for this date then fetch some
                if not os.path.isfile(filename):

                    print("Fetching data for", T, date)

                    #Otherwise, query finnhub for data and catch any connection errors
                    t1 = int(dt.datetime.combine(date, dt.time(14, 30)).timestamp())
                    t2 = int(dt.datetime.combine(date, dt.time(21, 0)).timestamp())
                    try: 
                        res = client.stock_candles(symbol=T, resolution=1, _from=t1, to=t2)
                        count += 1 #Update counter once we have res for that date
                    except:
                        print(f"Error encountered in api call for ticker {T} on date {date} - retrying date")
                        # count += 1 #Uncomment to skip date instead
                        fail_count += 1
                        continue

                    #For testing
                    if fail_count > 5:
                        print(f"(Failed to retrieve data {fail_count} times for {T} on {date}")


                    if res["s"] == "no_data": #Finnhub api returned no data for that day
                        print("API returned no data for", T, date)
                        fail_count += 1
                        count += 1
                    
                    else:
                        fail_count = 0 #Reset fail count

                        df = pd.DataFrame(res)
                        df.set_index(df.t.map(lambda x: dt.datetime.fromtimestamp(x).time()), inplace=True)
                        df.index.name = date

                        #Drop s (status) column
                        df.drop("s", axis=1, inplace=True)

                        #Save trading day's data
                        if not os.path.isdir(data_dir + T):
                            print(f"No existing data found for ticker {T} - making new dir at path: {data_dir + T}")
                            os.mkdir(data_dir + T + "/")
                        
                        #Save data in ticker specific folder with name {T}--{date} (as csv for future portability)
                        df.to_csv(filename)

                    time.sleep(1) #Avoid hitting finnhub's api call freq limit

                else:
                    count += 1
                    # print("Data already found at", filename, "-- skipping")

                #Enforce upper limit on number of days to trace back
                if count > max_days:
                    print(f"Max days reached ({max_days}) for ticker {T} - moving to next ticker.")
                    break



    def convert_old_data(self, path) -> None:

        """Legacy method for converting old data to newer format."""

        old_df = pd.read_csv(path)
        # old_df = old_df[:10000] #Uncomment for quicker testing
        old_df.rename({"Unnamed: 0" : "Date", "Unnamed: 1" : "Ticker", "Unnamed: 2" : "Time"}, axis=1, inplace=True)    
        groups = old_df.set_index("Date").groupby(lambda date_str: date_str[:-3]) #Group df entries by year and month
        for date_str, df in groups:
            df.to_csv(self.data_dir / (date_str + ".csv"))

        return



# General testing & usage examples:

if __name__ == "__main__":

    import pathlib

    DATA_DIR = pathlib.Path("/home/scottd/Dropbox/Other-Programming/SP500-Trading/data/")
    ticker_data = TickerData(DATA_DIR)

    # print(TickerData.create_time_range())

    # ticker_data.convert_old_data("/home/scottd/SP500-Ticker-Data/full-data.csv.gz")
    # df = ticker_data.load_raw_data()

    # df = ticker_data.clean_data(10, save_name=ticker_data.data_dir / "cleaned-data.csv")
    # print(len(df) / 391, len(df), df.count(), sep="\n")
    
    # aug_df = data.construct_augmented_df(save_name=ticker_data.data_dir / "aug-data.csv")

    # print("Creating data array")
    # ticker_data.create_data_array(save_name="data-array.npy") 

    # ds = ticker_data.create_torch_dataset(t0=dt.time(19, 0), N_classes=10)

