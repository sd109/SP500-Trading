
import os, sys, time, finnhub
from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset

class TickerData:

    """
    Class for collecting, cleaning and working with minute by minute ticker data from the S&P 500 companies.
    """
    
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
            self.load_data(raw=False)
        if self.aug_df is None:
            print("Loading extra data")
            self.load_aug_df()

        groups = self.cleaned_df.groupby(["Date", "Ticker"]) 
        # groups = list(groups)[:10] #Uncomment to use subset for quicker testing
        data_list = []
        date_list = []
        for (date, ticker), df_subset in tqdm(groups): #Serial seems to be about 30% faster than parallel in this particular case
            idxs = self.aug_df.Date == date
            data_list.append([df_subset.c.values, df_subset.v.values, self.aug_df.c_avg[idxs].values, self.aug_df.v_avg[idxs].values, self.aug_df.c_vwa[idxs].values])
            date_list.append(str(date))

        data_array = np.array(data_list, dtype=dtype)
        date_array = np.array(date_list)

        if save_name is not None:
            print("Saving data array to:", save_name)
            np.savez(self.data_dir / save_name, dates=date_array, data=data_array)

        return date_array, data_array



    def create_dataset(self, cutoff_date, t0, t1=dt.time(20, 55), N_classes=2, pytorch=False):

        """Performs data labelling and splits into training and testing sets based on cutoff date for input to classifier models.
        (returns torch.Dataset if pytorch=True otherwise returns numpy arrays in the order X_train, X_test, Y_train, ..., ratios_test)
        """

        dates, full_data = np.load(self.data_dir / "data-array.npz").values()
        dates = np.array(list(map(dt.date.fromisoformat, dates)))
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
        
        # Check class labels were assigned evenly (or approx evenly if N_classes=2)
        for y in Y.unique():
            print("Class value:", y, "\t Instance count:", sum(Y.flatten() == y))

        #Split into training and test sets based on cutoff date (using a cutoff date avoids data leakage into the future for prices that haven't happened yet)
        train_date_idxs = dates <= cutoff_date
        test_date_idxs = dates > cutoff_date
        print("Number of items in training set =", sum(train_date_idxs))
        print("Number of items in test set =", sum(test_date_idxs))

        X_train, X_test = X[train_date_idxs, :, :],  X[test_date_idxs, :, :]
        Y_train, Y_test = Y[train_date_idxs], Y[test_date_idxs]
        ratios_train, ratios_test = ratios[train_date_idxs], ratios[test_date_idxs]

        if pytorch:
            return TensorDataset(X_train, Y_train, ratios_train), TensorDataset(X_test, Y_test, ratios_test)
        else:
            return X_train.numpy(), X_test.numpy(), Y_train.numpy(), Y_test.numpy(), ratios_train.numpy(), ratios_test.numpy()



    def fetch_new_data(self, max_days=370, start_date=None, max_fail_count=10):

        """Method for fetching new data using the Finnhub API and adding it to existing raw csv data.
        (Includes various failsafe checks and network error catching functionality.)
        """

        if len(sys.argv) != 2:
            raise Exception("Must provide Finnhub API key as only command line arg when fetching new data")

        #Fetch list of S&P 500 constituents from wikipedia table
        TICKERS = pd.read_html("https://en.wikipedia.org/wiki/List_of_S&P_500_companies")[0].Symbol.to_list() #[:10] #Use slice for faster testing
        print("Tickers found from wiki:", TICKERS) #Print these to make sure wiki format hasn't changed

        if start_date == None:
            start_date = dt.date.today().replace(day=1) - dt.timedelta(days=1) #Start on the last day of the previous month and work backwards
        elif not isinstance(start_date, dt.date):
            start_date = dt.date.fromisoformat(start_date) #Convert to datetime instance

        #Initialize some useful variables
        tmp_data = {"Date":[], "Ticker":[], "Time":[], "c":[], "v":[]} #Use tmp dict of arrays for efficiency then convert to pd.DataFrame for saving to csv
        fail_counter = 0 #Keep track of errors in data retrieval
        day_counter = 0
        current_month = start_date.month

        #Initialize Finnhub instance with supplied API token
        client = finnhub.Client(api_key=sys.argv[1])
        # client.DEFAULT_TIMEOUT = 30 #Update network timeout

        while fail_counter < max_fail_count:

            #Get the date for which we're about to fetch data
            date = start_date - dt.timedelta(days=day_counter)
            if date.weekday() == 5 or date.weekday() == 6: #Skip weekends
                day_counter += 1
                continue

            #Check if updated date has changed month, if so then save existing data to disk
            if date.month != current_month:
                print("\nSaving data to:", filename, "\n----------------------------------------\n\n")
                df = pd.DataFrame(tmp_data) #Dict to dataframe
                df.to_csv(filename, index=False) #Write to disk
                current_month = date.month #Reset current month
                for k in tmp_data.keys(): #Empty tmp data
                    tmp_data[k] = []
                

            print(f"\nStarting update process for date: {date}\n----------------------------------------")
            filename = self.data_dir / f"raw/{date.isoformat()[:-3]}.csv" #Slice day out of date string

            #Check for existing data
            if os.path.isfile(filename):
                print(f"Data already exists for {date.isoformat()[:-3]}\nMove or delete existing data if you want to fetch it again.")
                break

            #Otherwise start fetching
            for (ticker_counter, T) in enumerate(TICKERS):

                print(f"Fetching data for ticker ({ticker_counter+1}) \t{T}")

                #Otherwise, query finnhub for data and catch any connection errors
                t1 = int(dt.datetime.combine(date, dt.time(14, 30)).timestamp())
                t2 = int(dt.datetime.combine(date, dt.time(21, 0)).timestamp())
                try: 
                    res = client.stock_candles(symbol=T, resolution=1, _from=t1, to=t2) #Fetch data
                    #Store it in tmp dict for conversion to DataFrame later
                    tmp_data["c"] += res["c"]
                    tmp_data["v"] += res["v"]
                    tmp_data["Date"] += [dt.date.fromtimestamp(t) for t in res["t"]]
                    tmp_data["Time"] += [dt.datetime.fromtimestamp(t).time() for t in res["t"]] #dt.time has no fromtimestamp constructor
                    tmp_data["Ticker"] += [T for _ in res["c"]]

                except:
                    print(f"Error encountered in API call for ticker {T} on date {date}") #How do we make it retry this ticker if it fails?
                    fail_count += 1
                    continue

                if res["s"] == "no_data": #Finnhub api returned no data for that day
                    print("API returned no data for", T, date)
                    fail_count += 1
               
                else:
                    fail_count = 0 #Reset fail count

                time.sleep(1) #Avoid hitting finnhub's api call freq limit

            day_counter += 1 #Update counter once we've fetched data for all tickers on that date

            #Save data here for quick testing purposes
            # print("\nSaving data to:", filename, "\n\n")
            # df = pd.DataFrame(tmp_data) #Dict to dataframe
            # df.to_csv(f"{filename}-{random.random()}", index=False) #Write to disk

            #Enforce upper limit on number of days to trace back while fetching data
            if day_counter >= max_days:
                print(f"Max days reached ({max_days} days)\nQuitting data gathering loop.")
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

    # DATA_DIR = pathlib.Path("/home/scottd/Dropbox/Other-Programming/SP500-Trading/data/")
    # ticker_data = TickerData(DATA_DIR)

    # print(TickerData.create_time_range())

    # ticker_data.convert_old_data("/home/scottd/SP500-Ticker-Data/full-data.csv.gz")
    # df = ticker_data.load_raw_data()

    # df = ticker_data.clean_data(10, save_name=ticker_data.data_dir / "cleaned-data.csv")
    # print(len(df) / 391, len(df), df.count(), sep="\n")
    
    # aug_df = data.construct_augmented_df(save_name=ticker_data.data_dir / "aug-data.csv")

    # print("Creating data array")
    # ticker_data.create_data_array(save_name="data-array.npy") 

    # ds = ticker_data.create_torch_dataset(t0=dt.time(19, 0), N_classes=10)

    DATA_DIR = pathlib.Path("/home/scottd/Dropbox/Other-Programming/SP500-test/")
    ticker_data = TickerData(DATA_DIR)
    ticker_data.fetch_new_data(start_date="2022-01-31", max_days=300, max_fail_count=10) #Need > 31 max_days to ensure some saving is done
