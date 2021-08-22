import investpy
import os
import string
from pathlib import Path

import pandas
import pandas as pd
import quandl


class Dataset:
    """This is a class used to create the dataset
    """

    def __init__(self):
        """Constructor method
        """
        self.__btc_csv_path = self.__get_absolut_path_to_data("data/BNC_BLX.csv")
        self.__future_btc_csv_path = self.__get_absolut_path_to_data("data/CME_DL_BTC1!.csv")
        self.__blockchain_btc_csv_path = self.__get_absolut_path_to_data("data/blockchain.csv")
        self.__data_csv_path = self.__get_absolut_path_to_data("data/dataset.csv")
        self.__data_csv_with_future_path = self.__get_absolut_path_to_data("data/dataset_with_future.csv")
        self.__btc_data = self.fetch_btc_data()
        return

    def fetch_btc_data(self):
        """Store historical data about Bitcoin (BNC_BLX) in a pandas dataframe, from 03/06/2012 to
        15/07/2021. (time,open,high,low,close,EMA,MA,Volume,Volume MA,Basis,Upper,Lower,RSI)

        :return: BTC data (time,open,high,low,close,EMA,MA,Volume,Volume MA,Basis,Upper,Lower,RSI) 03/06/2012-15/07/2021
        :rtype: pandas.DataFrame
        """
        btc_abs_path = self.__get_absolut_path_to_data(self.__btc_csv_path)
        btc_data = pd.read_csv(btc_abs_path)
        btc_data = btc_data.dropna()
        btc_data['Position'] = btc_data['close'].diff().fillna(0.0)
        btc_data.loc[btc_data['Position'] > 0.0, 'Position'] = True
        btc_data.loc[btc_data['Position'] <= 0.0, 'Position'] = False
        btc_data['time'] = pd.to_datetime(btc_data['time'], unit='s')
        btc_data.insert(0, 'date', btc_data["time"])
        btc_data = btc_data.set_index('time')
        return btc_data

    def fetch_future_btc_data(self):
        """Store future (CME) from 17/12/2017 to 15/07/2021.

        :return: Future CME BTC data (close) 17/12/2017-15/07/2021
        :rtype: pandas.DataFrame
        """
        abs_path = self.__get_absolut_path_to_data(self.__future_btc_csv_path)
        future_btc_data = pd.read_csv(abs_path)
        future_btc_data = future_btc_data.drop(
            ["open", "high", "low", "MA", "Volume", "Volume MA", "Basis", "Upper", "Lower", "RSI"], axis=1)
        future_btc_data = future_btc_data.dropna()
        future_btc_data['time'] = pd.to_datetime(future_btc_data['time'], unit='s').dt.date
        future_btc_data = future_btc_data.rename(columns={"close": "Close-future"}, errors="raise")
        future_btc_data = future_btc_data.set_index("time")
        index = pd.date_range('2017-12-17', '2021-07-15')
        future_btc_data.index = pd.DatetimeIndex(future_btc_data.index)
        future_btc_data = future_btc_data.reindex(index)
        future_btc_data.fillna(method='ffill')
        return future_btc_data

    @staticmethod
    def fetch_blockchain_btc_data():
        """Store historical data about Blockchain data from Quandl's API which use Blockchain.com as source.
        DIFF: Bitcoin Difficulty, A relative measure of how difficult it is to find a new block. The difficulty is
        adjusted periodically as a function of how much hashing power has been deployed by the network of miners.
        MWNTD: Bitcoin My Wallet Number of Transaction Per Day
        MWTRV: Bitcoin My Wallet Transaction Volume, 24hr Transaction Volume of our web wallet service.
        MIREV: Bitcoin Miners Revenue, total value of coinbase block rewards and transaction fees paid to miners.
        HRATE: Bitcoin Hash Rate, the estimated number of tera hashes per second (trillions of hashes per second) the
        Bitcoin network is performing.
        BLCHS: Bitcoin api.blockchain Size, the total size of all block headers and transactions.
        Not including database indexes.
        AVBLS: Bitcoin Average Block Size, the average block size in MB.
        BCDDY: Bitcoin Days Destroyed (Minimum Age 1 Year). A non-cumulative version of Bitcoin Days Destroyed.
        Bitcoin Days Destroyed is a measure of the transaction volume of Bitcoin. If someone has 100 BTC that they
        received a week ago and they spend it then 700 bitcoin days have been destroyed. If they take those 100BTC and
        send them to several addresses and then spend them then although the total transaction volume could be
        arbitrarily large the number of bitcoin days destroyed is still 700.
        ATRCT: Bitcoin Median Transaction Confirmation Time, the median time for a transaction to be accepted into a mined
        block and added to the public ledger (note: only includes transactions with miner fees).
        TOTBC: Total Bitcoins, the total number of bitcoins that have already been mined; in other words, the current
        supply of bitcoins on the network.
        MKTCP: Bitcoin Market Capitalization, the total USD value of bitcoin supply in circulation, as calculated by the
        daily average market price across major exchanges.

        :return: blockchain information from 03/06/2012 to 15/07/2021.
        :rtype: pandas.DataFrame
        """
        endpoint_quandl = "BCHAIN/{}"
        api_quandl = "vHzZzEA5BpYFshNB7YXs"
        endpoints = ("DIFF", "MWNTD", "MWTRV", "MIREV", "HRATE", "BLCHS", "AVBLS", "BCDDY", "ATRCT", "TOTBC", "MKTCP")
        dataframes = []
        for endpoint in endpoints:
            data = quandl.get(endpoint_quandl.format(endpoint),
                              start_date="2012-06-03",
                              end_date="2021-07-15",
                              authtoken=api_quandl)
            data = data.rename(columns={"Value": f"Value-{endpoint}"})
            dataframes.append(data)
        return pd.concat(dataframes, axis=1)

    @staticmethod
    def fix_blockchains_dataframe(dataframe: pandas.DataFrame):
        index = pd.date_range('2012-06-03', '2021-07-15')
        dataframe.index = pd.DatetimeIndex(dataframe.index)
        dataframe = dataframe.reindex(index)
        column_name = dataframe.columns[1]
        dataframe[column_name] = dataframe.iloc[:, 1].ffill().add(dataframe.iloc[:, 1].bfill()).div(2)
        return dataframe

    @staticmethod
    def fetch_data_by_symbol():
        """Store historical data about Crude Oil WTI Futures (exchange ICE) and AUX/USD (exchange NASDAQ) in pandas
        dataframe. Fake data is created for the weekend with the previous close value of Friday.

        :return: Gold and crude oil close data from 03/06/2012 to 15/07/2021.
        :rtype: pandas.DataFrame
        """
        symbols = ["crude oil", "xau"]
        dataframes = []
        for symbol in symbols:
            search_result = investpy.search_quotes(text=symbol, n_results=1)
            dataframe = search_result.retrieve_historical_data(from_date='03/06/2012', to_date='15/07/2021')
            dataframe = dataframe.drop(["Open", "High", "Low", "Volume", "Change Pct"], axis=1)
            dataframe = dataframe.dropna()
            index_date = pd.date_range("2012-06-03", "2021-07-15")
            dataframe = dataframe.reindex(index_date, fill_value=None)
            dataframe = dataframe.fillna(method='ffill')
            dataframe = dataframe.fillna(method='bfill')
            dataframe = dataframe.rename(columns={'Close': f'Close-{symbol}'})
            dataframes.append(dataframe)
        return pd.concat(dataframes, axis=1)

    def get_csv_path(self):
        """Returns the absolute path to the dataset.csv file.

        :return: the absolute path to dataset.csv.
        :rtype: string
        """
        return self.__data_csv_path

    def get_csv_with_future_path(self):
        """Returns the absolute path to the dataset_with_future.csv file.

        :return: the absolute path to dataset_with_future.csv.
        :rtype: string
        """
        return self.__data_csv_with_future_path

    def get_btc_data(self):
        """Returns OHLCV data about Bitcoin, daily interval.

        :return: Daily data about Bitcoin with Open, High, Low, Close and Volume values.
        :rtype: pandas.DataFrame
        """
        return self.__btc_data

    @staticmethod
    def __get_absolut_path_to_data(rel_path: string):
        """Returns the absolute path to the current file.

        :param rel_path: relative path to the current file
        :type rel_path: string
        :return: the absolute path to current file.
        :rtype: string
        """
        script_dir = os.path.dirname(__file__)
        abs_file_path = os.path.join(script_dir, rel_path)
        return abs_file_path

    def create_dataset(self):
        """Create a csv file with data from all the biggest exchanges
        """
        csv_file = Path(self.get_csv_path())
        csv_future_file = Path(self.get_csv_with_future_path())
        if not csv_file.is_file() and not csv_future_file.is_file():
            dataframe = pd.concat([self.get_btc_data(),
                                   self.fetch_data_by_symbol(),
                                   self.fetch_blockchain_btc_data()], axis=1)
            dataframe = dataframe.fillna("?")
            dataframe.to_csv(self.get_csv_path(), index=False)
            dataframe = dataframe.loc[dataframe["date"] >= "2017-12-17"]
            dataframe_with_future = pd.concat([dataframe, self.fetch_future_btc_data()], axis=1)
            dataframe_with_future = dataframe_with_future.fillna("?")
            dataframe_with_future.to_csv(self.get_csv_with_future_path(), index=False)
        return


def main():
    dataset = Dataset()
    dataset.create_dataset()


if __name__ == '__main__':
    main()
