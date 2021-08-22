import os
import string

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas import DataFrame

from source.fetch.fetch import Dataset


class Candlesticks:
    """This is a class used to create the dataset
    """
    def __init__(self):
        """Constructor method
        """
        self.__btc_ohlcv_path = self.__get_path_to_data("data/btc_ohlcv.csv")
        return

    def get_btc_ohlcv_path(self):
        """Returns the absolute path to the btc_ohlcv.csv file.

        :return: the absolute path to btc_ohlcv.csv.
        :rtype: string
        """
        return self.__btc_ohlcv_path

    @staticmethod
    def __get_path_to_data(rel_path: string):
        """Returns the absolute path to the current file.

        :param rel_path: relative path to the current file
        :type rel_path: string
        :return: the absolute path to current file.
        :rtype: string
        """
        script_dir = os.path.dirname(__file__)
        abs_file_path = os.path.join(script_dir, rel_path)
        return abs_file_path

    @staticmethod
    def display_candlesticks_chart(dataframe: DataFrame):
        """Display a candlesticks chart from csv file with OHLCV data.

        :param dataframe: absolute path to the csv file
        :type dataframe: pandas.DataFrame
        """
        # Create subplots and mention plot grid size
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, subplot_titles=('OHLC', 'Volume'),
                            row_width=[0.2, 0.7])
        # Plot OHLC on 1st row
        fig.add_trace(go.Candlestick(x=dataframe.index, open=dataframe["open"], high=dataframe["high"],
                                     low=dataframe["low"], close=dataframe["close"], name="OHLC"), row=1, col=1)
        # Bar trace for volumes on 2nd row without legend
        fig.add_trace(go.Bar(x=dataframe.index, y=dataframe['Volume'], showlegend=False), row=2, col=1)
        fig.update(layout_xaxis_rangeslider_visible=False)
        # Display chart
        fig.show()
        return


def main():
    dataset = Dataset()
    dataframe_btc = dataset.get_btc_data()
    candlesticks = Candlesticks()
    candlesticks.display_candlesticks_chart(dataframe_btc)


if __name__ == '__main__':
    main()
