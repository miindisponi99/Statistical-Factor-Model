import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yfinance as yf


class StockData:
    def __init__(self, start_date, end_date):
        self.index_ticker = "FTSEMIB.MI"
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = self.load_tickers()
        self.data = self.download_data(self.tickers)
        self.volume_data = self.download_volume_data(self.tickers)
        self.index_data = self.download_data(self.index_ticker)
        self.monthly_data = self.resample_data(self.data)
        self.monthly_volume_data = self.resample_data(self.volume_data)
        self.index_monthly_data = self.resample_data(self.index_data)
        self.returns = self.calculate_returns(self.monthly_data)
        self.index_returns = self.calculate_returns(self.index_monthly_data)

    def load_tickers(self):
        with open("Tickers.json", "r") as json_file:
            tickers_json = json.load(json_file)
        return tickers_json["tickers"]

    def download_data(self, tickers):
        return yf.download(
            tickers, start=self.start_date, end=self.end_date, progress=False
        )["Open"]

    def download_volume_data(self, tickers):
        return yf.download(
            tickers, start=self.start_date, end=self.end_date, progress=False
        )["Volume"]

    def resample_data(self, data):
        return data.resample("ME").first()

    def calculate_returns(self, data):
        return data.pct_change().dropna()


class CorrelationMatrix:
    def __init__(self, returns):
        self.returns = returns
        self.correlation_matrix = self.calculate_correlation_matrix()

    def calculate_correlation_matrix(self):
        return self.returns.corr()

    def plot_heatmap(self):
        plt.figure(figsize=(20, 16))
        sns.heatmap(
            self.correlation_matrix,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            fmt=".1f",
        )
        plt.title("Correlation Heatmap of Returns")
        plt.show()