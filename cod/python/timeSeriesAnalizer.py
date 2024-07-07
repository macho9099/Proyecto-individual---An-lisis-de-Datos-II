#Carga de libraries
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.seasonal import seasonal_decompose

class timeSeriesAnalizer:
    def __init__(self, ticker, start, end):
        print("###################################################################")
        print(f"Creando objeto para la serie de {ticker}")
        self.ticker = ticker
        self.start = start
        self.end = end
        #Extrayendo datos desde yahoo finance
        self.data = yf.download(tickers=ticker, start=start, end=end)
        #Tomando el precio de cierre ajustado
        self.data_adjclose = self.data[["Adj Close"]]
        self.data['Close_Diff'] = self.data['Adj Close'].diff()
        self.data.dropna(inplace=True)

        #Calculando estadisticos
        self.mean = self.data_adjclose.mean()
        self.variance = self.data_adjclose.var()
        self.skewness = skew(self.data_adjclose)
        self.kurt = kurtosis(self.data_adjclose)

    def show_basic_statistics(self, difference=False):
        #Showing descriptive statistics
        print("############################################")
        print("Showing descriptive statistics!:")
        print(self.data.describe())
        if(difference==False):
            print("############################################")
            print("Showing Moment based statistics!:")
            print(f'Mean: {self.mean[0]}')
            print(f'Variance: {self.variance[0]}')
            print(f'Skewness: {self.skewness[0]}')
            print(f'Kurtosis: {self.kurt[0]}')
        else:
            print("############################################")
            print("Showing Moment based statistics!:")
            print(f'Mean: {self.data[["Close_Diff"]].mean()[0]}')
            print(f'Variance: {self.data[["Close_Diff"]].var()[0]}')
            print(f'Skewness: {skew(self.data[["Close_Diff"]])[0]}')
            print(f'Kurtosis: {kurtosis(self.data[["Close_Diff"]])[0]}')


    def show_distribution(self, difference=False):
        print("###################################################################")
        if(difference):
          print(f"Showing distribution of First Difference of {self.ticker}")
          fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))
          self.data[['Close_Diff']].plot(ax=ax1)
          ax1.set_title("First Difference Adj Price of "+ self.ticker)
          self.data[['Close_Diff']].hist(bins=10, ax=ax2)
          ax2.set_title("Histogram of First Difference Adj Price of "+ self.ticker)
          probplot(self.data['Close_Diff'], dist="norm", plot=plt)
          ax3.set_title('QQ-Plot');
        else:
          print(f"Showing distribution of {self.ticker}")
          fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))
          self.data_adjclose.plot(ax=ax1)
          ax1.set_title("Adj Price of "+ self.ticker)
          self.data_adjclose.hist(bins=10, ax=ax2)
          ax2.set_title("Histogram of Adj Price of "+ self.ticker)
          probplot(self.data_adjclose["Adj Close"], dist="norm", plot=plt)
          ax3.set_title('QQ-Plot');

    def show_acf_pacf(self, difference=False):
        if difference:
          print("###################################################################")
          print(f"Showing ACF and PACF of First Difference for {self.ticker}")
          fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
          plot_acf(self.data[['Close_Diff']], lags=20, ax=ax1)
          ax1.set_title('Autocorrelation Function of First Difference of '+ self.ticker)
          plot_pacf(self.data[['Close_Diff']], lags=20, ax=ax2)
          ax2.set_title('Partial Autocorrelation Function of First Difference of '+ self.ticker);
        else:
          print(f"Showing ACF and PACF of {self.ticker}")
          fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
          plot_acf(self.data_adjclose, lags=20, ax=ax1)
          ax1.set_title('Autocorrelation Function of '+ self.ticker)
          plot_pacf(self.data_adjclose, lags=20, ax=ax2)
          ax2.set_title('Partial Autocorrelation Function '+ self.ticker);

    def apply_adf_test(self, difference=False):
        print("###################################################################")
        if difference:
          print(f"Applying ADF Test to First Difference of {self.ticker}")
          series = self.data[["Close_Diff"]]
        else:
          print(f"Applying ADF Test to {self.ticker}")
          series = self.data_adjclose

        # ADF Test
        result = adfuller(series, autolag='AIC')
        print(f'ADF Statistic: {result[0]}')
        print(f'n_lags: {result[1]}')
        print(f'p-value: {result[1]}')
        for key, value in result[4].items():
          print('Critial Values:')
          print(f'   {key}, {value}')
        if(result[1] < 0.05):
          print("We Reject null hypothesis of non stationarity")
        else:
          print("We Fail to reject null hypothesis of non stationarity")


    def apply_STL_decompose(self, difference=False):
        print("###################################################################")
        print(f"Applying STL Decomposition to {self.ticker}")
        # Descomposición de la serie temporal
        if difference:
          series = self.data[["Close_Diff"]]
        else:
          series = self.data_adjclose
        decomposition = seasonal_decompose(series, model='additive', period=252)  # Usando datos diarios, period es aprox. un año
        fig = decomposition.plot()
        fig.axes[0].set_ylabel('Observado')
        fig.axes[1].set_ylabel('Tendencia')
        fig.axes[2].set_ylabel('Estacionalidad')
        fig.axes[3].set_ylabel('Residuo')
        fig.axes[0].set_xlabel('Fecha')
        fig.axes[1].set_xlabel('Fecha')
        fig.axes[2].set_xlabel('Fecha')
        fig.axes[3].set_xlabel('Fecha')
        plt.show()