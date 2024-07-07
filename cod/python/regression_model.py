#Carga de libraries
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class regression_model:
    def __init__(self, variable_objetivo, objeto_serietemporal, start_test_date, end_test_date):
        print("Construyendo modelo de regresión basado en tendencia temporal")
        self.variable_objetivo = variable_objetivo
        self.objeto_serietemporal = objeto_serietemporal
        self.start_date_test = start_test_date
        self.end_date = end_test_date
        self.test = objeto_serietemporal.data
        self.test = self.test[self.test.index >= self.start_date_test]

        self.train = self.objeto_serietemporal.data
        self.train = self.train[self.train.index < self.start_date_test]

        self.data_adjclose = self.train[self.variable_objetivo]

        # Detrend
        self.timeTrend = np.linspace(1, len(self.data_adjclose), len(self.data_adjclose))
        self.timeTrend = sm.add_constant(self.timeTrend)

        self.timeTrend_test = np.linspace(len(self.data_adjclose), len(self.test), len(self.test))
        self.timeTrend_test = sm.add_constant(self.timeTrend_test)

        # Fit OLS
        model = sm.OLS(self.data_adjclose, self.timeTrend)
        self.fit_g = model.fit()
        print(self.fit_g.summary())

    def plot_residuals(self):
        res = self.fit_g.resid
        res.plot(linewidth=1.3, xlabel="Year", ylabel="Residuals")
        plt.show()

    def forecast(self):
        predictions = self.fit_g.predict(self.timeTrend_test)
        return(predictions)

    def get_metrics(self):
        print("###################################################################")
        print("Obteniendo metricas")
        predictions = self.forecast()
        rmse = np.sqrt(mean_squared_error(self.test[self.variable_objetivo], predictions))
        mape = mean_absolute_percentage_error(self.test[self.variable_objetivo], predictions)
        mae = mean_absolute_error(self.test[self.variable_objetivo], predictions)
        results_df =  pd.DataFrame({'RMSE':[rmse],
                                    'MAPE':[mape],
                                    'MAE':[mae]})
        return(results_df)