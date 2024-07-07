#Carga de libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Implementar el método de forecast estacional naive
class naive_estacional_model:
    def __init__(self, variable_objetivo, objeto_serietemporal, start_test_date, end_test_date):
        print("Construyendo modelo naive estacional")
        self.variable_objetivo = variable_objetivo
        self.objeto_serietemporal = objeto_serietemporal
        self.start_date_test = start_test_date
        self.end_date = end_test_date
        self.train = objeto_serietemporal.data
        self.train = self.train[self.train.index < self.start_date_test]
        self.start_date_test = start_test_date
        self.end_date = end_test_date
        self.test = objeto_serietemporal.data
        self.test = self.test[self.test.index >= self.start_date_test]
        self.data_adjclose = self.train[self.variable_objetivo]
        self.results_df = None

    def get_metrics(self):
        print("###################################################################")
        print("Obteniendo metricas")
        predictions = self.forecast()
        rmse = np.sqrt(mean_squared_error(self.test[self.variable_objetivo], predictions))
        mape = mean_absolute_percentage_error(self.test[self.variable_objetivo], predictions)
        mae = mean_absolute_error(self.test[self.variable_objetivo], predictions)
        self.results_df =  pd.DataFrame({'RMSE':[rmse],
                                  'MAPE':[mape],
                                  'MAE':[mae]})
        return(self.results_df)

    def forecast(self, season_length=252):
        steps = len(self.test)
        forecast = []
        for i in range(steps):
          forecast.append(self.data_adjclose.iloc[-season_length + (i % season_length)])
        return forecast