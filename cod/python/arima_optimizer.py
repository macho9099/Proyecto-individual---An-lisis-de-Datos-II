#Carga de libraries
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
import itertools


class arima_optimizer:
    def __init__(self, variable_objetivo, objeto_serietemporal, start_date_test, end_date_test):
        print("###################################################################")
        print("Creando estructuras basicas para el objeto ARIMA")
        self.variable_objetivo = variable_objetivo
        self.objeto_serietemporal = objeto_serietemporal
        self.start_date_test = start_date_test
        self.end_date_test = end_date_test

        self.results_df = None
        self.final_arima_model_results = None
        self.best_params = None

        self.test = self.objeto_serietemporal.data
        self.test = self.test[self.test.index >= self.start_date_test]

        self.train = self.objeto_serietemporal.data
        self.train = self.train[self.train.index < self.start_date_test]
        print("###################################################################")
        print('Especificaciones de el set de entrenamiento')
        print(self.train.info())

        print("###################################################################")
        print('Especificaciones de el set de prueba')
        print(self.test.info())

    def optimize(self):
        print("###################################################################")
        print("Optimizando modelo ARIMA")

        # Define the range of p, d, and q values
        p_values = range(0, 3)  # Replace with your desired range
        d_values = range(0, 3)  # Replace with your desired range
        q_values = range(0, 3)  # Replace with your desired range

        # Create a list with all possible combinations of p, d, and q
        pdq_combinations = list(itertools.product(p_values, d_values, q_values))
        print(f"Probando este número de combinaciones de hiperparámetros: {len(pdq_combinations)}")

        # Initialize variables to store best parameters and minimum AIC
        best_params = None
        best_aic = float("inf")
        bests_rmse = float("inf")

        p_list = []
        d_list = []
        q_list = []
        aic_list = []
        rmse_list = []
        mape_list = []
        mae_list  = []

        # Iterate through all combinations
        for pdq in pdq_combinations:
          print(f'Construyendo y evaluando ARIMA: {pdq}')
          #try:
          # Fit ARIMA model
          model = sm.tsa.ARIMA(self.train[self.variable_objetivo], order=pdq)
          results = model.fit()

          # Check AIC (Akaike Information Criterion)
          aic_list.append(results.aic)

          # Make predictions on the testing set
          predictions = results.predict(start=len(self.train), end=len(self.train)+len(self.test)-1)

          # Calculate RMSE and MAPE
          rmse = np.sqrt(mean_squared_error(self.test[self.variable_objetivo], predictions))
          mape = mean_absolute_percentage_error(self.test[self.variable_objetivo], predictions)
          mae = mean_absolute_error(self.test[self.variable_objetivo], predictions)

          #Saving results in lists
          rmse_list.append(rmse)
          mape_list.append(mape)
          mae_list.append(mae)
          p_list.append(pdq[0])
          d_list.append(pdq[1])
          q_list.append(pdq[2])

        # Initialize a DataFrame to store results
        self.results_df =  pd.DataFrame({'p': p_list,
                                    'd': d_list,
                                    'q': q_list,
                                    'AIC': aic_list,
                                    'RMSE':rmse_list,
                                    'MAPE':mape_list,
                                    'MAE':mae_list})

          #except Exception as e:
            #print('Hubo un error!')
            #continue

        # Sort results by AIC in ascending order
        self.results_df = self.results_df.sort_values(by='RMSE')

        return(self.results_df)

    def get_optimal_ARIMA(self):
      # Fit the final model with the best parameters
      self.best_params = (self.results_df.iloc[0]['p'], self.results_df.iloc[0]['d'], self.results_df.iloc[0]['q'])
      self.final_arima_model = sm.tsa.ARIMA(self.train[self.variable_objetivo], order= self.best_params)
      self.final_arima_model_results = self.final_arima_model.fit()

      # Print the summary of the final model
      print(self.final_arima_model_results.summary())

      return(self.final_arima_model)

    def forecast(self):
      predictions = self.final_arima_model_results.predict(start=len(self.train), end=len(self.train)+len(self.test)-1)
      return(predictions)

    def show_residuals(self):
      # Standardized Residuals for Optimal ARIMA Model
      fit_res = self.final_arima_model_results.resid
      fit_res_sta = fit_res / np.std(fit_res, ddof=1)
      plt.plot(fit_res_sta)
      plt.ylabel("Standardized Residuals")
      plt.show()