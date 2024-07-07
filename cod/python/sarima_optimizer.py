#Carga de libraries
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
import itertools

class sarima_optimizer:
    def __init__(self, variable_objetivo, objeto_serietemporal, start_date_test, end_date_test):
        print("###################################################################")
        print("Creando estructuras basicas para el objeto SARIMA")
        self.variable_objetivo = variable_objetivo
        self.objeto_serietemporal = objeto_serietemporal
        self.start_date_test = start_date_test
        self.end_date_test = end_date_test

        self.results_df = None
        self.final_sarima_model_results = None
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
        print("Optimizando modelo SARIMA")

        # Define the range of p, d, q, P, D, Q, and seasonal period (m) values
        p_values = range(0, 3)  # Replace with your desired range
        d_values = range(0, 2)  # Replace with your desired range
        q_values = range(0, 3)  # Replace with your desired range
        P_values = range(0, 3)  # Replace with your desired range
        D_values = range(0, 2)  # Replace with your desired range
        Q_values = range(0, 3)  # Replace with your desired range
        m_values = [5]  # Replace with your seasonal period

        # Create a list with all possible combinations of SARIMA parameters
        sarima_combinations = list(itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, m_values))

        # Initialize a DataFrame to store results
        # Initialize variables to store best parameters and minimum AIC
        best_params = None
        best_aic = float("inf")
        bests_rmse = float("inf")

        p_list = []
        d_list = []
        q_list = []
        P_list = []
        D_list = []
        Q_list = []
        m_list = []
        aic_list = []
        rmse_list = []
        mape_list = []
        mae_list  = []

        # Iterate through all combinations
        for sarima_params in sarima_combinations:
            try:
              print(f'Construyendo y evaluando SARIMA: {sarima_params}')
              # Fit SARIMA model
              model = sm.tsa.SARIMAX(self.train[self.variable_objetivo], order=(sarima_params[0], sarima_params[1], sarima_params[2]),
                                seasonal_order=(sarima_params[3], sarima_params[4], sarima_params[5], sarima_params[6]))
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
              p_list.append(sarima_params[0])
              d_list.append(sarima_params[1])
              q_list.append(sarima_params[2])
              P_list.append(sarima_params[3])
              D_list.append(sarima_params[4])
              Q_list.append(sarima_params[5])
              m_list.append(sarima_params[6])

            except Exception as e:
              print('Hubo un error!')
              continue

        # Initialize a DataFrame to store results
        self.results_df =  pd.DataFrame({'p': p_list,
                                    'd': d_list,
                                    'q': q_list,
                                    'P': P_list,
                                    'D': D_list,
                                    'Q': Q_list,
                                    'm': m_list,
                                    'AIC': aic_list,
                                    'RMSE':rmse_list,
                                    'MAPE':mape_list,
                                    'MAE':mae_list})



        # Sort results by AIC in ascending order
        self.results_df = self.results_df.sort_values(by='RMSE')

        return(self.results_df)

    def get_optimal_SARIMA(self):

      # Select the best parameters
      self.best_params_sarima = self.results_df.iloc[0][['p', 'd', 'q', 'P', 'D', 'Q', 'm']]

      # Print the best parameters
      print(f"\nBest SARIMA parameters: {self.best_params_sarima}")

      # Fit the final SARIMA model with the best parameters
      final_model_sarima = sm.tsa.SARIMAX(self.train[self.variable_objetivo],
                                          order=(self.best_params_sarima['p'], self.best_params_sarima['d'], self.best_params_sarima['q']),
                                     seasonal_order=(self.best_params_sarima['P'], self.best_params_sarima['D'], self.best_params_sarima['Q'],
                                                     self.best_params_sarima['m']))
      self.final_results_sarima = final_model_sarima.fit()

      # Print the summary of the final SARIMA model
      print(self.final_results_sarima.summary())

      return(self.final_results_sarima)

    def forecast(self):
      predictions = self.final_results_sarima.predict(start=len(self.train), end=len(self.train)+len(self.test)-1)
      return(predictions)