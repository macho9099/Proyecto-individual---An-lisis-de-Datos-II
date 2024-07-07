from statsmodels.tsa.arima.model import ARIMA

class STL_model:
  def __init__(self, variable_objetivo, objeto_serietemporal, start_date_test, end_date_test):
    print("###################################################################")
    print("Creando estructuras basicas para el objeto STL")
    self.variable_objetivo = variable_objetivo
    self.objeto_serietemporal = objeto_serietemporal
    self.start_date_test = start_date_test
    self.end_date_test = end_date_test
    self.train = self.objeto_serietemporal.data
    self.train = self.train[self.train.index < self.start_date_test]
    self.test = self.objeto_serietemporal.data
    self.test = self.test[self.test.index >= self.start_date_test]
    self.adj_close = self.train[self.variable_objetivo]
    self.stlf = STLForecast(self.adj_close.to_numpy(), ARIMA, model_kwargs=dict(order=(1, 1, 0), trend="t"), period=252)
    self.stlf_res = self.stlf.fit()

    self.results_df = None

  def forecast(self):
    predictions = self.stlf_res.forecast(steps=len(self.test))
    return(predictions)

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
