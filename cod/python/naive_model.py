class naive_model:
    def __init__(self, variable_objetivo, objeto_serietemporal, start_test_date, end_test_date):
        print("Construyendo modelo naive")
        self.variable_objetivo = variable_objetivo
        self.objeto_serietemporal = objeto_serietemporal
        self.start_date_test = start_test_date
        self.end_date = end_test_date

        self.test = objeto_serietemporal.data
        self.test = self.test[self.test.index >= self.start_date_test]

        self.train = self.objeto_serietemporal.data
        self.train = self.train[self.train.index < self.start_date_test]
        self.data_adjclose = self.train[self.variable_objetivo]
        self.results_df = None

        # Calculo del promedio
        self.valor_naive = self.data_adjclose.iloc[-1]
        print(f'El modelo naive predice a un valor de {self.valor_naive}')

    def forecast(self):
        predictions = [self.valor_naive] * len(self.test)
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