from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
from itertools import product

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


def rmse(y_actual, y_predicted):
    rms = mean_squared_error(y_actual, y_predicted, squared=False)
    return rms

def mae(y_actual, y_predicted):
    rms = mean_absolute_error(y_actual, y_predicted)
    return rms


def mape(y_actual, y_predicted): 
    return np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100

# THMP  RELH WSPD PRES
FEAT = 'PRES'
# data = pd.read_csv('/location/to/Kentucky_Data/2020/BMTN_ALL_2020.csv').head(2880)
# data = pd.read_csv('/location/to/Kentucky_Data/2020/BMTN_ALL_2020.csv').head(288)
data = pd.read_csv('/location/to/Kentucky_Data/2020/LSML_ALL_2020.csv').head(10000)

ad_fuller_result = adfuller(data[FEAT])
print(ad_fuller_result)

def optimize_SARIMA(parameters_list, d, D, s, exog):
    """
        Return dataframe with parameters, corresponding AIC and SSE
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
        exog - the exogenous variable
    """
    results = []
    
    for param in tqdm_notebook(parameters_list):
        try: 
            model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
            
        aic = model.aic
        results.append([param, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)x(P,Q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df

p = range(0, 3, 1)
d = 1
q = range(0, 3, 1)
P = range(0, 3, 1)
D = 1
Q = range(0, 3, 1)
# s = 4
parameters = product(p, q, P, Q)
parameters_list = list(parameters)
# print(len(parameters_list))

result_df = optimize_SARIMA(parameters_list, 1, 1, 4, data[FEAT])
print(result_df)

best_model = SARIMAX(data[FEAT], order=(0, 1, 2), seasonal_order=(0, 1, 2, 4)).fit(dis=-1)
print(best_model.summary())

data['arima_model'] = best_model.fittedvalues
data['arima_model'][:4+1] = np.NaN
forecast = best_model.predict(start=data.shape[0], end=data.shape[0]+1)

forecast = data['arima_model'].append(forecast)

# plt.figure(figsize=(15, 7.5))
# plt.plot(forecast[:288], color='r', label='model')
# forecast[:500].to_csv(FEAT,index=False)
# plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
# plt.plot(data[FEAT], label='actual')
# plt.legend()
# plt.show()


print(rmse(data[FEAT][5:10000], forecast[5:10000]))
print(mae(data[FEAT][5:10000], forecast[5:10000]))
print(mape(data[FEAT][5:10000], forecast[5:10000]))