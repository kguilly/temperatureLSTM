import pandas as pd
import numpy as np

temperaturefile = "tempData/1895-2022.csv"
anomalyFile = "tempData/Global_Temperature-anomaly.csv"
emissionsFile = "tempData/World_CO2.xlsx"

temperatureData = pd.read_csv(temperaturefile)
anomalyData = pd.read_csv(anomalyFile)
emissionsData = pd.read_excel(emissionsFile)


newTemperatureData = pd.DataFrame(columns=['Year', 'Value (Degrees F)'])
# take the average for all the months out of each year
takeAvg = pd.DataFrame(columns=['Month', 'value'])
prevYear = temperatureData.Date.iloc[0].astype(int).astype(str)[0:4]

exit()
#################33
# reading emissions data into pd.df
newEmissionData = pd.DataFrame(columns=['year', 'co2'])
newEmissionData['year'] = emissionsData['year']
newEmissionData['co2'] = emissionsData['co2']
newEmissionData = newEmissionData.apply(pd.to_numeric, errors='coerce')
newEmissionData.dropna(how='any', inplace=True)
newEmissionData = newEmissionData.reset_index(drop=True)
# newEmissionData.to_csv(path_or_buf="emission_data.csv", header=True)
emissionData1850 = newEmissionData.drop(newEmissionData.index[0:100], axis=0, inplace=False)
emissionData1850.reset_index(drop=True, inplace=True)
emissionData1850.drop([170, 171], axis=0, inplace=True)
emissionData1850.to_csv("1850-2019_EmissionData.csv", sep=',', encoding='utf-8', index=False)
