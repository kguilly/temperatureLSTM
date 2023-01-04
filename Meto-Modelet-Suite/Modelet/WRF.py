from __future__ import print_function
import argparse
import sys
import numpy as np
ID = np.random.randint(2) # Randomly generate  0 and 1

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #str(ID) # Randomly select GPU

import tensorflow as tf
import keras
from keras.models import load_model
from weather import Weather
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, LSTM, Dense,  Add

from sklearn.preprocessing import MinMaxScaler
import matplotlib
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.utils.fixes import parse_version
from scipy.signal import find_peaks
from keras.utils import to_categorical

#Setting gpus by Puru
gpus = tf.config.list_physical_devices('GPU')
print("*** Checking gpus: ",gpus)
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)
## GPU setting ends here

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 24})
# plt.rcParams['figure.figsize'] = 16, 12

if parse_version(matplotlib.__version__) >= parse_version('2.1'):
    density_param = {'density': True}
else:
    density_param = {'normed': True}



m = Weather()
m.latent_dim = 256

stationList = ['BMTN', 'CCLA', 'FARM', 'HUEY', 'LXGN','ELST','FCHV','QKSD','LSML','LGRN','CROP','DANV']  # include your station list here
# Default station and paramID
station = 'BMTN'
paramID = 1  # 1: Temperature, 2: Relative Humidity, 3: Soil temperature, 4: Wind speed, 5: Pressure, and 6: Soil moisture
top_range = 1 # test for high range
train = True
###############################################
# Training date range
during = 90  # How many days of data in each year
year_during = 2  # How many years of data
start_date = "0715"  # Start date of each year
year = "2018"
seq = 5

# Test date range
T_year_during = 1
T_during = 80
T_start_date = "0720"
T_year = "2020"

mapper_epochs = 100
model_epochs = 60
decoderLen = 12 # for60 minutes prediction
##############################################

print(len(sys.argv))
if len(sys.argv) > 1 and len(sys.argv) < 3:
    print("please follow the execution like: ")
    print("python modelet_name (say, kentucky_macro.py) station_name (say, BMBL) parmaID (say, 1)")
    exit(0)

# python kentucky_micro BMBL 3
if len(sys.argv) == 3:
    station = sys.argv[1]
    paramID = sys.argv[2]
    paramID = int(paramID)

if station not in stationList:
    print("Invalid station Name, please check the list")
    exit(1)

if paramID > 4 or paramID < 1:
    print("invalid paramID")
    exit(1)

m.station = station

if paramID == 1:
    m.test_feature_list = ['TAIR']
    m.feature_list_full = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'TAIR', 'THMP', 'TDPT', 'ST02', 'SM02', 'SRAD',
                           'PRES', 'WDSD', 'RELH', 'DayOfTheYear', 'WSSD']
    m.feature_list_full_value = ['TAIR', 'THMP', 'TDPT', 'ST02', 'SM02', 'SRAD', 'PRES', 'WDSD', 'RELH', 'DayOfTheYear',
                                 'WSSD']

    m.wrf_feature_list_full = ['Year', 'Month', 'Date', 'Hour', 'Minute', 'ATT66', 'ATT67', 'ATT59', 'ATT111', 'ATT73',
                               'ATT94', 'ATT28', 'ATT69', 'ATT109', 'ATT68']
    m.wrf_feature_list_value = ['ATT66', 'ATT67', 'ATT59', 'ATT111', 'ATT73', 'ATT94', 'ATT28', 'ATT69', 'ATT109',
                                'ATT68']
if paramID == 2:
    m.test_feature_list = ['RELH']
    m.feature_list_full = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'RELH','SRAD', 'WSSD', 'WDSD', 'WSMX', 'WSPD',
                           'WSMN', 'THMP', 'PRES', 'TAIR']
    m.feature_list_full_value = ['RELH','SRAD', 'WSSD', 'WDSD', 'WSMX', 'WSPD', 'WSMN', 'THMP', 'PRES', 'TAIR']

    m.wrf_feature_list_full = ['Year', 'Month', 'Date', 'Hour', 'Minute', 'ATT70', 'ATT4', 'ATT108', 'ATT110', 'ATT112',
                               'ATT91', 'ATT90', 'ATT61', 'ATT113', 'ATT111']
    m.wrf_feature_list_value = ['ATT70', 'ATT4', 'ATT108', 'ATT110', 'ATT112', 'ATT91', 'ATT90', 'ATT61', 'ATT113',
                                'ATT111']

if paramID == 3:
    m.test_feature_list = ['WSPD']
    m.feature_list_full = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'WSPD', 'WSMX', 'WSMN', 'WSSD', 'PRES', 'ST02',
                           'RELH', 'SM02']
    m.feature_list_full_value = ['WSPD', 'WSMX', 'WSMN', 'WSSD', 'PRES', 'ST02', 'RELH', 'SM02']

    m.wrf_feature_list_full = ['Year', 'Month', 'Date', 'Hour', 'Minute', 'ATT8', 'ATT73', 'ATT89', 'ATT115', 'ATT98',
                               'ATT18', 'ATT121', 'ATT125']
    m.wrf_feature_list_value = ['ATT8', 'ATT73', 'ATT89', 'ATT115', 'ATT98', 'ATT18', 'ATT121', 'ATT125']

if paramID == 4:
    m.test_feature_list = ['PRES']
    m.feature_list_full = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'PRES', 'TDPT', 'TAIR', 'THMP', 'ST02', 'WSMN',
                           'WSPD']
    m.feature_list_full_value = ['PRES', 'TDPT', 'TAIR', 'THMP', 'ST02', 'WSMN', 'WSPD']

    m.wrf_feature_list_full = ['Year', 'Month', 'Date', 'Hour', 'Minute', 'ATT57', 'ATT39', 'ATT40', 'ATT18', 'ATT94',
                               'ATT125', 'ATT128']
    m.wrf_feature_list_value = ['ATT57', 'ATT39', 'ATT40', 'ATT18', 'ATT94', 'ATT125', 'ATT128']


print(station)
print(paramID)

print(m.test_feature_list)
print(m.feature_list_full)
print(m.wrf_feature_list_full)
#######################################################

global_path = "cache/" + m.station + "_WRF_"+str(top_range) +"_" + m.test_feature_list[0] + "_all"

#train_x, train_y = m.read_kentucky_file(year=year, year_during=year_during, start_date=start_date, during=during, shift=5, seq=seq)

#train_x = m.sanitize(np.array(train_x))
#train_y = m.sanitize(np.array(train_y))

test_x, test_y = m.read_kentucky_file(year=T_year, year_during=T_year_during, start_date=T_start_date, during=T_during,
                                  shift=5, seq=seq)
test_wrf_x = m.read_wrf_file(year=T_year, year_during=T_year_during, start_date=T_start_date, during=T_during, shift=5,
                             seq=seq)

#test_wrf_x.to_csv('test_wrf_x.csv')

test_x = m.sanitize(np.array(test_x))
true = m.sanitize(np.array(test_y))
test_wrf_x = m.sanitize(np.array(test_wrf_x))

# filter extreme weather condition for WRF
#test_wrf_x, true = m.ext_weather(test_wrf_x, test_y, threshold, top_range)

if paramID == 1:
    pred = test_wrf_x[:, 0] - 273

elif paramID == 5:
    pred = test_wrf_x[:, 0]/100
else:
    pred = test_wrf_x[:, 0]

m.print_MSE_WRF(true, pred, lstm_path=global_path)
