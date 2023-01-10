import csv
import os
import copy
import math
import random
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from datetime import timedelta
from datetime import time
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

# import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
# from imblearn.keras import BalancedBatchGenerator
# from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from keras.datasets import cifar10

from numpy import moveaxis

from keras.models import Model
from keras.layers import Input, LSTM

from functools import wraps
# from keras.utils.vis_utils import plot_model

from scipy.signal import filtfilt, butter

from matplotlib import rcParams
from keras.regularizers import l2

# rcParams['font.family'] = 'Times New Roman'
# rcParams['font.size'] = 32
# plt.rc('xtick', labelsize=32)
# plt.rc('ytick', labelsize=32)
# plt.rcParams['text.usetex'] = True
# plt.rcParams["font.weight"] = "bold"
# plt.rc('font', weight='bold')

import argparse
import sys


def memoize(read_cache=True, cache_name='default'):
    def _memorize(function):
        @wraps(function)
        def __memorize(*args, **kwargs):
            if read_cache == True:
                print('read ' + cache_name + ' from cache!')
                cache = pickle.load(open(cache_name, 'rb'))
                return cache
            else:
                cur = function(*args, **kwargs)
                pickle.dump(cur, open(cache_name, "wb"))
                return cur

        return __memorize

    return _memorize


class Weather():

    def __init__(self):

        self.version = "7.0"
        self.version_info = "error check, time zone mismatch"
        self.data_path_0 = "/location/to/alabama_data/"
        self.wrf_data_path_0 = "/location/to/WRF_Data/"
        self.kentucky_data_path_0 = "/location/to/Kentucky_Data/"
        self.co2_path = "/home/kalebg/Documents/GitHub/temperatureLSTM/EditedModels/ProvidedCSVs/"
        self.seq = 5

        self.test_feature_list = ['TAIR']
        self.feature_list_full = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'TAIR', 'THMP', 'TDPT', 'ST02', 'SM02', 'SRAD',
                               'PRES', 'WDSD', 'RELH', 'DayOfTheYear', 'WSSD']
        self.feature_list_full_value = ['TAIR', 'THMP', 'TDPT', 'ST02', 'SM02', 'SRAD', 'PRES', 'WDSD', 'RELH',
                                     'DayOfTheYear', 'WSSD']

        self.wrf_feature_list_full = ['Year', 'Month', 'Date', 'Hour', 'Minute', 'ATT66', 'ATT67', 'ATT59', 'ATT111',
                                   'ATT32',
                                   'ATT94', 'ATT28', 'ATT69', 'ATT109', 'ATT68']
        self.wrf_feature_list_value = ['ATT66', 'ATT67', 'ATT59', 'ATT111', 'ATT32', 'ATT94', 'ATT28', 'ATT69', 'ATT109',
                                    'ATT68']

    # This module process the South Alabama Mesonet data
    def read_lstm_file(self, year='2017', year_during=1, start_date="0101", during=1, shift=None, seq=5):
        print("Reading Mesonet File")
        last_year = False
        train_y_all = pd.DataFrame(columns=self.test_feature_list)

        name_list = self.feature_list_full
        train_x_all = pd.DataFrame(columns=self.feature_list_full_value)

        # alabama data
        for yd in range(year_during):  # year
            if yd == year_during - 1:
                last_year = True
            else:
                last_year = False

            curr_year = str(int(year) + yd)
            self.data_path = self.data_path_0 + curr_year + '/'
            year_folders = sorted(os.listdir(self.data_path))

            train_y = pd.DataFrame(columns=self.test_feature_list)
            train_x = pd.DataFrame(columns=name_list)

            file_count = 0
            read_flag = False

            for f_alabma in year_folders:
                # directory are in sorted order and we need to reach the right directory to extract the data
                if start_date and not read_flag:
                    if not f_alabma[4:] == str(start_date):
                        continue
                    read_flag = True

                record_path = self.data_path + f_alabma + '/'
                station_records = sorted(os.listdir(record_path))

                for r in station_records:

                    r_split = r.split('_')
                    if not self.station == r_split[1]:
                        continue

                    station_record = record_path + r
                    try:
                        station_data = pd.read_csv(station_record)
                    except:
                        continue

                    # print(station_data)

                    station_data_t = copy.deepcopy(station_data)

                    for column in station_data_t.columns:
                        if not column in self.test_feature_list:
                            del station_data_t[column]

                    for column in station_data.columns:
                        if not column in name_list:
                            del station_data[column]

                    station_data = station_data.fillna(0.0)
                    station_data = station_data.replace('--', 0.0)  # replace -- symbol to 0
                    station_data_t = station_data_t.fillna(0.0)
                    station_data_t = station_data_t.replace('--', 0.0)  # replace -- symbol to 0

                    train_x = train_x.append(station_data, ignore_index=True, sort=False)
                    train_y = train_y.append(station_data_t, ignore_index=True, sort=False)

                # print(train_x.shape, train_y.shape, file_count)
                if file_count == during - 1:
                    print("Combining data")
                    # curr_idx = None
                    # predict_idx = None
                    # new_train_x = pd.DataFrame(columns=name_list)
                    # new_train_y = pd.DataFrame(columns = self.test_feature_list)

                    full_train_date = []

                    start_date_time = datetime.strptime(curr_year + start_date, '%Y%m%d')
                    curr_date_time_date = start_date_time
                    for d in range(during):
                        curr_date_time_date = start_date_time + timedelta(days=d)
                        day_of_year = curr_date_time_date.date() - datetime.strptime(curr_year, '%Y').date()
                        for h in range(24):
                            curr_date_time_hour = curr_date_time_date + timedelta(hours=h)
                            for m in range(60):
                                curr_date_time_minute = curr_date_time_hour + timedelta(minutes=m)
                                full_train_date.append(
                                    [curr_date_time_minute.year, curr_date_time_minute.day, day_of_year.days + 1,
                                     curr_date_time_minute.hour, curr_date_time_minute.minute])

                    full_train_date_df = pd.DataFrame(full_train_date,
                                                      columns=['Year', 'DayOfMon', 'DayOfYear', 'Hour', 'Minute'])

                    load_train_date = train_x.loc[:, 'Year':'Minute']

                    load_train_date_y = pd.concat([load_train_date, train_y], axis=1, sort=False)  # after average

                    full_train_x = pd.merge(full_train_date_df, train_x, how='left',
                                            on=['Year', 'DayOfMon', 'DayOfYear', 'Hour', 'Minute'])

                    full_train_y = pd.merge(full_train_date_df, load_train_date_y, how='left',
                                            on=['Year', 'DayOfMon', 'DayOfYear', 'Hour', 'Minute'])

                    full_train_x.fillna(method='ffill', inplace=True)
                    full_train_y.fillna(method='ffill', inplace=True)

                    # print(full_train_x)

                    train_x_date = full_train_x.loc[::5, 'Year':'Minute']
                    train_x_average = full_train_x.loc[:, self.feature_list_full_value].groupby(
                        np.arange(len(full_train_x)) // seq).mean()

                    # train_x_average.reset_index(drop=True, inplace=True)

                    # train_x_average = pd.concat([train_x_date, train_x_value], axis=1, sort=False)   # after average

                    train_y_average = full_train_y.loc[:, self.test_feature_list].groupby(
                        np.arange(len(full_train_y)) // seq).mean()
                    # train_y_average.reset_index(drop=True, inplace=True)

                    # Why last 6 hours of data is discarded?
                    count_row = train_x_average.shape[0]
                    # train_x_average = train_x_average.head(count_row-6*(60//seq)) #disabled by puru
                    train_x_average.reset_index(drop=True, inplace=True)
                    # train_y_average = train_y_average.head(count_row-6*(60//seq)) #disabled by puru
                    train_y_average.reset_index(drop=True, inplace=True)

                    train_x_all = pd.concat([train_x_all, train_x_average], ignore_index=True, sort=False)
                    train_y_all = pd.concat([train_y_all, train_y_average], ignore_index=True, sort=False)
                    # print(train_x_all.shape)

                    if not last_year:
                        break
                    else:
                        train_x_all = train_x_all.fillna(0.0)
                        train_x_all = train_x_all.replace('--', 0.0)
                        train_y_all = train_y_all.fillna(0.0001)
                        train_y_all = train_y_all.replace('--', 0.00)

                        print(train_x_all.shape, train_y_all.shape)
                        return train_x_all, train_y_all
                file_count += 1

        return None, None

    # This module process the WRF-HRRR data
    def read_wrf_file(self, year='2017', year_during=1, start_date="0101", during=1, shift=None, seq=5):
        last_year = False

        name_list = self.wrf_feature_list_full
        train_x_all = pd.DataFrame(columns=self.wrf_feature_list_value)

        # alabama data
        for yd in range(year_during):  # year
            if yd == year_during - 1:
                last_year = True
            else:
                last_year = False

            curr_year = str(int(year) + yd)
            self.data_path = self.wrf_data_path_0 + curr_year + '/'
            year_folders = sorted(os.listdir(self.data_path))

            train_x = pd.DataFrame(columns=name_list)

            file_count = 0
            read_flag = False

            for f_alabma in year_folders:
                if start_date and not read_flag:
                    if not f_alabma[4:] == str(start_date):
                        continue
                    read_flag = True

                record_path = self.data_path + f_alabma + '/'
                station_records = sorted(os.listdir(record_path))

                for r in station_records:
                    r_split = r.split('.')
                    if not self.station == r_split[0]:
                        continue

                    station_record = record_path + r
                    try:
                        station_data = pd.read_csv(station_record)
                    except:
                        continue

                    # print(station_data)

                    for column in station_data.columns:
                        if not column in name_list:
                            del station_data[column]

                    # station_data = station_data.fillna(0.0) # we will use interpolate to fill the NA record
                    station_data = station_data.replace('--', 0.0)  # replace -- symbol with None

                    train_x = train_x.append(station_data, ignore_index=True, sort=False)

                if file_count == during - 1:

                    # train_x = train_x.astype({'cloud_base_pressure': 'float64','cloud_top_pressure': 'float64'})

                    # Expand to minutely and interpolate
                    start_date_time = datetime.strptime(curr_year + start_date, '%Y%m%d')
                    curr_date_time_date = start_date_time
                    full_train_date = []
                    for d in range(during):
                        curr_date_time_date = start_date_time + timedelta(days=d)
                        day_of_year = curr_date_time_date.date() - datetime.strptime(curr_year, '%Y').date()
                        for h in range(24):
                            curr_date_time_hour = curr_date_time_date + timedelta(hours=h)
                            for m in range(60):
                                curr_date_time_minute = curr_date_time_hour + timedelta(minutes=m)
                                full_train_date.append(
                                    [curr_date_time_minute.year, curr_date_time_minute.month,
                                     curr_date_time_minute.day, curr_date_time_minute.hour,
                                     curr_date_time_minute.minute])

                    # range of date and time for which data needs to be populated
                    full_train_date_df = pd.DataFrame(full_train_date,
                                                      columns=['Year', 'Month', 'Date', 'Hour', 'Minute'])
                    # get minutewise data
                    full_train_x = pd.merge(full_train_date_df, train_x, how='left',
                                            on=['Year', 'Month', 'Date', 'Hour'])

                    full_train_x.interpolate(method='linear', limit_direction='forward', inplace=True)

                    train_v = full_train_x.loc[:, self.wrf_feature_list_value]

                    train_x_average = train_v.groupby(np.arange(len(full_train_x)) // seq).mean()

                    ## Why last 6 hours of data is discarded?

                    # train_x_average = train_x_average.tail(count_row-6*(60//seq)) #disable by puru
                    train_x_average.reset_index(drop=True, inplace=True)

                    train_x_all = pd.concat([train_x_all, train_x_average], ignore_index=True, sort=False)

                    if not last_year:
                        break
                    else:
                        train_x_all = train_x_all.fillna(0.0)
                        train_x_all = train_x_all.replace('--', 0.0)

                        return train_x_all

                file_count += 1

        return None

    # This module process the yearly co2 emission data
    def read_co2_file(self, year='2017', year_during=1, during=1, shift=None, seq=5, metric='min'):
        print("Reading co2 File")
        last_year = False
        train_y_all = pd.DataFrame(columns=self.test_feature_list)
        # train_y_all = pd.DataFrame(columns=['TempAnomaly('+metric+')'])

        name_list = self.feature_list_full
        train_x_all = pd.DataFrame(columns=self.feature_list_full_value)
        # name_list = ['year', 'co2', 'Median_Anomaly', 'Lower_bount']
        # train_x_all = pd.DataFrame(columns=['co2', 'Median_Anomaly', 'Lower_Bound', 'UpperBound'])

        print(year_during)
        # load all the values from both files into df, combine into one single df, 
        # return them before doing anything to make sure everything looks okie
        self.data_path = self.co2_path
        train_y = pd.DataFrame(columns=['TempAnomaly('+metric+')'])
        train_x = pd.DataFrame(columns=name_list)

        tempdf = pd.read_csv(self.co2_path + "global_temperature_anomaly.csv")
        co2df = pd.read_excel(self.co2_path + "global_annual_co2.xlsx")
        tempdf.drop(['Entity', 'Code'],inplace=True, axis=1)
        co2df.drop(['country'], axis=1)
        co2df.rename(columns={'year':'Year'}, inplace=True)
        full_train_x = pd.merge(tempdf, co2df, how='right', on=['Year'])
        full_train_x.drop(['country'], axis=1, inplace=True)
        

        # full_train_x.fillna(method='ffill', inplace=True)
        # print(full_train_x.shape)
        full_train_x.interpolate(method='linear', limit_direction='forward', inplace=True)
        print("full_train_x: ", full_train_x.shape)

        train_x_average = full_train_x.loc[:, self.feature_list_full_value].groupby(
            np.arange(len(full_train_x)) // seq).mean()
        print("train_x_average: ", train_x_average.shape)
        # concatenate data together
        train_x_all = pd.concat([train_x_all, train_x_average], ignore_index=True, sort=False)
        # print(train_x_all.shape)
        print("train_x_all: ", train_x_all.shape)

        station_data_t = copy.deepcopy(train_x_all)
        for column in station_data_t.columns:
            if not column in self.test_feature_list:
                del station_data_t[column]
        # print(station_data_t)
        # move the label data to train_y_all

        train_y_all = pd.concat([train_y_all, station_data_t], ignore_index=True, sort=False)
        print("train_y_all: ", train_y_all.shape)

        # PROBLEM: TRAIN_Y_ALL IS FILLED WITH NAN VALUES WHEN ITS NOT SUPPOED TO BE 
        # print(train_x_all.shape, train_y_all.shape)
        return train_x_all, train_y_all


        #################################################################################
        # self.test_feature_list = the value that we're trying to predict
        # name_list = all the values from kentucky weather data that include the yyyymmdd
                # = self.feature_list_full
        # self.feature_list_full_value = just the values, no yyyymmdd
        #################################################################################


        for yd in range(year_during):  # year
            if yd == year_during - 1:
                last_year = True
            else:
                last_year = False
            print(yd)
            curr_year = str(int(year) + yd)
            # self.data_path = self.kentucky_data_path_0 + curr_year + '/'
            self.data_path = self.co2_path
            station_records = os.listdir(self.data_path)  # populates the stations file

            record_path = self.data_path

            train_y = pd.DataFrame(columns=self.test_feature_list)
            train_x = pd.DataFrame(columns=name_list)

            for r in station_records:
                r_split = r.split('_')
                if not self.station == r_split[0]:
                    continue

                station_record = record_path + r
                print(station_record)
                try:
                    station_data = pd.read_csv(station_record)
                except:
                    continue

                print(station_data.shape)
                for column in station_data.columns:
                    if not column in name_list:
                        del station_data[column]

                # station_data = station_data.fillna(0.0)
                station_data = station_data.replace('--', 0.0)  # replace -- symbol by None

                print(station_data.shape)

                # Pick the data for the given range date in a year
                start_date_time = datetime.strptime(curr_year + start_date, '%Y%m%d')
                curr_date_time_date = start_date_time
                full_train_date = []
                for d in range(during):
                    curr_date_time_date = start_date_time + timedelta(days=d)
                    day_of_year = curr_date_time_date.date() - datetime.strptime(curr_year, '%Y').date()
                    for h in range(24):
                        curr_date_time_hour = curr_date_time_date + timedelta(hours=h)
                        for m in range(60):
                            curr_date_time_minute = curr_date_time_hour + timedelta(minutes=m)
                            full_train_date.append(
                                [curr_date_time_minute.year, curr_date_time_minute.month, day_of_year.days + 1,
                                 curr_date_time_minute.day, curr_date_time_minute.hour, curr_date_time_minute.minute])

                # range of date and time for which data needs to be populated
                full_train_date_df = pd.DataFrame(full_train_date,
                                                  columns=['Year', 'Month', 'DayOfYear', 'Day', 'Hour', 'Minute'])

                # get minutewise data
                full_train_x = pd.merge(full_train_date_df, station_data, how='left',
                                        on=['Year', 'Month', 'Day', 'Hour', 'Minute'])

                # full_train_x.fillna(method='ffill', inplace=True)
                full_train_x.interpolate(method='linear', limit_direction='forward', inplace=True)

                train_x_average = full_train_x.loc[:, self.feature_list_full_value].groupby(
                    np.arange(len(full_train_x)) // seq).mean()

                # concatenate data together
                train_x_all = pd.concat([train_x_all, train_x_average], ignore_index=True, sort=False)
                print(train_x_all.shape)

            if last_year:
                station_data_t = copy.deepcopy(train_x_all)
                for column in station_data_t.columns:
                    if not column in self.test_feature_list:
                        del station_data_t[column]
                print(station_data_t.shape)
                # move the label data to train_y_all

                train_y_all = pd.concat([train_y_all, station_data_t], ignore_index=True, sort=False)

                print(train_x_all.shape, train_y_all.shape)
                return train_x_all, train_y_all

        return None, None

    # For testing extreme weather parameter: get threshold value
    def get_range(self, y):
        count = len(y)
        nth = int(count * 0.05)  # 5% of total number
        temp = np.partition(-y, nth, axis=0)

        top_range = min(-temp[: nth])  # start point of top 5% range

        temp = np.partition(y, nth, axis=0)
        bottom_range = max(temp[: nth])

        return top_range, bottom_range

    # Select extreme cases only
    def ext_weather(self, x, y, threshold, top_range=1):
        indexList = []
        if top_range == 1:
            for i in range(len(y)):
                if y[i] < threshold:
                    indexList.append(i)
        else:
            for i in range(len(y)):
                if y[i] > threshold:
                    indexList.append(i)

        x = np.delete(x, indexList, axis=0)
        y = np.delete(y, indexList, axis=0)
        return x, y

    # Generate sequence data for training and testing the Modelet
    def pre_seq(self, x, y, encoder_seq=12, decoder_seq=12):

        input_texts = x
        target_texts = y
        input_len = len(x)

        max_encoder_seq_len = encoder_seq
        max_decoder_seq_len = decoder_seq + 1
        # num_encoder_len = len(x[0])
        # num_decoder_len = len(y[0])

        num_encoder_shape = x.shape
        num_encoder_len = num_encoder_shape[1]  # TODO: one more control

        num_decoder_len = 1 + 2  # 2 control bit, -2: start, -1: end

        encoder_input_data = np.zeros((input_len, max_encoder_seq_len, num_encoder_len), dtype='float32')
        decoder_input_data = np.zeros((input_len, max_decoder_seq_len, num_decoder_len), dtype='float32')
        decoder_target_data = np.zeros((input_len, max_decoder_seq_len, num_decoder_len), dtype='float32')

        eff_input_len = 0
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):

            for t in range(0, max_encoder_seq_len):
                if i + t + encoder_seq < input_len:
                    if t == 0 and i + t + encoder_seq * 2 < input_len:
                        eff_input_len += 1

                    encoder_input_data[i, t,] = input_texts[i + t]

            for t in range(0, max_decoder_seq_len):
                if i + t + encoder_seq < input_len:
                    if t == 0:
                        decoder_input_data[i, t, -2] = 1
                        decoder_target_data[i, t, 0] = target_texts[i + t + encoder_seq]
                        # added by Puru
                        decoder_input_data[i, t, 0] = target_texts[i + t]  # reference is current value

                    else:
                        decoder_input_data[i, t, 0] = target_texts[i + t + encoder_seq - 1]
                        decoder_target_data[i, t, 0] = target_texts[i + t + encoder_seq]

            if i + t + encoder_seq + 1 < input_len:
                decoder_target_data[i, t, -1] = 1

        encoder_input_data = encoder_input_data[:eff_input_len]
        decoder_input_data = decoder_input_data[:eff_input_len]
        decoder_target_data = decoder_target_data[:eff_input_len]

        return encoder_input_data, decoder_input_data, decoder_target_data

    # Generate sequence data for training and testing simple NN
    def pre_seq_NN(self, x, y, seq=5):

        input_len = len(x)
        eff_input_len = input_len - int(60 / seq)

        x_data = x[:eff_input_len]
        y_data = y[input_len - eff_input_len:]

        return x_data, y_data

    # Dump the predicted and true data
    def draw_lstm_without_e(self, tru, pre, scaler, lstm_path=None):
        '''
        draw predciton and ground dataset with standard deviation
        '''

        tru = scaler.inverse_transform(tru)
        pre = scaler.inverse_transform(pre)

        idx = 0
        test_len = 2400
        tru_10 = []
        pre_10 = []
        for i in range(min(len(tru), test_len)):
            tru_10.append(tru[i][idx])
            pre_10.append(pre[i][idx])

        pickle.dump(pre, open(lstm_path + '.pre', "wb"))
        pickle.dump(tru, open(lstm_path + '.tru', "wb"))
        '''
        fig, ax1 = plt.subplots()

        x = [i * 5 for i in range(0, len(tru_10))]

        plt.plot(x, tru_10, label='true')
        plt.plot(x, pre_10, label='prediction')

        plt.ylabel("Value")
        plt.xlabel("Minutes")

        plt.legend()
        plt.show()
        '''

    # Dump the predicted and true data
    def draw_ML_without_e(self, tru, pre, scaler, lstm_path=None):
        '''
        draw predciton and ground dataset with standard deviation
        '''

        tru = scaler.inverse_transform(tru)
        pre = scaler.inverse_transform(pre)
        print(tru.shape, pre.shape)

        test_len = 2400
        tru_10 = []
        pre_10 = []
        for i in range(min(len(tru), test_len)):
            tru_10.append(tru[i])
            pre_10.append(pre[i])

        pickle.dump(pre, open(lstm_path + '.pre', "wb"))
        pickle.dump(tru, open(lstm_path + '.tru', "wb"))

        '''
        fig, ax1 = plt.subplots()

        x = [i * 5 for i in range(0, len(tru_10))]

        plt.plot(x, tru_10, label='true')
        plt.plot(x, pre_10, label='prediction')

        plt.ylabel("Value")
        plt.xlabel("Minutes")
        plt.legend()
        plt.show()
        '''
    # Compute error matrices and dump them
    def print_MSE(self, tru, predd, scaler, lstm_path=None, decoderLen=6):
        '''
        Compute MSE
        '''
        pred = pickle.load(open(lstm_path + '.pre', "rb"))
        true = pickle.load(open(lstm_path + '.tru', "rb"))

        MSE = mean_squared_error(true[:, 0:decoderLen], pred[:, 0:decoderLen], multioutput='raw_values', squared=False)
        # MSE = mean_squared_error(true[:, 0], pred[:, 0], squared=False)
        MAE = mean_absolute_error(true[:, 0], pred[:, 0], multioutput='raw_values')
        tru, pre = self.mapeProcess(true[:, 0], pred[:, 0])
        MAPE = mean_absolute_percentage_error(tru, pre)

        # MSE to file
        with open(lstm_path + 'MSE_error.txt', "a") as fid:
            x = np.array([1])
            MSE.tofile(fid, sep=",", format="RMSE %s")
            x.tofile(fid, sep=",", format=",%s\nMAE ")
            MAE.tofile(fid, sep=",", format="%s")
            x.tofile(fid, sep=",", format=",%s\nMAPE ")
            MAPE.tofile(fid, sep=",", format="%s")
            x.tofile(fid, sep=",", format=",%s\n")

        print("RMSE_" + lstm_path + " = ", MSE[0])

        with open(lstm_path + '_check.txt', 'a+') as fid:
            fid.seek(0)
            val = fid.readline()
            print(val)
            if val:
                if float(val) > MSE[0]:
                    fid.truncate(0)
                    fid.write(str(MSE[0]))
                    with open(lstm_path + "_Best_pred.txt", 'w') as fig:
                        predic = pred[:, 0]
                        predic = predic.flatten()
                        # print(predic.shape)
                        for p in predic:
                            fig.write(str(p) + "\n")
                        # predic.tofile(fig, sep="", format="%s\n")
            else:
                # print("Not exist")
                fid.write("9999")
                with open(lstm_path + "_Best_pred.txt", 'w') as fig:
                    predic = pred[:, 0]
                    predic = predic.flatten()
                    # print(predic.shape)
                    for p in predic:
                        fig.write(str(p) + "\n")

    # Compute error matrices and dump them
    def print_MSE_ML(self, truuuu, preddd, scaler, lstm_path=None):
        '''
        Compute MSE
        '''
        pred = pickle.load(open(lstm_path + '.pre', "rb"))
        true = pickle.load(open(lstm_path + '.tru', "rb"))

        MSE = mean_squared_error(true, pred, multioutput='raw_values', squared=False)
        MAE = mean_absolute_error(true, pred, multioutput='raw_values')
        tru, pre = self.mapeProcess(true, pred)
        MAPE = mean_absolute_percentage_error(tru, pre)

        # MSE to file
        with open(lstm_path + str(self.seq) + 'MSE_error.txt', "a") as fid:
            x = np.array([1])
            MSE.tofile(fid, sep=",", format="RMSE %s")
            x.tofile(fid, sep=",", format=",%s\nMAE ")
            MAE.tofile(fid, sep=",", format="%s")
            x.tofile(fid, sep=",", format=",%s\nMAPE ")
            MAPE.tofile(fid, sep=",", format="%s")
            x.tofile(fid, sep=",", format=",%s\n")

        print("RMSE_" + lstm_path + " = ", MSE[0])

        with open(lstm_path + '_check.txt', 'a+') as fid:
            fid.seek(0)
            val = fid.readline()
            print(val)
            if val:
                if float(val) > MSE[0]:
                    fid.truncate(0)
                    fid.write(str(MSE[0]))
                    with open(lstm_path + str(self.seq) + "_Best_pred.txt", 'w') as fig:
                        predic = pred[:, 0]
                        predic = predic.flatten()
                        # print(predic.shape)
                        for p in predic:
                            fig.write(str(p) + "\n")
                        # predic.tofile(fig, sep="", format="%s\n")
            else:
                # print("Not exist")
                fid.write("9999")
                with open(lstm_path + str(self.seq) + "_Best_pred.txt", 'w') as fig:
                    predic = pred[:, 0]
                    predic = predic.flatten()
                    # print(predic.shape)
                    for p in predic:
                        fig.write(str(p) + "\n")

    # Filter out data if true value is smaller than 0.35 (MAPE is badly skewed for smaller true value)
    def mapeProcess(self, true, pred):
        tru = []
        pre = []
        for i in range(len(true)):
            if true[i] > 0.35:
                tru.append(true[i])
                pre.append(pred[i])
        return tru, pre

    def print_MSE_WRF(self, true, pred, lstm_path=None):
        '''
        Compute MSE
        '''
        MSE = mean_squared_error(true, pred, multioutput='raw_values', squared=False)
        MAE = mean_absolute_error(true, pred, multioutput='raw_values')
        tru, pre = self.mapeProcess(true, pred)
        MAPE = mean_absolute_percentage_error(tru, pre)
        # MSE to file
        with open(lstm_path + str(self.seq) + 'MSE_error.txt', "a") as fid:
            x = np.array([1])
            MSE.tofile(fid, sep=",", format="RMSE %s")
            x.tofile(fid, sep=",", format=",%s\nMAE ")
            MAE.tofile(fid, sep=",", format="%s")
            x.tofile(fid, sep=",", format=",%s\nMAPE ")
            MAPE.tofile(fid, sep=",", format="%s")
            x.tofile(fid, sep=",", format=",%s\n")

        print("RMSE_" + lstm_path + " = ", MSE[0])

    # Data sanitization: the dataset must be numpyArray
    def sanitize(self, dataSet):
        runningAvg = dataSet[0, :]
        average = np.average(dataSet, axis=0)  # if very first one is NA data
        start = True
        count = 0
        for k in range(len(dataSet)):
            data = dataSet[k, :]
            for i in range(len(data)):
                if data[i] == 0.0:
                    count = count + 1
                    dataSet[k, i] = runningAvg[i]
                    data[i] = runningAvg[i]
                    if start:
                        dataSet[k, i] = average[i]
                        data[i] == average[i]
                        runningAvg[i] = average[i]
            start = False
            runningAvg = runningAvg * 0.9 + data * 0.1
        print("Total number of sanitized entry = ", count)
        return dataSet

