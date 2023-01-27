import os
import copy
import pickle

import xlrd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta


from functools import wraps
# from keras.utils.vis_utils import plot_model




def memoize(read_cache = True, cache_name = 'default'):
    def _memorize(function):
        @wraps(function)
        def __memorize(*args, **kwargs):
            if read_cache == True:
                print('read '+cache_name+ ' from cache!')
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
        '''
        station: specify a weather station from station list
        test_feature_list: specify which weather parameter want to be predicted

        '''
        self.version = "7.0"
        self.version_info = "error check, time zone mismatch"
        self.data_path_0 = "alabama/"
        self.neighbor_dat_path_0 = "alabama_neighbor/"
        self.wrf_data_path_0 = "wrf/"
        self.data_path = None
        self.wrf_data_path = None
        # self.station_list = ['andalusia', 'atmore','bayminette','castleberry','dixie','elberta','fairhope','florala', 'foley', 'gasque','geneva','jay','kinston','leakesville','loxley','mobiledr','mobileusaw','saraland']
        self.station_list = ['elberta','fairhope','florala', 'foley', 'gasque','geneva','jay','kinston','leakesville','loxley','mobiledr','mobileusaw','saraland']

        # self.exp = ['Precip_TB3_Tot','Temp_C','WndDir_2m','WndDir_10m','WndSpd_2m','WndSpd_10m','WndSpd_Vert','WndSpd_2m_Max','WndSpd_10m_Max','WndSpd_Vert_Max','WndSpd_Vert_Min','WndSpd_2m_Std','WndSpd_10m_Std','WndSpd_2m_WVc_1','WndSpd_2m_WVc_2', 'WndSpd_2m_WVc_3','WndSpd_2m_WVc_4','WndSpd_10m_WVc_1','WndSpd_10m_WVc_2','WndSpd_10m_WVc_3','WndSpd_10m_WVc_4','WndSpd_Vert_Tot']
        
        self.exp = ['Precip_TB3_Tot','WndDir_2m','WndDir_10m','WndSpd_2m','WndSpd_10m','WndSpd_Vert']
        
        #self.feature_list_full = ['Year','DayOfMon','DayOfYear','Hour','Minute','PTemp', "IRTS_Trgt", 'Precip_TX_Tot', 'IRTS_Body', "SoilSfcT", "SoilT_5cm","SoilT_10cm","SoilT_20cm","SoilT_50cm","SoilT_100cm","AirT_1pt5m","AirT_2m","AirT_9pt5m","AirT_10m","RH_2m","RH_10m","Pressure_1","Pressure_2","TotalRadn","QuantRadn","WndDir_2m","WndDir_10m","WndSpd_2m","WndSpd_10m","WndSpd_Vert","Vitel_100cm_a","Vitel_100cm_b","Vitel_100cm_c","Vitel_100cm_d","WndSpd_2m_Max","WndSpd_10m_Max","WndSpd_Vert_Max","WndSpd_Vert_Min","WndSpd_2m_Std","WndSpd_10m_Std","SoilType","eR","el","Temp_C","er_tc","wfv", "NaCl","SoilCond","SoilCond_tc","SoilWaCond_tc", "WndSpd_2m_WVc_1", "WndSpd_2m_WVc_2","WndSpd_2m_WVc_3","WndSpd_2m_WVc_4","WndSpd_10m_WVc_1", "WndSpd_10m_WVc_2","WndSpd_10m_WVc_3","WndSpd_10m_WVc_4","WndSpd_Vert_Tot"]
        #self.feature_list_full_value = ['PTemp', "IRTS_Trgt", 'Precip_TX_Tot', 'IRTS_Body', "SoilSfcT", "SoilT_5cm","SoilT_10cm","SoilT_20cm","SoilT_50cm","SoilT_100cm","AirT_1pt5m","AirT_2m","AirT_9pt5m","AirT_10m","RH_2m","RH_10m","Pressure_1","Pressure_2","TotalRadn","QuantRadn","WndDir_2m","WndDir_10m","WndSpd_2m","WndSpd_10m","WndSpd_Vert","Vitel_100cm_a","Vitel_100cm_b","Vitel_100cm_c","Vitel_100cm_d","WndSpd_2m_Max","WndSpd_10m_Max","WndSpd_Vert_Max","WndSpd_Vert_Min","WndSpd_2m_Std","WndSpd_10m_Std","SoilType","eR","el","Temp_C","er_tc","wfv", "NaCl","SoilCond","SoilCond_tc","SoilWaCond_tc", "WndSpd_2m_WVc_1", "WndSpd_2m_WVc_2","WndSpd_2m_WVc_3","WndSpd_2m_WVc_4","WndSpd_10m_WVc_1", "WndSpd_10m_WVc_2","WndSpd_10m_WVc_3","WndSpd_10m_WVc_4","WndSpd_Vert_Tot"]
        
        self.feature_list_full = ['Year','DayOfMon','DayOfYear','Hour','Minute','Temp_C','PTemp','IRTS_Body', "Vitel_100cm_a","Vitel_100cm_b","Vitel_100cm_d","SoilCond", "SoilWaCond_tc", "eR", "wfv","RH_10m"]
        # self.feature_list_full_value = ['Record Date  MM/DD/YYYY', 
        #                                 'Air Temperature Max (Degrees Fahrenheit F)', 
        #                                 'Relative Humidity Max (Percent)',
        #                                 'Precipitation (Inches n.nn)  total rain fall that occurred for the day.',
        #                                 'Wind Speed (Miles Per Hour)  Resultant / Average speed for the day.',
        #                                 'Wind Direction (Degrees)   Resultant Direction for the day',
        #                                 "Solar Radiation (Langley's)  that occurred for the day."
        #                                 ]
        # self.feature_list_full_value = ['TIMESTAMP', 
        #                         'AirTF_Avg', 
        #                         'AirTF_Max',
        #                         'AirTF_Min',
        #                         'AirTF',
        #                         'RH_Max',
        #                         'RH_Min',
        #                         'RH',
        #                         'WS_mph_Max',
        #                         'WS_mph_Min',
        #                         'WS_mph_Avg',
        #                         'WS_mph',
        #                         'WindDir',
        #                         'SlrW_Max',
        #                         'SlrW_Min',
        #                         'SlrW',
        #                         'SlrMJ_Tot',
        #                         'T109_F_2in_Avg', 
        #                         'T109_F_2in', 
        #                         'T109_F_4in_Avg', 
        #                         'T109_F_4in', 
        #                         'WS_mph_S_WVT', 
        #                         'WindDir_D1_WVT', 
        #                         'WindDir_SD1_WVT', 
        #                         'WS_mph_S_WVTR', 
        #                         'WS_mph_U_WVTR', 
        #                         'WindDir_DU_WVTR', 
        #                         'WindDir_SDU_WVTR',
        #                         'Rain_in_Tot',
        #                         ]
        self.feature_list_full_value = ['TIMESTAMP', 
                        'AirTF_Avg', 
                        # 'AirTF_Max',
                        # 'AirTF_Min',
                        'AirTF',
                        # 'RH_Max',
                        # 'RH_Min',
                        'RH',
                        # 'WS_mph_Max',
                        # 'WS_mph_Min',
                        # 'WS_mph_Avg',
                        # 'WS_mph',
                        # 'WindDir',
                        # 'SlrW_Max',
                        # 'SlrW_Min',
                        'SlrW',
                        'SlrMJ_Tot',
                        'T109_F_2in_Avg', 
                        'T109_F_2in', 
                        'T109_F_4in_Avg', 
                        'T109_F_4in', 
                        # 'WS_mph_S_WVT', 
                        # 'WindDir_D1_WVT', 
                        # 'WindDir_SD1_WVT', 
                        # 'WS_mph_S_WVTR', 
                        # 'WS_mph_U_WVTR', 
                        # 'WindDir_DU_WVTR', 
                        # 'WindDir_SDU_WVTR',
                        'Rain_in_Tot',
                        ]
        # self.wrf_feature_list_full = ['Year','Month','Date','Hour','Minute','temperature', 'windspd_ground',  'windspd_10m', 'u_500hpa', 'v_500hpa', 'u_850hpa', 'v_850hpa', 'u_1000hpa', 'v_1000hpa', 'u_10m', 'v_10m','humi','cloud_base_pressure','cloud_top_pressure','3000_h','1000_h','ground_mis']
        # self.feature_list_full = ["AirT_2m", "Precip_TX_Tot", "WndSpd_10m", "WndDir_10m","RH_2m","RH_10m",'WndDir_2m', 'WndSpd_2m']
        # self.wrf_feature_list_full = ['Year','Month','Date','Hour','Minute','u_500hpa', 'v_500hpa', 'cloud_base_pressure','cloud_top_pressure','3000_h']
        self.wrf_feature_list_full = ['Year','Month','Date','Hour','Minute','u_500hpa', 'v_500hpa', 'cloud_base_pressure','cloud_top_pressure','3000_h','1000_h','ground_mis', 'u_10m','v_10m','u_250hpa','v_250hpa','u_80m','v_80m','surface_pressure']
        # self.wrf_feature_list_value = ['temperature', 'windspd_ground',  'windspd_10m', 'u_500hpa', 'v_500hpa', 'u_850hpa', 'v_850hpa', 'u_1000hpa', 'v_1000hpa', 'u_10m', 'v_10m','humi','cloud_base_pressure','cloud_top_pressure','3000_h','1000_h','ground_mis']
        # self.wrf_feature_list_value = ['u_500hpa', 'v_500hpa', 'cloud_base_pressure','cloud_top_pressure','3000_h']
        self.wrf_feature_list_value = ['u_500hpa', 'v_500hpa', 'cloud_base_pressure','cloud_top_pressure','3000_h','1000_h','ground_mis', 'u_10m','v_10m','u_250hpa','v_250hpa','u_80m','v_80m','surface_pressure']
        self.test_feature_list = ["PTemp"]
        self.lat_lon = [None, None]


    def read_co2(self, start_data=None, end_date=None, order=1):
        if order == 1:
            data = pd.read_excel("Data/co2_all.xlsx")
            data.year = data.year.astype("Int64")
            data = data.set_index(['year'])
            data.index = pd.to_datetime(data.index, format='%Y')
            return data

        if order == 2:
            data = pd.read_csv("Data/temp_all.csv")
            # data.year = data.year.astype("Int64")
            # data = data.set_index(['year'])
            # data.index = pd.to_datetime(data.index, format='%Y')
            return data




    def read_soil_data_o(self, start_date, end_date):
        data = pd.read_excel("Data/Soy02 Sensor Data and Irrigation.xlsx")
        data = data.set_index(['Timestamp (UTC)'])
        data.index = pd.to_datetime(data.index)
        data_filter = data.loc[start_date:end_date]
        return data_filter

    def read_soil_data(self, start_date, end_date):
        data = pd.read_excel("Data/Soy02 Sensor Data and Irrigation.xlsx")
        # print(data.columns.values)
        # data = data[['Timestamp (UTC)', 'Sensor Reading @ 15 cm', 'Sensor Reading @ 30 cm', 'Sensor Reading @ 60 cm']]
        data = data[['Timestamp (UTC)']+self.test_feature_list]
        # data['Timestamp (UTC)'] = pd.to_datetime(data['Timestamp (UTC)'])
        data = data.set_index(['Timestamp (UTC)'])
        # data_mean = data.groupby(data.index.strftime("%Y-%m-%d")).mean()
        # data_mean.index = pd.to_datetime(data_mean.index)
        data.index = pd.to_datetime(data.index)
        data_filter = data.loc[start_date:end_date]
        data_mean = data_filter.groupby(pd.Grouper(freq='120Min', offset=0)).mean()

        return data_mean


    def read_station_data(self, start_date, end_date,):
        data = pd.read_excel("Data/Stoneville-W-new.xlsx")
        # workbook = xlrd.open_workbook("Data/Stoneville W-daily.xls", ignore_workbook_corruption=True)
        # data = pd.read_excel(workbook)
        # print(data.columns.values)
        
        # print(list(data.columns.values))
        data = data[self.feature_list_full_value]
        
        # data.drop(data.tail(2).index,inplace=True) # drop last n rows, sum and average row
        # data['Record Date  MM/DD/YYYY'] = pd.to_datetime(data['Record Date  MM/DD/YYYY'], format="%m/%d/%Y")
        # data = data.set_index(['Record Date  MM/DD/YYYY'])
        data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], format="%m/%d/%Y")
        data = data.set_index(['TIMESTAMP'])
   
        # data_mean = data.groupby(pd.Grouper(freq='30Min', offset=0)).mean()
        # data_mean.index = pd.to_datetime(data_mean.index)
   
        # data_mean.resample('60min', base=30, label='right').first()
        data_filter = data.loc[start_date:end_date]
        data_mean = data_filter.groupby(pd.Grouper(freq='120Min', offset=0)).mean()
        return data_mean

  
    def read_lstm_file(self, year = '2017', year_duration = 1, start_date = "0101", duration = 1, shift = None, seq = 5):
        print("Reading Mesonet File")
        last_year = False
        train_y_all = pd.DataFrame(columns = self.test_feature_list)

        name_list = self.feature_list_full
        train_x_all = pd.DataFrame(columns=self.feature_list_full_value)

        # alabama data
        for yd in range(year_duration):    # year
            if yd == year_duration-1:
                last_year = True
            else:
                last_year = False

            curr_year = str(int(year)+yd)
            self.data_path = self.data_path_0 + curr_year+'/'
            year_folders = os.listdir(self.data_path)


            train_y = pd.DataFrame(columns = self.test_feature_list)
            train_x = pd.DataFrame(columns = name_list)
            
            file_count = 0
            read_flag = False

            for f_alabma in year_folders:
                
                
                if start_date and not read_flag:
                    if not f_alabma[4:] == str(start_date):
                        continue
                    read_flag=True


                record_path = self.data_path+f_alabma+'/'
                station_records = os.listdir(record_path)

                for r in station_records:
                    
                    r_split = r.split('_')
                    if not self.station == r_split[1]:
                        continue

                    station_record = record_path+r
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

                    
                    station_data = station_data.fillna(0.0001)
                    station_data = station_data.replace('--', 0.0001)       # replace -- symbol to 0
                    station_data_t = station_data_t.fillna(0.0001)
                    station_data_t = station_data_t.replace('--', 0.0001)       # replace -- symbol to 0


                    train_x = train_x.append(station_data, ignore_index=True, sort=False)
                    train_y = train_y.append(station_data_t, ignore_index=True, sort=False)

                    
                print(train_x.shape, train_y.shape, file_count)
                if file_count == duration-1:
                    print("Combining data")
                    # curr_idx = None
                    # predict_idx = None
                    # new_train_x = pd.DataFrame(columns=name_list)
                    # new_train_y = pd.DataFrame(columns = self.test_feature_list)

                    full_train_date = []

                    start_date_time = datetime.strptime(curr_year+start_date, '%Y%m%d')
                    curr_date_time_date = start_date_time
                    for d in range(duration):
                        curr_date_time_date = start_date_time + timedelta(days=d)
                        day_of_year = curr_date_time_date.date()-datetime.strptime(curr_year, '%Y').date()
                        for h in range(24):
                            curr_date_time_hour = curr_date_time_date + timedelta(hours=h)
                            for m in range(60):
                                curr_date_time_minute = curr_date_time_hour + timedelta(minutes=m)
                                full_train_date.append([curr_date_time_minute.year,curr_date_time_minute.day, day_of_year.days+1, curr_date_time_minute.hour, curr_date_time_minute.minute])
                    
                    full_train_date_df = pd.DataFrame(full_train_date, columns=['Year','DayOfMon','DayOfYear','Hour','Minute'])

                    load_train_date = train_x.loc[:,'Year':'Minute']

                    load_train_date_y = pd.concat([load_train_date, train_y], axis=1, sort=False)   # after average

                    full_train_x = pd.merge(full_train_date_df, train_x,  how='left', on=['Year','DayOfMon','DayOfYear','Hour','Minute'])

                    full_train_y = pd.merge(full_train_date_df, load_train_date_y,  how='left', on=['Year','DayOfMon','DayOfYear','Hour','Minute'])


                    full_train_x.fillna(method='ffill', inplace=True)
                    full_train_y.fillna(method='ffill', inplace=True)

                    print(full_train_x)


                    train_x_date = full_train_x.loc[::5,'Year':'Minute']
                    train_x_average = full_train_x.loc[:,self.feature_list_full_value].groupby(np.arange(len(full_train_x))//seq).mean()

                    # train_x_average.reset_index(drop=True, inplace=True)

                    # train_x_average = pd.concat([train_x_date, train_x_value], axis=1, sort=False)   # after average

                    train_y_average = full_train_y.loc[:,self.test_feature_list].groupby(np.arange(len(full_train_y))//seq).mean()
                    # train_y_average.reset_index(drop=True, inplace=True)

                    count_row = train_x_average.shape[0]
                    train_x_average = train_x_average.head(count_row-6*(60//seq))
                    train_x_average.reset_index(drop=True, inplace=True)
                    train_y_average = train_y_average.head(count_row-6*(60//seq))
                    train_y_average.reset_index(drop=True, inplace=True)

                    train_x_all = pd.concat([train_x_all, train_x_average], ignore_index=True, sort=False)
                    train_y_all = pd.concat([train_y_all, train_y_average], ignore_index=True, sort=False)


                    if not last_year:
                        break
                    else:
                        train_x_all = train_x_all.fillna(0.0001)
                        train_x_all = train_x_all.replace('--', 0.0001)
                        train_y_all = train_y_all.fillna(0.0001)
                        train_y_all = train_y_all.replace('--', 0.0001)


                        print(train_x_all.shape,train_y_all.shape)
                        return train_x_all, train_y_all
                file_count += 1
                

        return None, None
      
    def read_wrf_file(self, year = '2017', year_duration = 1, start_date = "0101", duration = 1, shift = None, seq = 5):
        last_year = False
 
        name_list = self.wrf_feature_list_full
        train_x_all = pd.DataFrame(columns=self.wrf_feature_list_value)
    
        # alabama data
        for yd in range(year_duration):    # year
            if yd == year_duration-1:
                last_year = True
            else:
                last_year = False

            curr_year = str(int(year)+yd)
            self.data_path = self.wrf_data_path_0 + curr_year+'/'
            year_folders = sorted(os.listdir(self.data_path))

            train_x = pd.DataFrame(columns=name_list)
            
            file_count = 0
            read_flag = False

            for f_alabma  in  year_folders:
                if start_date and not read_flag:
                    if not f_alabma[4:] == str(start_date):
                        continue
                    read_flag=True

                record_path = self.data_path+f_alabma+'/'
                station_records = os.listdir(record_path)

                for r in station_records:
                    r_split = r.split('_')
                    if not self.station == r_split[1]:
                        continue

                    station_record = record_path+r
                    try:
                        station_data = pd.read_csv(station_record)
                    except:
                        continue

                    # print(station_data)

                    for column in station_data.columns:
                        
                        if not column in name_list:
                            del station_data[column]
                        # else:
                            # station_data[column].isin(['--'])
                    
                    

                    station_data = station_data.fillna(0.0001)
                    station_data = station_data.replace('--', 0.0001)       # replace -- symbol to 0

                    train_x = train_x.append(station_data, ignore_index=True, sort=False)                    
   

                if file_count == duration-1:
                    train_x = train_x.astype({'cloud_base_pressure': 'float64','cloud_top_pressure': 'float64'})
                    train_v = train_x.loc[:,self.wrf_feature_list_value]
                    train_x_average = train_v.groupby(np.arange(len(train_x))//seq).mean()

                    count_row = train_x_average.shape[0]
                    train_x_average = train_x_average.tail(count_row-6*(60//seq))
                    train_x_average.reset_index(drop=True, inplace=True)
                    train_x_all = pd.concat([train_x_all, train_x_average], ignore_index=True, sort=False)


                    if not last_year:
                        break
                    else:
                        train_x_all = train_x_all.fillna(0.0001)
                        train_x_all = train_x_all.replace('--', 0.0001)

                        

                        return train_x_all

                file_count += 1

        return None





    def pre_seq(self, x, y, encoder_seq = 12,  decoder_seq = 6):

        input_texts = x
        target_texts = y
        input_len = len(x)

        max_encoder_seq_len = encoder_seq
        max_decoder_seq_len = decoder_seq+1
        # num_encoder_len = len(x[0])
        # num_decoder_len = len(y[0])

        num_encoder_shape = x.shape
        num_encoder_len = num_encoder_shape[1]  #  TODO: one more control


        num_decoder_len = 1+2  # 2 control bit, -2: start, -1: end

        encoder_input_data = np.zeros((input_len, max_encoder_seq_len, num_encoder_len), dtype='float32')
        decoder_input_data = np.zeros((input_len, max_decoder_seq_len, num_decoder_len), dtype='float32')
        decoder_target_data = np.zeros((input_len, max_decoder_seq_len, num_decoder_len), dtype='float32')


        eff_input_len = 0
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):

            for t in range(0, max_encoder_seq_len):
                if i+t+encoder_seq < input_len:
                    if t == 0 and i+t+encoder_seq*2 < input_len:
                        eff_input_len += 1

                    encoder_input_data[i, t, ] = input_texts[i+t]


            for t in range(0, max_decoder_seq_len):
                if i+t+encoder_seq < input_len:
                    if t == 0:
                        decoder_input_data[i, t, -2] = 1
                        decoder_target_data[i, t, 0] = target_texts[i+t+encoder_seq]

                    else:
                        decoder_input_data[i, t, 0] = target_texts[i+t+encoder_seq-1]
                        decoder_target_data[i, t, 0] = target_texts[i+t+encoder_seq]


            if i+t+encoder_seq+1 < input_len:
                decoder_target_data[i, t, -1] = 1
        

        encoder_input_data = encoder_input_data[:eff_input_len]
        decoder_input_data = decoder_input_data[:eff_input_len]
        decoder_target_data = decoder_target_data[:eff_input_len]
        # decoder_target_data[-1][-1] = 1
        
        return encoder_input_data, decoder_input_data, decoder_target_data

 

    def draw_lstm_without_e(self, tru, pre, scaler,file_label,  lstm_path = None):
        '''
        draw predciton and ground dataset with standard deviation
        '''

        tru = scaler.inverse_transform(tru)
        pre = scaler.inverse_transform(pre)

        idx = 0
        test_len = 2400
        tru_10 = []
        pre_10 = []
        for i in range(len(tru)):
            tru_10.append(tru[i][idx])
            pre_10.append(pre[i][idx])

        pickle.dump(tru, open(lstm_path+file_label+'.true', "wb"))
        pickle.dump(pre, open(lstm_path+file_label+'.pre', "wb"))


        # fig, ax1 = plt.subplots()

        # x = [i*5 for i in range(0, len(tru_10))]
        # lns1 = plt.plot(x, tru_10, 'r', label='true', linewidth=2)
        # ax2 = ax1.twinx()
        # lns2 = ax2.plot(x, pre_10, label='prediction')

        # lns = lns1+lns2
        # labs = [l.get_label() for l in lns]
        # ax2.legend(lns, labs, loc=0, frameon=False)

        # plt.ylabel("15 cm mositor")
        # plt.xlabel("minutes")        
        # plt.savefig(lstm_path+file_label+'.pdf')
        # plt.show() 

