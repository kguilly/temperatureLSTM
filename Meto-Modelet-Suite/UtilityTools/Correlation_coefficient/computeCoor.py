import pandas as pd
import numpy as np
import sys
from sklearn.feature_selection import mutual_info_regression

def df_shifted(df, target=None, lag=0):
    if not lag and not target:
        return df       
    new = {}
    for c in df.columns:
        if c == target:
            new[c] = df[target]
        else:
            new[c] = df[c].shift(periods=lag)
    return  pd.DataFrame(data=new)



def custom_mi_reg(a, b):
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    return  mutual_info_regression(a, b)[0] # should return a float value
    
    
#df_mi = df.corr(method=custom_mi_reg) how to call the method

'''
#Alabama mesonet
#read data set
df = pd.read_csv('agricola_2019-01-01_2020-01-01.csv')

#list required features

features = ['AirT_2m','AirT_1pt5m','PTemp','IRTS_Body','SoilSfcT','IRTS_Trgt','Temp_C','DayOfYear','SoilCond_tc','eI_tc','AirT_10m','wfv','eR_tc','eR','eI','SoilCond','QuantRadn','TotalRadn','Vitel_100cm_c','WndSpd_10m_WVc_4','AirT_9pt5m','WndSpd_2m_WVc_4','WndSpd_10m_WVc_3','WndSpd_2m_WVc_3','WndDir_10m','WndDir_2m','Hour','WndSpd_2m_Std','SoilT_5cm','SoilT_10cm','SoilT_20cm','SoilT_50cm','SoilT_100cm','WndSpd_10m_Std','NaCI',	'Minute','Precip_TX_Tot','Precip_TB3_Tot','Precip_TX_Today','Precip_TB3_Today','WndSpd_Vert_Max','Pressure_2','WndSpd_2m_Max','WndSpd_Vert','WndSpd_Vert_Tot','WndSpd_Vert_Min','WndSpd_10m_Max','Vitel_100cm_b','WndSpd_2m','WndSpd_2m_WVc_1','WndSpd_10m','WndSpd_10m_WVc_1','WndSpd_2m_WVc_2','WndSpd_10m_WVc_2','RH_10m','RH_2m','Pressure_1','Vitel_100cm_a','Vitel_100cm_d']

for feature in features:
    df_new = df_shifted(df, feature, lag=15) #Compute lag-15 (15 minutes, data set contains minutely spaced data set) for given feature
    df_new.dropna(inplace=True)
    
    #rel = df_new.corr() # for cross-correlation ranking    
    rel = df_new.corr(method=custom_mi_reg) # for mutual-information ranking
    
    rel =  rel.sort_values(by=feature, key=abs, ascending=False)
    rel.to_csv(feature+'.csv', columns=[feature])
    
#'''

'''
########### 
# Compute correlation between features of WRF data set 
# wrf_data_set.csv contains the extracted data set from WRF for a given station 
# from July 15, 2018 to November 30, 2019 (excluding 2019's Jan and Feb data) 
# It looks the scaling factor between data set in WRF is changed from that date

df = pd.read_csv('wrf_data_set.csv',low_memory=False)
print(df.shape)
df.dropna(inplace=True)
df.drop(columns = ['ATT102','ATT103','ATT104','ATT105','ATT106'], inplace = True) # these columns have a lot of missing data
print(df.shape)
    
#df.to_csv('Check_ATT107.csv',columns=['ATT102'])

#rel = df_new.corr() # for cross-correlation ranking
    
rel = df_new.corr(method=custom_mi_reg) # for mutual-information ranking
    
features = ['ATT1',	'ATT2',	'ATT3',	'ATT4',	'ATT5',	'ATT6',	'ATT7',	'ATT8',	'ATT9',	'ATT10',	'ATT11',	'ATT12',	'ATT13',	'ATT14',	'ATT15',	'ATT16',	'ATT17',	'ATT18',	'ATT19',	'ATT20',	'ATT21',	'ATT22',	'ATT23',	'ATT24',	'ATT25',	'ATT26',	'ATT27',	'ATT28',	'ATT29',	'ATT30',	'ATT31',	'ATT32',	'ATT33',	'ATT34',	'ATT35',	'ATT36',	'ATT37',	'ATT38',	'ATT39',	'ATT40',	'ATT41',	'ATT42',	'ATT43',	'ATT44',	'ATT45',	'ATT46',	'ATT47',	'ATT48',	'ATT49',	'ATT50',	'ATT51',	'ATT52',	'ATT53',	'ATT54',	'ATT55',	'ATT56',	'ATT57',	'ATT58',	'ATT59',	'ATT60',	'ATT61',	'ATT62',	'ATT63',	'ATT64',	'ATT65',	'ATT66',	'ATT67',	'ATT68',	'ATT69',	'ATT70',	'ATT71',	'ATT72',	'ATT73',	'ATT74',	'ATT75',	'ATT76',	'ATT77',	'ATT78',	'ATT79',	'ATT80',	'ATT81',	'ATT82',	'ATT83',	'ATT84',	'ATT85',	'ATT86',	'ATT87',	'ATT88',	'ATT89',	'ATT90',	'ATT91',	'ATT92',	'ATT93',	'ATT94',	'ATT95',	'ATT96',	'ATT97',	'ATT98',	'ATT99',	'ATT100',	'ATT101',	'ATT107',	'ATT108',	'ATT109',	'ATT110',	'ATT111',	'ATT112',	'ATT113',	'ATT114',	'ATT115',	'ATT116',	'ATT117',	'ATT118',	'ATT119',	'ATT120',	'ATT121',	'ATT122',	'ATT123',	'ATT124',	'ATT125',	'ATT126',	'ATT127',	'ATT128',	'ATT129',	'ATT130',	'ATT131',	'ATT132']

for feature in features:   
    rel =  rel.sort_values(by=feature, key=abs,ascending=False)
    rel.to_csv(feature+'.csv', columns=[feature])

#'''

#'''
########### 
#Kentucky
# Compute correlation between features of KentukyMesonet data set 
# BMBL_Barbourville-3-E_Knox.csv contains the data set Barbourville station 
# from Jan 1, 2020 to Dec 31, 2020 

df = pd.read_csv('BMBL_Barbourville-3-E_Knox.csv')
print(df.shape)
df.drop(columns = ['NET', 'STID', 'UTME'], inplace = True) 
df.dropna(inplace=True)
print(df.shape)

#list required features

features = ['TAIR',	'RELH',	'THMP',	'TDPT',	'WSPD',	'WDIR',	'WSMX',	'WDMX',	'WSMN',	'WSSD',	'WDSD',	'SRAD',	'PRCP']
for feature in features:
    df_new = df_shifted(df, feature, lag=12) #Compute lag-1 (15 minutes lag, data set contains record for every 15 minutes)for given feature (
    df_new.dropna(inplace=True)
    #rel = df_new.corr() # for cross-correlation ranking
    
    rel = df_new.corr(method=custom_mi_reg) # for mutual-information ranking
    rel =  rel.sort_values(by=feature, key=abs, ascending=False)
    rel.to_csv("KentukyMesonet/"+feature+'.csv', columns=[feature])
#'''