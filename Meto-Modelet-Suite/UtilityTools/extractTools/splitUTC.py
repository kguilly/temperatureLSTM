import csv
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path       
import re

# point at "/home/canadmin/prefer/WRF/" 
data_path = "Kentucky/"

#move to 2017 and 2018           
year_folders = sorted(os.listdir(data_path))

print(year_folders)
#'''

for f_alabma  in  year_folders:
    print(f_alabma)
    newPath = data_path + f_alabma+'/'
    folders = sorted(os.listdir(newPath))

    for location  in  folders:
        file_name = newPath+location
        print(file_name)
        fName = file_name.split('.')

        if not fName[1] =='txt':
            continue

        df = pd.read_csv(file_name)
        #print(df)
        df = pd.concat([df.drop('UTME', axis = 1),(df.UTME.str.split('-').str[:3].apply(pd.Series).rename(columns={0:'Year', 1:'Month', 2:'day'}))], axis = 1)
        df = pd.concat([df.drop('day', axis = 1),(df.day.str.split(' ').str[:2].apply(pd.Series).rename(columns={0:'Day', 1:'time'}))], axis = 1)
        df = pd.concat([df.drop('time', axis = 1),(df.time.str.split(':').str[:3].apply(pd.Series).rename(columns={0:'Hour', 1:'Minute',2:'Sec'}))], axis = 1)

        #fName = file_name.split('.')
        df.to_csv(fName[0]+'.csv', index=False)
        os.remove(file_name)
        # df.to_csv(file_name, header=['Year','Month','Date','Hour','u_500hpa', 'v_500hpa', 'cloud_base_pressure', 'cloud_top_pressure', '3000_h', '1000_h', 'ground_mis', 'u_10m', 'v_10m', 'u_250hpa', 'v_250hpa', 'u_80m', 'v_80m', 'surface_pressure'], index=False)
