'''
This spider extract data from http://deltaweather.extension.msstate.edu/

still have bugs, can not fetch the file, solve it later
'''

import json
import requests
from absl import app
from absl import flags
from datetime import datetime, timedelta



FLAGS = flags.FLAGS
flags.DEFINE_string('station', default='Stoneville W', help='the station name')
flags.DEFINE_string('start_date', default='2023-01-01', help='the start date')
flags.DEFINE_string('end_date', default='2023-01-02', help='the end date')


class Extractor():
    def __init__(self):
        self.version = 1.0
        self.info = 'the initial verstion that extract data for deltaweather'
        self.data_path = 'data/'
        self.url = 'http://deltaweather.extension.msstate.edu/hourly-report'

        self.station_id = None
        self.start_date = None
        self.end_date = None
        # self.to_view = "DATE=DATE,TIME=TIME,JDATE=JDATE,AIRMAX=AIRMAX,AIRMAXT=AIRMAXT,AITMIN=AITMIN,AITMINT=AITMINT,AIROB=AIROB,RHUMMAX=RHUMMAX,RHUMMAXT=RHUMMAXT,RHUMNIN=RHUMNIN,RHUMNINT=RHUMNINT,RHUMOB=RHUMOB,PRECIP=PRECIP,WSPEED=WSPEED,WDIRECTION=WDIRECTION,SRAD=SRAD,ST2=ST2,ST2OB=ST2OB,ST4=ST4,ST4OB=ST4OB,"
        self.op = "Export Data"
        self.form_build_id = "form-y3EwnryX6s6PoqN_tUxkqWfkqtghquj03YChgbWw2A8"
        self.form_id= "export_hourly_form"
        self.station_to_id = self.read_station_to_id()

    def read_station_to_id(self):
        station_to_id = {}
        with open('station_to_id', 'r') as file_reader:
            for line in file_reader:
                line_split = line.strip().split('\t')
                station_to_id[line_split[0]] = line_split[1]
        return station_to_id


    def set_parameter(self, station, start_date, end_date):
        self.station_id = self.station_to_id[station]
        self.start_date = start_date
        self.end_date = end_date
    
        
    def run(self):
        file_name = self.station_id+'+'+self.start_date+'+'+self.end_date
        data = {'station': self.station_id, 
                'fileName': file_name, 
                'startDate': self.start_date,
                'endDate': self.end_date,
                'to_view': self.to_view,
                'op': self.op,
                'form_build_id': self.form_build_id,
                'form_id': self.form_id
                }
        # data = 'station=DREC-2034&fileName=A%26O+Farm-hourly-2023-01-10-through-2023-01-09&startDate=2023-01-09&endDate=2023-01-10&to_view=DATE%3DDATE%2CTIME%3DTIME%2CJDATE%3DJDATE%2CAIRMAX%3DAIRMAX%2CAIRMAXT%3DAIRMAXT%2CAITMIN%3DAITMIN%2CAITMINT%3DAITMINT%2CAIROB%3DAIROB%2CRHUMMAX%3DRHUMMAX%2CRHUMMAXT%3DRHUMMAXT%2CRHUMNIN%3DRHUMNIN%2CRHUMNINT%3DRHUMNINT%2CRHUMOB%3DRHUMOB%2CPRECIP%3DPRECIP%2CWSPEED%3DWSPEED%2CWDIRECTION%3DWDIRECTION%2CSRAD%3DSRAD%2CST2%3DST2%2CST2OB%3DST2OB%2CST4%3DST4%2CST4OB%3DST4OB%2C&op=Export+Data&form_build_id=form-y3EwnryX6s6PoqN_tUxkqWfkqtghquj03YChgbWw2A8&form_id=export_hourly_form'
    
        headers = {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
                    "Accept-Encoding": "gzip, deflate",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Cache-Control": "max-age=0",
                    "Connection": "keep-alive",
                    "Content-Length": "598",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Host": "deltaweather.extension.msstate.edu",
                    "Origin": "http://deltaweather.extension.msstate.edu",
                    "Referer": "http://deltaweather.extension.msstate.edu/hourly-report",
                    "Upgrade-Insecure-Requests": "1",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        }
        r = requests.post(self.url, files=json.dumps(data), headers=headers)
        # r = requests.post(self.url, headers=headers, files=data)
        # with open(file_name, 'w') as file_out:

        print(r.text)




def main(_):
    spider = Extractor()

    station = FLAGS.station
    # start_date = datetime.strptime(FLAGS.start_date, '%Y-%m-%d')
    # end_date = datetime.strptime(FLAGS.end_date, '%Y-%m-%d')
    spider.set_parameter(station, FLAGS.start_date, FLAGS.end_date)
    spider.run()




if __name__ == '__main__':
    app.run(main)