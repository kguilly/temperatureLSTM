import requests
from bs4 import BeautifulSoup

url = "http://deltaweather.extension.msstate.edu/stations"
r = requests.get(url)
soup = BeautifulSoup(r.text, "html.parser")

tab = soup.find("table",{"class":"views-table cols-11 table"})

station_tr = tab.find_all()
with open('station_to_id', 'w') as file_writer:
    for station_tr in tab.find_all('td', 'views-field views-field-title'):
        a = station_tr.find('a', href=True)
        station_str = a.text
        file_writer.write(station_str+'\t')
        station_url = "http://deltaweather.extension.msstate.edu"+a['href']
        print(station_url)
        station_page = requests.get(station_url)

        station_soup = BeautifulSoup(station_page.text, "html.parser")
        for find_id_div in station_soup.find_all('div', {"class":"field-item even"}):
            if find_id_div.text.startswith('DREC') | find_id_div.text.startswith('VT') | find_id_div.text.startswith('DS') | find_id_div.text.startswith('COOP'):
                print(find_id_div.text)
                file_writer.write(find_id_div.text+'\n')
                file_writer.flush()
        # break
    


# print(tab)
