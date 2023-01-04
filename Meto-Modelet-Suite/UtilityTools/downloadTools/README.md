# PREFER Tools

This repository contains tools and data for supporting the [PREFER](http://prefer-nsf.org) effort.

## Aeris

Aeris Weather provides historical weather data with a subscription. The `aeris.py` python script can list stations near a location and fetch data.

### Find Stations

Example 1. Find stations near the location of Foley, Alabama:

`python3 aeris.py stations --location 'foley,al'`

The output contains the station ID, latitude and longitude, and the station city and state.

```id: usw00063869, location: 30.5486,-87.875, elevation: 29 ft, name: fairhope 3 ne, state: al, distance: 0 mi
id: usc00012813, location: 30.5467,-87.88, elevation: 7 ft, name: fairhope 2 ne, state: al, distance: 0.325 mi
id: usc00016988, location: 30.5653,-87.701, elevation: 49.1 ft, name: robertsdale, state: al, distance: 10.417 mi
id: usw00013838, location: 30.6264,-88.068, elevation: 7.9 ft, name: mobile dwtn ap, state: al, distance: 12.676 mi
id: usc00010583, location: 30.8839,-87.785, elevation: 82.6 ft, name: bay minette, state: al, distance: 23.776 mi
...
```

### Fetch Station Data

The `fetch` command retrieves historical data for a specific year.

Example 1. Obtain Aeris data from 30.5486,-87.875 for 2017 and show verbose output:

`python3 aeris.py fetch --location '30.5486,-87.875' --year 2017 --verbose`

The output is:

```Creating aeris_data/alabama/2017/20170101/alabama_MIDAR152_2017_01_01.csv
Creating aeris_data/alabama/2017/20170102/alabama_MIDAR152_2017_01_02.csv
Creating aeris_data/alabama/2017/20170103/alabama_MIDAR152_2017_01_03.csv
Creating aeris_data/alabama/2017/20170104/alabama_MIDAR152_2017_01_04.csv
Creating aeris_data/alabama/2017/20170105/alabama_MIDAR152_2017_01_05.csv
Creating aeris_data/alabama/2017/20170106/alabama_MIDAR152_2017_01_06.csv

...
```
The filename consists of the state, the station ID, and the date.

Note that the data is placed in the `aeris_data` folder in the current directory.

## Spider

Spider retrieves HRRR model data with the `hrrr.py` script from [http://home.chpc.utah.edu/~u0553130/Brian_Blaylock/].

### Retrieve HRRR Model Data

Example 1. Obtain HRRR data for January 2-3, 2019:

`python3 hrrr.py --begin_date 20190102 --end_date 20190103`

The output shows the data as it downloads from the server:

```data/2018/20180101/hrrr.20180101.00.00.grib2
data/2018/20180101/hrrr.20180101.01.00.grib2

...
```

