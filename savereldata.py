import pandas as pd
import numpy as np

rural_data = pd.read_csv("rural_data.csv", delimiter = ";", low_memory=False)
urban_data = pd.read_csv("urban_data.csv", delimiter = ";", low_memory=False)

# drop irrelevant columns and nans
urban_data = urban_data.drop(labels=['organisation', 'type', 'site_eu', 'type_alt', 
                             'type_airbase', 'municipality.name', 'province.name', 
                             'air.quality.area', 'bc', 'co', 'nh3', 'ox', 'pm25', 'so2', 
                             'benzeen', 'h2s', 'meta_paraxyleen', 'tolueen', 'ufp', 'ff', 
                             'fx', 't10n', 'td', 'sq', 'dr', 'vv', 'no', 'nox', 'o3', 'ws',
                             't', 'q', 'hourly_rain', 'p', 'n', 'rh', 'wd', 'pm10'], axis=1)
urban_data = urban_data.dropna(axis=0)

rural_data = rural_data.drop(labels=['organisation', 'type', 'site_eu', 'type_alt', 
                             'type_airbase', 'municipality.name', 'province.name', 
                             'air.quality.area', 'bc', 'co', 'nh3', 'ox', 'pm25', 'so2', 
                             'benzeen', 'h2s', 'meta_paraxyleen', 'tolueen', 'ufp', 'ff', 
                             'fx', 't10n', 'td', 'sq', 'dr', 'vv', 'no', 'nox', 'o3', 'ws',
                             't', 'q', 'hourly_rain', 'p', 'n', 'rh', 'wd', 'pm10'], axis=1)

sites = rural_data.site.unique()

rural_data = rural_data[(rural_data["site"]!="722") & (rural_data["site"]!="934") 
                        & (rural_data["site"]!="NL49556") 
                        & (rural_data["site"]!="NL01437")] # drop sites which have no or sparse no2 data

# print(rural_data[rural_data["date"]=="2018-05-05 14:00:00"])
# print(rural_data.site.unique())
# rural_data = rural_data.dropna(axis=0)
# print(rural_data.site.unique())

# print(rural_data[rural_data["date"]=="2018-05-05 14:00:00"])
# print(rural_data[rural_data["site"]=="NL49556"].no2)
# nan_inds = pd.isnull(rural_data).any(1).to_numpy().nonzero()[0]
# nan_inds = rural_data.loc[pd.isna(rural_data["no2"]), :].index
nan_inds = pd.isnull(rural_data).any(1).to_numpy()
nan_dates = rural_data.date[nan_inds].unique()
# for nandate in nan_dates:
#     print(nandate)
print(len(nan_dates))

rural_data = rural_data[~rural_data.date.isin(nan_dates)]
# print(len(nan_dates))
# rural_data = rural_data[rural_data.date!=nan]

# ensure same start and end dates for all stations
# urban_data = urban_data[(urban_data["date"]>="2018-01-04 14:00:00") & (urban_data["date"]<="2018-12-06 13:00:00")]
# rural_data = rural_data[(rural_data["date"]>="2018-01-04 14:00:00") & (rural_data["date"]<="2018-12-06 13:00:00")]

# Save
urban_data.to_csv("Urban_NaNdeleted.csv", sep=',')
rural_data.to_csv("Rural_NaNdeleted.csv", sep=',')