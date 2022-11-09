import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.geometry import Point
from shapely.prepared import prep

import datetime as dt


def kriging(seldate=dt.date(2018, 6, 1), dattype="rural", return_vgram=False):
    rural_data = pd.read_csv("Rural_NaNdeleted.csv", delimiter = ",", low_memory=False)
    urban_data = pd.read_csv("Urban_NaNdeleted.csv", delimiter = ",", low_memory=False)

    if dattype=="rural":
        lats = rural_data["lat"].unique()
        lons = rural_data["lon"].unique()
    elif dattype=="urban":
        lats = urban_data["lat"].unique()
        lons = urban_data["lon"].unique()

    # first, pick a single time and focus on rural data
    if dattype=="rural":
        date0 = rural_data.iloc[0].date
        date0 = rural_data[pd.to_datetime(rural_data.date).dt.date==seldate].iloc[0].date
        data0 = rural_data[rural_data["date"]==date0]
        
    elif dattype=="urban":
        date0 = urban_data.iloc[0].date
        data0 = urban_data[urban_data["date"]==date0]

    no2_0 = data0.no2.to_numpy()
    lat_0 = data0.lat.to_numpy()
    lon_0 = data0.lon.to_numpy()

    n_points = len(lat_0)
    n_pairs = int(n_points*(n_points-1)/2)

    distances = np.zeros(n_pairs)
    no2_diffs = np.zeros(n_pairs)

    ind=0
    for i in range(n_points):
        for j in range(i+1, n_points):
            distances[ind] = ((lat_0[i]-lat_0[j])**2 + (lon_0[i]-lon_0[j])**2)**0.5
            no2_diffs[ind] = 0.5 * (no2_0[i]-no2_0[j])**2
            ind += 1

    # get experimental variogram from raw variogram with 5 intervals, spaced
    # such that an equal number of raw data are in each interval
    sorted_dists = np.array_split(np.sort(distances), 5)
    dist_sorted_inds = np.argsort(distances)
    dist_sorted_no2s = np.array_split(no2_diffs[dist_sorted_inds], 5)
    bins = np.array([(d[0]+d[-1])/2 for d in sorted_dists])
    exp_vari = np.array([np.mean(sno2) for sno2 in sorted_dists])

    # generate linear variogram model
    FIT = np.polyfit(bins, exp_vari, 1)
    vari_fit = np.poly1d(FIT)
    print(FIT)

    h_hires = np.linspace(0, 4, 1000)

    # get model variogram matrix for kriging
    distance_matrix = ((np.stack([lat_0 for i in range(n_points)], axis=0)
                -np.stack([lat_0 for i in range(n_points)], axis=1))**2 
                + (np.stack([lon_0 for i in range(n_points)], axis=0)
                -np.stack([lon_0 for i in range(n_points)], axis=1))**2)**0.5

    # variogram_matrix = (np.stack([rural_no2_0 for i in range(n_points)], axis=0)
    #               -np.stack([rural_no2_0 for i in range(n_points)], axis=1))**2 * 0.5

    variogram_matrix = vari_fit(distance_matrix)

    print("trace: ", np.trace(variogram_matrix)) # should be zero
    print("Condition number is finite: ", np.isfinite(np.linalg.cond(variogram_matrix)))
    print("Determinant: ", np.linalg.det(variogram_matrix))

    inverse_variogram_matrix = np.linalg.inv(variogram_matrix)

    # generate mesh of points we want to interpolate
    minlat = 50.5
    maxlat = 54
    latgrid = np.arange(minlat, maxlat, 0.01)

    minlon = 3
    maxlon = 7.5
    longrid = np.arange(minlon, maxlon, 0.01)

    XX, YY = np.meshgrid(longrid, latgrid)

    interp_no2 = np.zeros(XX.shape)

    # find weights
    for i, x in enumerate(longrid):
        for j, y in enumerate(latgrid):
            interp_distances = np.array([((x-lons[k])**2+(y-lats[k])**2)**0.5 for k in range(n_points)])
            b = vari_fit(interp_distances)

            weights = np.matmul(inverse_variogram_matrix, b)
            # weights /= np.sum(weights) # normalisation

            interp_no2[j, i] = np.dot(weights, no2_0)

    if return_vgram:
        return XX, YY, interp_no2, date0, lons, lats, distances, no2_diffs, bins, exp_vari, h_hires, vari_fit
    else:
        return XX, YY, interp_no2, date0, lons, lats

def kriging_av(date0="Annual", dattype="rural", return_vgram=False):
    """Returns average over summer months"""
    rural_data = pd.read_csv("Rural_NaNdeleted.csv", delimiter = ",", low_memory=False)
    urban_data = pd.read_csv("Urban_NaNdeleted.csv", delimiter = ",", low_memory=False)

    if date0=="JJA":
        rural_data = rural_data[(pd.to_datetime(rural_data.date).dt.month==6) | (pd.to_datetime(rural_data.date).dt.month==7) | (pd.to_datetime(rural_data.date).dt.month==8)]
        urban_data = urban_data[(pd.to_datetime(urban_data.date).dt.month==6) | (pd.to_datetime(urban_data.date).dt.month==7) | (pd.to_datetime(urban_data.date).dt.month==8)]
    elif date0=="DJF":
        rural_data = rural_data[(pd.to_datetime(rural_data.date).dt.month==1) | (pd.to_datetime(rural_data.date).dt.month==2) | (pd.to_datetime(rural_data.date).dt.month==12)]
        urban_data = urban_data[(pd.to_datetime(urban_data.date).dt.month==1) | (pd.to_datetime(urban_data.date).dt.month==2) | (pd.to_datetime(urban_data.date).dt.month==12)]

    # taking average over appropriate time period
    if dattype=="rural":
        lats = rural_data["lat"].unique()
        lons = rural_data["lon"].unique()

        sites = rural_data.site.unique()
        lats_, lons_, no2_ = np.zeros(len(sites)), np.zeros(len(sites)), np.zeros(len(sites))
        for i, site in enumerate(sites):
            atsite = rural_data[rural_data.site==site]
            atsite_simple = atsite.drop(['Unnamed: 0', 'date', 'date_end', 'site', 'name'], axis=1)
            row = atsite_simple.mean(axis=0)
            lats_[i] = row.lat
            lons_[i] = row.lon
            no2_[i] = row.no2

        data0 = pd.DataFrame({'lat': lats_, 'lon': lons_, 'no2': no2_})

    elif dattype=="urban":
        lats = urban_data["lat"].unique()
        lons = urban_data["lon"].unique()

        sites = urban_data.site.unique()
        lats_, lons_, no2_ = np.zeros(len(sites)), np.zeros(len(sites)), np.zeros(len(sites))
        for i, site in enumerate(sites):
            atsite = rural_data[urban_data.site==site]
            atsite_simple = atsite.drop(['Unnamed: 0', 'date', 'date_end', 'site', 'name'], axis=1)
            row = atsite_simple.mean(axis=0)
            lats_[i] = row.lat
            lons_[i] = row.lon
            no2_[i] = row.no2

        data0 = pd.DataFrame({'lat': lats_, 'lon': lons_, 'no2': no2_})

    

    # # first, pick a single time and focus on rural data
    # if dattype=="rural":
    #     date0 = rural_data.iloc[0].date
    #     data0 = rural_data[rural_data["date"]==date0]
    # elif dattype=="urban":
    #     date0 = urban_data.iloc[0].date
    #     data0 = urban_data[urban_data["date"]==date0]

    # print(pd.to_datetime(rural_data.date).dt.month)

    no2_0 = data0.no2.to_numpy()
    lat_0 = data0.lat.to_numpy()
    lon_0 = data0.lon.to_numpy()

    n_points = len(lat_0)
    n_pairs = int(n_points*(n_points-1)/2)

    distances = np.zeros(n_pairs)
    no2_diffs = np.zeros(n_pairs)

    ind=0
    for i in range(n_points):
        for j in range(i+1, n_points):
            distances[ind] = ((lat_0[i]-lat_0[j])**2 + (lon_0[i]-lon_0[j])**2)**0.5
            no2_diffs[ind] = 0.5 * (no2_0[i]-no2_0[j])**2
            ind += 1

    # get experimental variogram from raw variogram with 5 intervals, spaced
    # such that an equal number of raw data are in each interval
    sorted_dists = np.array_split(np.sort(distances), 5)
    dist_sorted_inds = np.argsort(distances)
    dist_sorted_no2s = np.array_split(no2_diffs[dist_sorted_inds], 5)
    bins = np.array([(d[0]+d[-1])/2 for d in sorted_dists])
    exp_vari = np.array([np.mean(sno2) for sno2 in sorted_dists])

    # generate linear variogram model
    FIT = np.polyfit(bins, exp_vari, 1)
    vari_fit = np.poly1d(FIT)
    print(FIT)

    h_hires = np.linspace(0, 4, 1000)

    # get model variogram matrix for kriging
    distance_matrix = ((np.stack([lat_0 for i in range(n_points)], axis=0)
                -np.stack([lat_0 for i in range(n_points)], axis=1))**2 
                + (np.stack([lon_0 for i in range(n_points)], axis=0)
                -np.stack([lon_0 for i in range(n_points)], axis=1))**2)**0.5

    # variogram_matrix = (np.stack([rural_no2_0 for i in range(n_points)], axis=0)
    #               -np.stack([rural_no2_0 for i in range(n_points)], axis=1))**2 * 0.5

    variogram_matrix = vari_fit(distance_matrix)

    print("trace: ", np.trace(variogram_matrix)) # should be zero
    print("Condition number is finite: ", np.isfinite(np.linalg.cond(variogram_matrix)))
    print("Determinant: ", np.linalg.det(variogram_matrix))

    inverse_variogram_matrix = np.linalg.inv(variogram_matrix)

    # generate mesh of points we want to interpolate
    minlat = 50.5
    maxlat = 54
    latgrid = np.arange(minlat, maxlat, 0.01)

    minlon = 3
    maxlon = 7.5
    longrid = np.arange(minlon, maxlon, 0.01)

    XX, YY = np.meshgrid(longrid, latgrid)

    interp_no2 = np.zeros(XX.shape)

    # find weights
    for i, x in enumerate(longrid):
        for j, y in enumerate(latgrid):
            interp_distances = np.array([((x-lons[k])**2+(y-lats[k])**2)**0.5 for k in range(n_points)])
            b = vari_fit(interp_distances)

            weights = np.matmul(inverse_variogram_matrix, b)
            # weights /= np.sum(weights) # normalisation

            interp_no2[j, i] = np.dot(weights, no2_0)

    if return_vgram:
        return XX, YY, interp_no2, date0, lons, lats, distances, no2_diffs, bins, exp_vari, h_hires, vari_fit
    else:
        return XX, YY, interp_no2, date0, lons, lats

def plot_map(XX, YY, interp_no2, date0, lons, lats, dattype="rural"):
    # Downloaded from https://gadm.org/download_country.html
    fname = 'gadm41_NLD_1.shp' #0 is country border, 1 is provinces, 2 is municipalities

    adm1_shapes = list(shpreader.Reader(fname).geometries())
    borders = list(shpreader.Reader('gadm41_NLD_0.shp').geometries())
    borders_prep = [prep(b_) for b_ in borders]
    nb = len(borders_prep)

    regions_prep = [prep(r_) for r_ in adm1_shapes]
    regions_prep.pop(5)
    regions_prep.pop(11)
    nr = len(regions_prep)

    points = np.empty(XX.shape, dtype=object)
    contains = np.empty((XX.shape[0], XX.shape[1], nr), dtype=bool)
    for i in range(XX.shape[0]):
        print(i)
        for j in range(XX.shape[1]):
            contains[i, j, :] = [b_.contains(Point(XX[i, j], YY[i, j])) for b_ in regions_prep]
            # contains[i, j] = regions_prep[14].contains(Point(XX[i, j], YY[i, j]))

    contains = np.any(contains, axis=2)
    masked_no2 = np.where(contains, interp_no2, np.nan)

    fig = plt.figure(figsize=(8,6))
    ax = plt.axes()

    plt.title(str(date0)+" NO2 Concentrations (Kriging)")
    # ax.coastlines(resolution='10m')

    ax.set_extent([3, 7.5, 50.5, 54])

    if dattype=="rural":
        ax.plot(lons, lats, ".", markersize=10, label="Rural Data")
    else:
        ax.plot(lons, lats, ".", markersize=10, label="Urban Data")

    no2mesh = ax.pcolormesh(XX, YY, masked_no2, cmap=mpl.colormaps['Blues'].reversed())

    ax.add_geometries(adm1_shapes, linewidth=0.5,
                    edgecolor='white', facecolor='None', alpha=1)

    cbar = fig.colorbar(no2mesh, ax=ax)
    cbar.set_label("NO2 Concentration [$\mu g m^{-3}$]")

    ax.legend()
    plt.show()


XX, YY, interp_no2, date0, lons, lats = kriging_av(date0="DJF")
# XX, YY, interp_no2, date0, lons, lats = kriging()
plot_map(XX, YY, interp_no2, date0, lons, lats)