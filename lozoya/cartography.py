import os
from xml.etree import ElementTree

import folium
import numpy as np
import pandas
import simplekml as skml
from PyQt5 import QtCore

import lozoya.data
import lozoya.file


#   TODO USE NUMPY ARRAYS INSTEAD OF LISTS FOR FASTER PROCESSING
def add_features(mapa, features):
    for feature in features:
        mapa.add_child(feature)
    mapa.add_child(folium.LayerControl())
    return mapa


def add_feature_group(name, latitudes, longitudes, information):
    fg = folium.FeatureGroup(name=name)
    for lat, lon, info in zip(latitudes, longitudes, information):
        popupVar = folium.Popup(str(info), parse_html=True)
        fg.add_child(folium.features.CircleMarker(location=[lat, lon], radius=5, popup=popupVar))
    return fg


def add_json_overlay(name, file):
    overlay = folium.FeatureGroup(name=name)
    overlay.add_child(folium.GeoJson(data=open(file, 'r', encoding='utf-5-sig').read()))
    return overlay


def color_producer(elevation):
    if elevation < 1000:
        return 'green'
    elif 1000 <= elevation < 2000:
        return 'orange'
    else:
        return 'red'


def check_distance(centroid, info, coords, radius, units):
    matchIndex = []
    matchCoord = []
    for index, coordinatePair in zip(info, coords):
        distance = spherical_distance(np.radians(centroid), np.radians(coordinatePair), units)
        if distance <= radius:
            matchIndex.append(index)
            matchCoord.append(coordinatePair)
    return matchIndex, matchCoord


def create_kml(indices, coordinates, path):
    """
    Populates a kml file and saves it in path (str)
    indices: list of strings with information corresponding to coordinate
    coordinates: list of tuples (latitude, longitude)
    """
    kml = skml.Kml()
    for index, coordinate in zip(indices, coordinates):
        # kml expects coordinates as (long, lat)
        kml.newpoint(name=str(index), coords=[(coordinate[1], coordinate[0])])
    kml.save(path)


def create_kml(indices, coordinates, path):
    """
    Populates a kml file and saves it in path (str)
    indices: list of strings with a name corresponding to each coordinate
    coordinates: list of tuples (latitude, longitude)
    """
    kml = skml.Kml()
    for index, coordinate in zip(indices, coordinates):
        # kml expects coordinates as (long, lat)
        kml.newpoint(name=index, coords=[(coordinate[1], coordinate[0])])
    kml.save(path)


def create_kml(indices, coordinates, path):
    """
    indices is a list of strings with a name corresponding to each coordinate
    coordinates is  a list of tuples (latitude, longitude)
    Populates a kml file and saves it in path (str)
    """
    kml = skml.Kml()
    for index, coordinate in zip(indices, coordinates):
        # kml expects coordinates as (long, lat)
        kml.newpoint(name=index, coords=[(coordinate[1], coordinate[0])])
    kml.save(path + KML_EXT)


def create_kml(indices, coordinates, path):
    """
    indices is a list of strings with a name corresponding to each coordinate
    coordinates is  a list of tuples (latitude, longitude)
    Populates a kml file and saves it in path (str)
    """
    kml = skml.Kml()
    for index, coordinate in zip(indices, coordinates):
        kml.newpoint(name=index, coords=[(-coordinate[1], coordinate[0])])
    kml.save(path + '.kml')


#   TODO USE NUMPY ARRAYS INSTEAD OF LISTS FOR FASTER PROCESSING
def create_map(path, tile, mapContainer, focus=None):
    mapa = folium.Map(tiles=tile, location=focus, zoom_start=7)
    mapa.save(path)
    mapContainer.load(QtCore.QUrl.fromLocalFile(path))
    return mapa


def dms_to_deg(coordinate):
    """
    To deal with NBI's particularly egregious method of expressing coordinates
    coordinate is a tuple (latitude, longitude)
    returns a tuple of properly formatted coordinates
    """
    try:
        lat, long = coordinate
        lat, long = str(lat), str(long)
        if len(lat) != 1:
            lat = int(lat[0:2]) + int(lat[2:4]) / 60 + int(lat[4:lat.__len__()]) / 360000
        if len(long) != 1:
            long = int(long[0:3]) + int(long[3:5]) / 60 + int(long[5:long.__len__()]) / 360000
        return (lat, -long)
    except:
        return coordinate


def dms_to_deg(coordinate):
    """
    To deal with NBI's particularly egregious method of expressing coordinates
    coordinate is a tuple (latitude, longitude)
    returns a tuple of properly formatted coordinates
    """
    try:
        # lat = str(coordinate[0])
        # long = str(coordinate[1])
        lat, long = coordinate
        lat, long = str(lat), str(long)
        if lat != 0:
            lat = int(lat[0:2]) + int(lat[2:4]) / 60 + int(lat[4:lat.__len__()]) / 360000
        if long != 0:
            long = int(long[0:3]) + int(long[3:5]) / 60 + int(long[5:long.__len__()]) / 360000
        return (lat, long)
    except:
        return coordinate


def do_this(map1):
    data = pandas.read_csv(r'C:\Users\frano\PycharmProjects\Big\Maps\Volcanoes_USA.txt')
    latitudes = list(data["LAT"])
    longitudes = list(data["LON"])
    information = list(data["ELEV"])
    fgv = add_feature_group("Volcanoes", latitudes, longitudes, information)
    fgp = add_json_overlay("Population", r'C:\Users\frano\PycharmProjects\Big\Maps\antarctic_ice_edge.json')
    fgr = add_json_overlay("Random Route", r'C:\Users\frano\PycharmProjects\Big\Maps\el_paso_random_route.json')
    map1 = add_features(map1, (fgv, fgp, fgr))
    map1.add_child(folium.ClickForMarker())
    map1.save(BASE_MAP)
    return map1


def gather_coordinates(dir):
    paths = read_folder(dir, TXT_EXT)
    dataList = []
    indices = []
    coordinates = []
    for i, path in enumerate(paths):
        dataList.append(read_data(dir + '/' + path))
        dataList[i].fillna(0, inplace=True)
    for i, dF in enumerate(dataList):
        for index, row in dF.iterrows():
            coordinate = dms_to_deg((int(dF.loc[index, LATITUDE]), int(dF.loc[index, LONGITUDE])))
            if coordinate not in coordinates:  # if not (coordinate in coordinates): ??
                coordinates.append(coordinate)
                indices.append(index)
    return indices, coordinates


def gather_coordinates(dir):
    indices = []
    coordinates = []
    for path in lozoya.data_api.read_folder(dir, lozoya.data_api.TXT_EXT):
        data = lozoya.data_api.read_data(lozoya.data_api.os.path.join(dir, path))
        if not data.empty:
            data = lozoya.data_api.clean_data(data)
            for index, row in data.iterrows():
                coordinate = (
                    float(data.loc[index, lozoya.data_api.vars.LATITUDE]),
                    float(data.loc[index, lozoya.data_api.vars.LONGITUDE]))
                if not (index in indices):
                    coordinates.append(coordinate)
                    indices.append({column: row.values[x] for x, column in enumerate(data.columns.values)})
    return indices, coordinates


def gather_coordinates(dir):
    indices = []
    coordinates = []
    for path in lozoya.file.read_folder(dir, DataProcessor.TXT_EXT):
        data = lozoya.file.read_data(os.path.join(dir, path))
        if not data.empty:
            data = clean_data(data)
            for index, row in data.iterrows():
                coordinate = (float(data.loc[index, DataProcessor.vars.LATITUDE]),
                              float(data.loc[index, DataProcessor.vars.LONGITUDE]))
                if not (index in indices):
                    coordinates.append(coordinate)
                    indices.append({column: row.values[x] for x, column in enumerate(data.columns.values)})
    return indices, coordinates


def gather_coordinates(dir):
    paths = lozoya.file.read_folder(dir, TXT_EXT)
    dataList = []
    indices = []
    coordinates = []
    for i, path in enumerate(paths):
        dataList.append(read_data(dir + '/' + path))
        dataList[i].fillna(0, inplace=True)
    try:
        for i, dF in enumerate(dataList):
            for index, row in dF.iterrows():
                coordinate = dms_to_deg((int(dF.loc[index, LATITUDE]), int(dF.loc[index, LONGITUDE])))
                if coordinate not in coordinates:
                    coordinates.append(coordinate)
                    indices.append(index)
    except:
        pass
    return indices, coordinates


def gather_coordinates(dir):
    paths = read_folder(dir, TXT_EXT)
    dataList = []
    indices = []
    coordinates = []
    for i, path in enumerate(paths):
        dataList.append(read_data(os.path.join(dir, path)))
        dataList[i].fillna(0, inplace=True)
    try:
        for i, dF in enumerate(dataList):
            for index, row in dF.iterrows():
                # coordinate = dms_to_deg((int(dF.loc[index, LATITUDE]), int(dF.loc[index, LONGITUDE])))
                coordinate = (float(dF.loc[index, LATITUDE]), float(dF.loc[index, LONGITUDE]))
                if not (coordinate in coordinates):
                    coordinates.append(coordinate)
                    indices.append(index)
    except:
        pass
    return indices, coordinates


def gather_coordinates(dir):
    paths = read_folder(dir, TXT_EXT)
    dataList = []
    indices = []
    coordinates = []
    for i, path in enumerate(paths):
        print(path)
        data = read_data(os.path.join(dir, path), columns=(LATITUDE, LONGITUDE))
        if not data.empty:
            data.fillna(0, inplace=True)
            dataList.append(data)
    for i, dF in enumerate(dataList):
        try:
            print(i)
            for index, row in dF.iterrows():
                coordinate = dms_to_deg((int(dF.loc[index, LATITUDE]), int(dF.loc[index, LONGITUDE])))
                coordinate = (float(dF.loc[index, LATITUDE]), float(dF.loc[index, LONGITUDE]))
                print(coordinate)
                if not (coordinate in coordinates):
                    coordinates.append(coordinate)
                    indices.append({column: row.values[x] for x, column in enumerate(dF.columns.values)})
        except Exception as e:
            print(e)
    return indices, coordinates


def get_popups(info):
    popups = []
    for index in info:
        popup = ""
        for i in index:
            popup += (str(i) + ": " + str(index[i]) + "\n")
        popups.append(popup)
    return popups


def initiate_map():
    pass


def interpret_query(centroid, indices, coordinates, radius, units):
    memo = {}
    matchIndex = []
    matchCoord = []
    for index, coordinatePair in zip(indices, coordinates):
        if coordinatePair not in memo:
            distance = spherical_distance(np.radians(centroid), np.radians(coordinatePair), units)
            memo[coordinatePair] = distance
        else:
            distance = memo[coordinatePair]
        if distance <= radius:
            matchIndex.append(index)
            matchCoord.append(coordinatePair)
    return matchIndex, matchCoord


def insert_query_results_into_kml_file(self):
    pass


def load_map(mapa, path, mapContainer):
    mapa.save(path)
    mapContainer.load(QtCore.QUrl.fromLocalFile(path))
    return mapa


def plot_points(info, coords, popups, mapa):
    for index, coordinate, popup in zip(info, coords, popups):
        folium.Marker(coordinate, popup=popup).add_to(mapa)
    return mapa


def read_kml(self):
    file_name = 'file.xml'
    folder = os.path.join(os.path.expanduser("~"), "Desktop")
    full_file = os.path.abspath(os.path.join('folder', file_name))
    dom = ElementTree.parse(full_file)
    placemarks = dom.findall('placemark')
    for placemark in placemarks:
        print(placemarks.text)


def spherical_distance(centroid, point, units):
    """
    centroid and point are each a tuple of coordinates
    units is a string 'km','m','mi','ft'
    returns distance between two points on a sphere
    phi = 90 - latitude, theta = longitude
    For spherical coordinates (1, t1, p1) and (1, t2, p2)
    arc = arccos(sin(p1)*sin(p2)*cos(t1-t2) + cos(p1)*cos(p2))
    distance = arc*radius
    """
    p1 = 90 - centroid[0]
    p2 = 90 - point[0]
    t1 = centroid[1]
    t2 = point[1]
    arc = np.arccos(np.sin(p1) * np.sin(p2) * np.cos(t1 - t2) + np.cos(p1) * np.cos(p2))
    return arc * EARTH_RADIUS[units]
