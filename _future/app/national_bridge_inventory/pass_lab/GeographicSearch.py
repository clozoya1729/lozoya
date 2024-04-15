"""
Author: Christian Lozoya, 2017
"""

from Utilities.DataHandler import *


#   TODO USE NUMPY ARRAYS INSTEAD OF LISTS FOR FASTER PROCESSING
def gather_coordinates(dir):
    paths = read_folder(dir, TXT_EXT)
    dataList = []
    indices = []
    coordinates = []

    for i, path in enumerate(paths):
        dataList.append(read_data(os.path.join(dir, path)))
        dataList[i].fillna(0, inplace=True)

    for i, dF in enumerate(dataList):
        for index, row in dF.iterrows():
            coordinate = dms_to_deg(
                (int(dF.loc[index, LATITUDE]), int(dF.loc[index, LONGITUDE]))
            )

            if not (coordinate in coordinates):
                coordinates.append(coordinate)
                indices.append(index)
    return indices, coordinates


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
        if distance < radius:
            matchIndex.append(index)
            matchCoord.append(coordinatePair)
    return matchIndex, matchCoord


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
    kml.save(path)


def dms_to_deg(coordinate):
    """
    To deal with NBI's particularly egregious method of expressing coordinates
    coordinate is a tuple (latitude, longitude)
    returns a tuple of properly formatted coordinates
    """
    try:
        lat, long = coordinate
        lat, long = str(lat), str(long)
        if len(lat) != 1: lat = int(lat[0:2]) + int(lat[2:4]) / 60 + int(lat[4:lat.__len__()]) / 360000
        if len(long) != 1: long = int(long[0:3]) + int(long[3:5]) / 60 + int(long[5:long.__len__()]) / 360000
        return (lat, -long)
    except:
        return coordinate


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
