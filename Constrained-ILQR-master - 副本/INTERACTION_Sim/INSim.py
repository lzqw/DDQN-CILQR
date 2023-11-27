import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt

import xml.etree.ElementTree as xml
import pyproj
import math
import numpy as np
import INTERACTION_Sim.dataset_reader
import sys
import INTERACTION_Sim.tracks_vis
import time


def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center


def polygon_xy_from_motionstate(ms, width, length):
    lowleft = (ms.x - length / 2., ms.y - width / 2.)
    lowright = (ms.x + length / 2., ms.y - width / 2.)
    upright = (ms.x + length / 2., ms.y + width / 2.)
    upleft = (ms.x - length / 2., ms.y + width / 2.)
    return rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([ms.x, ms.y]), yaw=ms.psi_rad)


def get_item_iterator(d):
    assert isinstance(d, dict)
    if sys.version_info[0] == 2:
        item_iter = d.iteritems()
        assert hasattr(item_iter, "next")
    elif sys.version_info[0] == 3:
        item_iter = iter(d.items())
        assert hasattr(item_iter, "__next__")
    else:
        # should not happen
        raise RuntimeError("Only python 2 and 3 supported.")
    assert hasattr(item_iter, "__iter__")
    return item_iter


class Point:
    def __init__(self):
        self.x = None
        self.y = None


class LL2XYProjector:
    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin + 180.) / 6) + 1  # works for most tiles, and for all in the dataset
        self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]


def get_type(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "type":
            return tag.get("v")
    return None


def get_subtype(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "subtype":
            return tag.get("v")
    return None


def get_x_y_lists(element, point_dict):
    x_list = list()
    y_list = list()
    for nd in element.findall("nd"):
        pt_id = int(nd.get("ref"))
        point = point_dict[pt_id]
        x_list.append(point.x)
        y_list.append(point.y)
    return x_list, y_list


def set_visible_area(point_dict, axes):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9

    for id, point in get_item_iterator(point_dict):
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)

    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([min_x - 10, max_x + 10])
    axes.set_ylim([min_y - 10, max_y + 10])
    return  min_x,min_y,max_x,max_y

def mapsize(filename, lat_origin, lon_origin):
    # filename:D:\Project\RL-AD-Intersection\interaction-dataset-master\maps\TC_BGR_Intersection_VA.osm

    projector = LL2XYProjector(lat_origin, lon_origin)

    e = xml.parse(filename).getroot()
    point_dict = dict()
    for node in e.findall("node"):
        point = Point()
        point.x, point.y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
        point_dict[int(node.get('id'))] = point
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9
    for id, point in get_item_iterator(point_dict):
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)
    return  min_x,min_y,max_x,max_y

def draw_map_without_lanelet(filename, axes, lat_origin, lon_origin):
    # filename:D:\Project\RL-AD-Intersection\interaction-dataset-master\maps\TC_BGR_Intersection_VA.osm

    assert isinstance(axes, matplotlib.axes.Axes)

    axes.set_aspect('equal', adjustable='box')
    axes.patch.set_facecolor('lightgrey')

    projector = LL2XYProjector(lat_origin, lon_origin)

    e = xml.parse(filename).getroot()
    # <Element 'osm' at 0x00000224D4052F90>

    point_dict = dict()
    for node in e.findall("node"):
        point = Point()
        point.x, point.y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
        point_dict[int(node.get('id'))] = point
    min_x,min_y,max_x,max_y=set_visible_area(point_dict, axes)
    unknown_linestring_types = list()
    for way in e.findall('way'):
        way_type = get_type(way)
        if way_type is None:
            raise RuntimeError("Linestring type must be specified")
        elif way_type == "curbstone":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif way_type == "line_thin":
            way_subtype = get_subtype(way)
            if way_subtype == "dashed":
                type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="white", linewidth=1, zorder=10)
        elif way_type == "line_thick":
            way_subtype = get_subtype(way)
            if way_subtype == "dashed":
                type_dict = dict(color="white", linewidth=2, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="white", linewidth=2, zorder=10)
        elif way_type == "pedestrian_marking":
            type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
        elif way_type == "bike_marking":
            type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
        elif way_type == "stop_line":
            type_dict = dict(color="white", linewidth=3, zorder=10)
        elif way_type == "virtual":
            type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[2, 5])
        elif way_type == "road_border":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif way_type == "guard_rail":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif way_type == "traffic_sign":
            continue
        else:
            if way_type not in unknown_linestring_types:
                unknown_linestring_types.append(way_type)
            continue

        x_list, y_list = get_x_y_lists(way, point_dict)
        plt.plot(x_list, y_list, **type_dict)

