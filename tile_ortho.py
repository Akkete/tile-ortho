from typing import Any, Iterable, List, Tuple, Union
import math
import itertools
import pathlib
import shutil
import os

import numpy as np
from numpy.typing import NDArray
import geopandas as gpd
import easyidp as idp
import rasterio
import shapely


def geotiff_bounds(
        geotiff: idp.geotiff.GeoTiff
) -> Tuple[float, float, float, float]:
    """Determine the bounds of a an `easyidp.Geotiff` object.
    
    Returns a tuple `(minx, miny, maxx, maxy)` of geographic coordinates.
    """
    header: Any = geotiff.header
    height: int = header["height"]
    width: int = header["width"]
    array: NDArray = geotiff.pixel2geo(
        np.array([[0, height - 1], [width - 1, 0]]))
    return tuple(array.flatten().tolist())


def replace_geotiff_alpha(
        input_path: Union[os.PathLike, str], 
        output_path: Union[os.PathLike, str]
) -> None:
    """Open a geotiff, replace transparent pixels with white, save result."""
    
    with rasterio.open(input_path) as src:
        profile = src.profile
        data = src.read()  # shape: (bands, height, width)

        if src.count == 4:
            # Assume last band is alpha
            r, g, b, a = data
            alpha_mask = a == 0

            # Replace transparent pixels with white
            r[alpha_mask] = 255
            g[alpha_mask] = 255
            b[alpha_mask] = 255

            rgb_data = np.stack([r, g, b])
            profile.update(count=3)
        elif src.count == 3:
            # No alpha channel
            rgb_data = data
        else:
            raise ValueError(f"Unsupported band count: {src.count}")

        profile.update(dtype=rgb_data.dtype, count=3)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(rgb_data)


def get_tiles(
    areas: gpd.GeoDataFrame, 
    max_tile_width_m: float, 
    max_tile_height_m: float,
) -> Iterable[tuple[Any, ...]]:
    for _, row in areas.iterrows():
        min_x, min_y, max_x, max_y = row.geometry.bounds
        width = max_x - min_x
        height = max_y - min_y
        n_cols = math.ceil(width / max_tile_width_m)
        n_rows = math.ceil(height / max_tile_height_m)
        tile_width_m = width / n_cols
        tile_height_m = height / n_rows
        fields = row.iloc[:-1].to_list()
        for (i, j) in itertools.product(range(n_cols), range(n_rows)):
            tile = shapely.geometry.box(
                min_x + i * tile_width_m, 
                min_y + j * tile_height_m,
                min_x + (i + 1) * tile_width_m,
                min_y + (j + 1) * tile_height_m,
            ).intersection(row.geometry)
            min_x_rounded = int(min_x + i * tile_width_m)
            min_y_rounded = int(min_y + j * tile_height_m)
            tile_id = f"{min_x_rounded}_{min_y_rounded}"
            yield tile_id, *fields, tile 


def make_or_replace_dir(dir_path: pathlib.Path):
    """Create a directory. If it already exists, delete the old one."""
    assert not dir_path.is_file()
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)


def ovals_from_yolo_output(
    labels_txt_path: pathlib.Path, 
    bounds: Tuple[float, float, float, float],
) -> Tuple[List[int], List[shapely.geometry.Polygon]]:
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    classes = []
    polygons = []
    with open(labels_txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = bounds[0] + float(parts[1]) * width
            y_center = bounds[3] - float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height
            center_point = shapely.geometry.Point(x_center, y_center)
            unit_circle = center_point.buffer(1, resolution=4)
            oval = shapely.affinity.scale(
                unit_circle, box_width/2, box_height/2)
            polygons.append(oval)
            classes.append(class_id)
    return classes, polygons


def bboxes_from_yolo_output(
    labels_txt_path: pathlib.Path, 
    bounds: Tuple[float, float, float, float],
) -> Tuple[List[int], List[shapely.geometry.Polygon]]:
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    classes = []
    polygons = []
    with open(labels_txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = bounds[0] + float(parts[1]) * width
            y_center = bounds[3] - float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height
            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2
            polygons.append(shapely.geometry.Polygon(
                [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
            ))
            classes.append(class_id)
    return classes, polygons


def convert_to_yolo_format(
        object_bbox: tuple, 
        img_bbox: tuple, 
        class_id: tuple,
    ) -> str:
    """
    Converts an axis aligned bounding box to YOLO format.

    Input:
    - object_bbox: Object bounding box (min_x, min_y, max_x, max_y).
    - img_bbox: Image bounding box (min_x, min_y, max_x, max_y).
    - class_id: Object class ID.

    Output:
    - String in YOLO format (class_id, x_center, y_center, width, height).
    """
    obj_min_x, obj_min_y, obj_max_x, obj_max_y = object_bbox
    img_min_x, img_min_y, img_max_x, img_max_y = img_bbox
    img_width = img_max_x - img_min_x
    img_height = img_max_y - img_min_y
    obj_min_x_yolo = (obj_min_x - img_min_x) / img_width
    obj_max_x_yolo = (obj_max_x - img_min_x) / img_width
    obj_min_y_yolo = (img_max_y - obj_max_y) / img_height
    obj_max_y_yolo = (img_max_y - obj_min_y) / img_height
    x_center = (obj_min_x_yolo + obj_max_x_yolo) / 2.0
    y_center = (obj_min_y_yolo + obj_max_y_yolo) / 2.0
    width = obj_max_x_yolo - obj_min_x_yolo
    height = obj_max_y_yolo - obj_min_y_yolo

    return f"{class_id} {x_center} {y_center} {width} {height}"