from typing import Any, Union
import argparse
import logging
import yaml

import numpy as np
import geopandas as gpd
import pathlib
import easyidp as idp
import os

import tile_ortho


def main(
    orthophoto_path: Union[os.PathLike, str], 
    split_shapefile_path: Union[os.PathLike, str], 
    bboxes_ref_path: Union[os.PathLike, str],
    outdir: Union[os.PathLike, str],
    max_outer_tile_width_px: int = 896,
    max_outer_tile_height_px: int = 896,
    buffer_m: float = 5.0,
):
    # Parameters
    # ==========
    # Interpret possible path strings as paths
    outdir = pathlib.Path(outdir)
    orthophoto_path = pathlib.Path(orthophoto_path)
    split_shapefile_path = pathlib.Path(split_shapefile_path)
    bboxes_ref_path = pathlib.Path(bboxes_ref_path)


    # Loading data
    # ============
    orthophoto = idp.geotiff.GeoTiff(str(orthophoto_path))
    bounds = tile_ortho.geotiff_bounds(orthophoto)
    bboxes_ref = gpd.read_file(bboxes_ref_path, bbox=tuple(bounds))
    bboxes_ref["point"] = bboxes_ref.centroid
    split_areas = gpd.read_file(
        split_shapefile_path, 
        bbox=tuple(bounds)
    ).clip(bounds)

    # Convert all to the same coordinate reference system (CRS)
    # =========================================================
    logging.info(f"CRS of {orthophoto_path}: {orthophoto.crs}")
    logging.info(f"CRS of {bboxes_ref_path}: {bboxes_ref.crs}")
    logging.info(f"CRS of {split_shapefile_path}: {split_areas.crs}")
    crs: Any = orthophoto.crs
    bboxes_ref = bboxes_ref.to_crs(crs)
    split_areas = split_areas.to_crs(crs)
    logging.info(
        f"Transformed bounding boxes and split areas to use the same CRS as "
        f"the orthophoto ({crs})."
    )

    # Split into overlapping tiles
    # ============================
    header: Any = orthophoto.header
    scale_x: float = header["scale"][0]
    scale_y: float = header["scale"][1]
    max_inner_tile_width_m = (
        scale_x * max_outer_tile_width_px - 2 * buffer_m
    )
    max_inner_tile_height_m = (
        scale_y * max_outer_tile_height_px - 2 * buffer_m
    )
    tiles_gdf = gpd.GeoDataFrame(
        tile_ortho.get_tiles(
            areas=split_areas,  
            max_tile_width_m=max_inner_tile_width_m, 
            max_tile_height_m=max_inner_tile_height_m,
        ), 
        columns=["tile_id", *split_areas.columns[:-1], "tile_inner"], 
        geometry="tile_inner",
        crs=crs,
    )
    tiles_gdf["tile_outer"] = tiles_gdf["tile_inner"].buffer(
        distance = buffer_m,
        join_style = 2,   
    ).clip(bounds)

    # Create outdir
    # =============
    tile_ortho.make_or_replace_dir(outdir)
    for split in tiles_gdf["split"].unique():
        (outdir / f"images/{split}").mkdir(parents=True)
        (outdir / f"labels/{split}").mkdir(parents=True)

    # Save data description YAML
    # ==========================
    data_yaml = {}
    for split in tiles_gdf["split"].unique():
        data_yaml[split] = f"images/{split}"
    data_yaml["nc"] = 4
    data_yaml["names"] = ["healthy", "infected", "dead", "non-spruce"]
    with open(outdir/"data.yaml", "w") as file:
        yaml.safe_dump(data_yaml, file)

    # Save dataset in YOLO format
    # ===========================
    for _, row in tiles_gdf.iterrows():
        tile_id = row["tile_id"]
        tile = row["tile_outer"]
        tile_coords = np.array(tile.boundary.coords)
        split = row["split"]
        save_path = outdir/"images"/split/f"tile_{tile_id}.tif"
        orthophoto.crop_polygon(
            polygon_hv=tile_coords,
            is_geo=True,
            save_path=save_path, 
        )
        tile_ortho.replace_geotiff_alpha(save_path, save_path)
        for _, tree in bboxes_ref.clip(tile).iterrows():
            bbox = tree["geometry"].bounds
            class_id = tree["class"] 
            yolo_format = tile_ortho.convert_to_yolo_format(
                bbox, tile.bounds, class_id)
            with open(outdir/"labels"/split/f"tile_{tile_id}.txt", 'a') as f:
                f.write(yolo_format + "\n")


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description=(
        "Creates a YOLO dataset for tree detection."
    ))
    parser.add_argument("--ortho", help="input orthophoto", required=True)
    parser.add_argument("--split-areas", help="split areas shapefile", required=True)
    parser.add_argument("--ref-trees", help="trees shapefile", required=True)
    parser.add_argument("--outdir", help="output directory", required=True)
    parser.add_argument("--max-tile-size", help="max tile side length in pixels", required=True)
    parser.add_argument("--buffer-meters", help="amount of overlap in metres", required=True)
    args = parser.parse_args()

    # Convert arguments to proper types
    orthophoto_path = pathlib.Path(args.ortho)
    split_shapefile_path = pathlib.Path(args.split_areas)
    trees_shapefile_path = pathlib.Path(args.ref_trees)
    outdir = pathlib.Path(args.outdir)
    max_tile_size_px = int(args.max_tile_size)
    buffer_m = float(args.buffer_meters)

    main(
        orthophoto_path=orthophoto_path, 
        split_shapefile_path=split_shapefile_path, 
        bboxes_ref_path=trees_shapefile_path, 
        outdir=outdir, 
        max_outer_tile_width_px=max_tile_size_px,
        max_outer_tile_height_px=max_tile_size_px,
        buffer_m=buffer_m,
    )