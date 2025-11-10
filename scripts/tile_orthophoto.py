from typing import Any, Union
import argparse
import logging
import os

import numpy as np
import geopandas as gpd
import pathlib
import easyidp as idp

import tile_ortho


def main(
    orthophoto_path: Union[os.PathLike, str], 
    shapefile_path: Union[os.PathLike, str], 
    max_tile_size_px: int,
    buffer_m: float,
    outdir: Union[os.PathLike, str],
):
    
    # Parameters
    # ==========
    # Interpret possible path strings as paths
    outdir = pathlib.Path(outdir)
    orthophoto_path = pathlib.Path(orthophoto_path)
    shapefile_path = pathlib.Path(shapefile_path)
    # Numerical parameters
    max_outer_tile_width_px = max_tile_size_px
    max_outer_tile_height_px = max_tile_size_px

    # Loading data
    # ============
    orthophoto = idp.geotiff.GeoTiff(str(orthophoto_path))
    bounds = tile_ortho.geotiff_bounds(orthophoto)
    areas = gpd.read_file(
        shapefile_path, 
        bbox=tuple(bounds)
    ).clip(bounds)

    # Convert all to the same coordinate reference system (CRS)
    # =========================================================
    logging.info(f"CRS of {orthophoto_path}: {orthophoto.crs}")
    logging.info(f"CRS of {shapefile_path}: {areas.crs}")
    crs: Any = orthophoto.crs
    areas = areas.to_crs(crs)
    logging.info(
        f"Transformed bounding boxes and split areas to use the same CRS as "
        f"the orthophoto ({crs})."
    )

    # Split into overlapping tiles
    # ============================
    header: Any = orthophoto.header
    scale_x = header["scale"][0]
    scale_y = header["scale"][1]
    max_outer_tile_width_m = scale_x * max_outer_tile_width_px
    assert max_outer_tile_width_m > 2 * buffer_m, (
        "Maximum tile size smaller than twice the buffer size."
    )
    max_inner_tile_width_m = (
        max_outer_tile_width_m - 2 * buffer_m
    )
    max_outer_tile_height_m = scale_y * max_outer_tile_height_px
    assert max_outer_tile_height_m > 2 * buffer_m, (
        "Maximum tile size smaller than twice the buffer size."
    )
    max_inner_tile_height_m = (
        max_outer_tile_height_m - 2 * buffer_m
    )
    tiles = gpd.GeoDataFrame(
        tile_ortho.get_tiles(
            areas=areas,  
            max_tile_width_m=max_inner_tile_width_m, 
            max_tile_height_m=max_inner_tile_height_m,
        ), 
        columns=["tile_id", *areas.columns[:-1], "inner_tile"], 
        geometry="inner_tile",
        crs=crs,
    )
    tiles["outer_tile"] = tiles["inner_tile"].buffer(
        distance = buffer_m,
        join_style = 2,   
    ).clip(bounds)

    # Create outdir
    # =============
    tile_ortho.make_or_replace_dir(outdir)
    outdir_shapefiles = outdir / "tiles"
    outdir_shapefiles.mkdir()
    for split in tiles["split"].unique():
        (outdir / "images" / split).mkdir(parents=True)
    
    # Save cropped images
    # ===================
    for _, row in tiles.iterrows():
        tile_id = row["tile_id"]
        tile = row["outer_tile"]
        tile_coords = np.array(tile.boundary.coords)
        split = row["split"]
        save_path = outdir/"images"/split/f"tile_{tile_id}.tif"
        orthophoto.crop_polygon(
            polygon_hv=tile_coords,
            is_geo=True,
            save_path=save_path, 
        )
        tile_ortho.replace_geotiff_alpha(save_path, save_path)
    
    # Save tiles as a shapefile (so the inner tiles can be accessed later)
    # =========================
    tiles_inner = tiles.drop(columns=["outer_tile"])
    tiles_outer = tiles.set_geometry("outer_tile").drop(columns=["inner_tile"])
    tiles_inner.to_file(outdir_shapefiles/"tiles_inner.shp")
    tiles_outer.to_file(outdir_shapefiles/"tiles_outer.shp")


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description=(
        "Split an orthophoto to overlapping tiles."
    ))
    parser.add_argument("--ortho", help="input orthophoto", required=True)
    parser.add_argument("--shapefile", help="input shapefile, required=True")
    parser.add_argument("--outdir", help="output directory", required=True)
    parser.add_argument("--max-tile-size", help="max tile side length in pixels", required=True)
    parser.add_argument("--buffer-meters", help="amount of overlap in metres", required=True)

    args = parser.parse_args()

    # Check paths
    orthophoto_path = pathlib.Path(args.ortho)
    shapefile_path = pathlib.Path(args.shapefile)
    outdir = pathlib.Path(args.outdir)

    # Call main function to execute script
    main(
        orthophoto_path=orthophoto_path, 
        shapefile_path=shapefile_path, 
        max_tile_size_px=int(args.max_tile_size),
        buffer_m=float(args.buffer_meters),
        outdir=outdir, 
    )