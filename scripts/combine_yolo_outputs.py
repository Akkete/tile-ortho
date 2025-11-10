from typing import List, Literal
import argparse
import pathlib

import easyidp as idp
import pandas as pd
import geopandas as gpd

import tile_ortho


def main(
    yolo_labels_dir: pathlib.Path,
    tile_images_dir: pathlib.Path,
    tiles_shapefile: pathlib.Path,
    outfile: pathlib.Path,
    shape: Literal["rectangle", "oval"] = "rectangle"
):

    tiles_gdf = gpd.read_file(tiles_shapefile)

    polygons_gdf_parts: List[gpd.GeoDataFrame] = []
    for _, tile in tiles_gdf.iterrows():
        tile_id = tile["tile_id"]
        geotiff_path = tile_images_dir/f"tile_{tile_id}.tif"
        labels_path = yolo_labels_dir/f"tile_{tile_id}.txt"
        if not (geotiff_path.exists() and labels_path.exists()):
            continue
        tile_inner = tile["geometry"]
        orthophoto = idp.geotiff.GeoTiff(str(geotiff_path))
        crs = orthophoto.crs
        bounds = tile_ortho.geotiff_bounds(orthophoto)
        if shape == "oval":
            classes, polygons = (
                tile_ortho.ovals_from_yolo_output(labels_path, bounds))
        elif shape == "rectangle":
            classes, polygons = (
                tile_ortho.bboxes_from_yolo_output(labels_path, bounds))
        else:
            raise ValueError(
                f"Shape should be rectangle or oval. Got {shape} instead.")
        polygons_gdf_part = gpd.GeoDataFrame(
            data={"class": classes}, 
            geometry=polygons, 
            crs=crs
        )
        centroids = polygons_gdf_part.centroid.clip(tile_inner)
        polygons_gdf_part_clipped = polygons_gdf_part.loc[centroids.index, :]
        polygons_gdf_parts.append(polygons_gdf_part_clipped)

    polygons_gdf = gpd.GeoDataFrame(pd.concat(
        polygons_gdf_parts, 
        ignore_index=True
    ))

    polygons_gdf.to_file(outfile, mode="w")


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description=(
        "Create detected trees shapefile from YOLO outputs."
    ))
    parser.add_argument("--yolo-labels-dir", 
                        help="YOLO output labels", required=True)
    parser.add_argument("--tile-images-dir", 
                        help="original images folder", required=True)
    parser.add_argument("--tiles-shapefile", 
                        help="inner tiles shapefile", required=True)
    parser.add_argument("--outfile", 
                        help="output shapefile", required=True)
    parser.add_argument("--shape",
                        help="rectangle or oval", 
                        choices=["rectangle", "oval"],
                        default="rectangle")
    args = parser.parse_args()
    
    # Check paths
    yolo_labels_dir=pathlib.Path(args.yolo_labels_dir)
    tile_images_dir=pathlib.Path(args.tile_images_dir)
    tiles_shapefile=pathlib.Path(args.tiles_shapefile)
    outfile=pathlib.Path(args.outfile)

    # Call main function
    main(
        yolo_labels_dir=yolo_labels_dir,
        tile_images_dir=tile_images_dir,
        tiles_shapefile=tiles_shapefile,
        outfile=outfile,
        shape=args.shape,
    )