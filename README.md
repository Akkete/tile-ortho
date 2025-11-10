
# Tile orthophotos with this script

## Usage

The following commands create a [Conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment and run all the scripts with test data. 

```shell
conda env update -n tile-ortho -f environment.yml

conda activate tile-ortho

python tile_ortho.py \
    --ortho test_data/input/test_ortho.tif \
    --shapefile test_data/input/test_shapefile/test_shapefile.shp \
    --max-tile-size 512 \
    --buffer-meters 5.0 \
    --outdir test_data/output/tile_ortho

yolo predict \
    save_txt \
    model=test_data/input/yolo11m-spruce.pt \
    source=test_data/output/tile_ortho/images/test \
    imgsz=512 \
    project=test_data/output/yolo_predict \
    name=results

mkdir test_data/output/combine_yolo_outputs

python combine_yolo_outputs.py \
    --yolo-labels-dir test_data/output/yolo_predict/results/labels \
    --tile-images-dir test_data/output/tile_ortho/images/test \
    --tiles-shapefile test_data/output/tile_ortho/tiles/tiles_inner.shp \
    --outfile test_data/output/combine_yolo_outputs/detections.shp
```

Alternatively, all of the above can be run with [Snakemake](https://snakemake.readthedocs.io/en/stable/).

```shell
snakemake -c1 -p combine_yolo_outputs 
```

## Scripts

### Tile orthophoto

- Input: GeoTIFF orthophoto, Shapefile of areas of interest 
- Parameters: Maximum tile size in pixels, Buffer in metres
- Output: GeoTIFF tile images, Shapefile of inner tiles (no buffer), Shapefile of outer tiles (with buffer)
- Note: In my current use case I give the maximum tile size in pixels and the buffer in metres,  but the units could easily be whatever.
- Note: This script does not produce uniformly sized tiles.
- Procedure in detail:
    1. The orthophoto is divided into areas of interest, which are defined in an input Shapefile. These areas of interest can be used to define train-validation-test split, or they can be test plots, or they can simply be used to crop the orthophoto into a more manageable area.
    2. Each area of interest is split into non-overlapping ‘inner tiles’. First it is calculated how many tiles can fit horizontally and vertically in the area of interest given the maximum size parameter. Then, the area of interest is split into equally large tiles. Note that this means the tiles may not be square. Furthermore, while the tiles in one area of interest are the same size, the tiles in different areas of interest may end up with different sizes.
    3. Every ‘inner tile’ is expanded with a buffer in all directions, creating an ‘outer tile’. This means the outer tiles overlap by twice the buffer.
    4. The inner and outer tiles are saved in Shapefiles.

### Combine YOLO outputs

- Input: Plain text YOLO outputs, GeoTIFF tile images, Shapefile of inner tiles
- Parameters: detection shape (rectangle or oval)
- Output: Shapefile of combined detections
- Note: YOLO detections are rectangular, but this script can optionally convert them to ovals, which is a more natural shape for trees.
- Procedure in detail:
    1. For each tile of YOLO outputs, do the following:
       1. Convert the YOLO bounding boxes to georeferenced detection shapes.
       2. Calculate the centroid of each detection.
       3. Fetch the inner tile corresponding to this tile.
       4. Throw out all shapes which don’t have their centroid within the inner tile.
    2. Merge all the detection shapes obtained in the previous step.
    3. Save the detections in a Shapefile. 

### Prepare YOLO dataset

- Input: GeoTIFF orthophoto, Shapefile of areas of interest, Shapefile of objects to be detected
- Output: YOLO dataset of GeoTIFF images, plain text labels and YAML data description
- Parameters: Maximum tile size in pixels, Buffer in metres
- Note: Code and functionality heavily overlaps with Tile orthophoto. This also handles splitting the geometries of detectable objects on the tiles and converting them to the YOLO format. Doesn’t save anything else than the YOLO dataset proper, but could be modified to save e.g. the tiles.

## Requirements

Tested with Python version 3.10

Python packages required: Geopandas, EasyIDP, Shapely, Rasterio, NumPy, PyYAML.

Using Conda and Snakemake recommended.

## File overview and organisation

```
.
├── README.md
├── tile_ortho.py                <-- Useful functions used throughout the scripts
├── scripts                      <-- Sripts that can be executed from the command line 
│   ├── combine_yolo_outputs.py
│   ├── prepare_yolo_dataset.py
│   └── tile_orthophoto.py
├── test_data                    <-- Sample data for testing that everything works
│   ├── input
│   │   ├── areas_of_interest    <-- Shapefile of areas of interest
│   │   ├── ref_trees            <-- Shapefile of reference tree geometries
│   │   ├── test_ortho.tif       <-- Tiny ortho geotiff for testing
│   │   └── yolo11m-spruce.pt    <-- A YOLO model fine tuned to detect spruces from orthoimages 
│   └── output                   <-- Folder for test outputs
├── environment.yml              <-- Required packages
├── pyproject.toml               <-- For installing tile_ortho as a package
└── Snakefile                    <-- Convenient way to run the scripts
```

Note: There are two files with nearly identical names: 
- `tile_ortho.py` – library of utility functions
- `scripts/tile_orthophoto.py` – a script

Sorry for being confusing!

## TODO

- Prepare YOLO dataset and Tile orthophoto scripts overlap very heavily, with lots of straight up duplicate code. Refactor them in a sensible way.