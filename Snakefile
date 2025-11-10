import platform


project_name = "tile-ortho"
lockfile = f"environment.{platform.platform()}.lock.yml"


rule create_environment:
    input:
        env="environment.yml", 
        pyproject="pyproject.toml",
    output: lockfile
    shell:
        """
        conda env update -n {project_name} -f {input.env}
        conda env export -n {project_name} -f {output} --format=environment-yaml
        """


rule delete_environment:
    shell:
        """
        conda env remove -n {project_name}
        rm -f {lockfile}
        """


rule tile_ortho:
    input:
        script="scripts/tile_ortho.py",
        ortho="test_data/input/test_ortho.tif",
        shapefile="test_data/input/areas_of_interest/areas_of_interest.shp",
        environment=lockfile,
    output:
        parent_dir=directory("test_data/output/tile_ortho/"),
        test_dir=directory("test_data/output/tile_ortho/images/test/"),
    params:
        max_tile_size_px=512,
        buffer_m=5.0,
    shell:
        "python {input.script} "
        "--ortho {input.ortho} "
        "--shapefile {input.shapefile} "
        "--max-tile-size {params.max_tile_size_px} " 
        "--buffer-meters {params.buffer_m} "
        "--outdir {output.parent_dir}"


rule yolo_predict:
    input:
        data="test_data/output/tile_ortho/images/test/",
        environment=lockfile,
    output:
        directory("test_data/output/yolo_predict/")
    params:
        img_size=512,
        model="test_data/input/yolo11m-spruce.pt",
    shell:
        "yolo predict "
        "save_txt "
        "model={params.model} "
        "source={input.data} "
        "imgsz={params.img_size} "
        "project={output} "
        "name=results"


rule combine_yolo_outputs:
    input:
        script="scripts/combine_yolo_outputs.py", 
        yolo_outputs="test_data/output/yolo_predict/",
        tiles="test_data/output/tile_ortho/",
        environment=lockfile,
    output:
        shapefile="test_data/output/combine_yolo_outputs/detections.shp",
    shell:
        "python {input.script} "
        "--yolo-labels-dir {input.yolo_outputs}/results/labels "
        "--tile-images-dir {input.tiles}/images/test "
        "--tiles-shapefile {input.tiles}/tiles/tiles_inner.shp "
        "--outfile {output.shapefile} "
        "--shape oval"


rule prepare_yolo_dataset:
    input:
        script="scripts/prepare_yolo_dataset.py",
        ortho="test_data/input/test_ortho.tif",
        shapefile="test_data/input/areas_of_interest/areas_of_interest.shp",
        ref_trees="test_data/input/ref_trees/ref_trees.shp",
        environment=lockfile,
    output:
        directory("test_data/output/prepare_yolo_dataset/"),
    params:
        max_tile_size_px=512,
        buffer_m=5.0,
    shell:
        "python {input.script} "
        "--ortho {input.ortho} "
        "--split-areas {input.shapefile} "
        "--ref-trees {input.ref_trees} "
        "--max-tile-size {params.max_tile_size_px} "
        "--buffer-meters {params.buffer_m} "
        "--outdir {output}"


rule test_all:
    input:
        tile_ortho="test_data/output/tile_ortho/",
        yolo_predict="test_data/output/yolo_predict/",
        combine_yolo_outputs="test_data/output/combine_yolo_outputs/detections.shp",
        prepare_yolo_dataset="test_data/output/prepare_yolo_dataset/",