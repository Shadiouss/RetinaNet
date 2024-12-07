# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 00:21:57 2024

@author: QuadroRTX
"""

import os
import glob
import shutil
import geopandas as gpd
from pathlib import Path
from data import annotations_split, annotations_merge  # Assuming these are implemented elsewhere
from data import preprocess_data


def get_image_name(file_path):
    """
    Extracts the file name (without extension) from a given file path.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: File name without the extension.
    """
    return os.path.splitext(os.path.basename(file_path))[0]  # Returns the file name without extension

def preprocess(annotations, image_path, directory_to_save_crops, patch_size=400, patch_overlap=0.0,
               merge_name=None, split=0.3, seed=None, label=None, max_empty=None):
    """
    Prepares training data by splitting large images into smaller patches and creating
    relevant annotation files from shapefiles.

    Args:
        annotations (str or list): Path(s) to the shapefile(s) containing annotations.
        image_path (str or list): Path(s) to the image(s) corresponding to the annotations.
        directory_to_save_crops (str): Directory where the cropped images will be saved.
        patch_size (int): Size of the image patches (default=400).
        patch_overlap (float): Overlap between patches as a fraction of the patch size (default=0.0).
        merge_name (str): Name of the merged annotation file (default=None).
        split (float): Fraction of data to use for testing (default=0.3).
        seed (int): Random seed for splitting annotations (default=None).
        label (dict or str): Label(s) to assign to annotations (default=None).
        max_empty (int): Maximum number of empty patches allowed (default=None).
    """
    if label is not None:  # If label is provided, update the shapefiles
        for shp in annotations:
            # Load the shapefile using geopandas
            gdf = gpd.read_file(shp)
            print(f"Columns in {shp}: {gdf.columns}")

            # Remove extra spaces from column names if needed
            gdf.columns = gdf.columns.str.strip()

            # Check if 'xmin', 'ymin', 'xmax', 'ymax' exist
            required_columns = ['xmin', 'ymin', 'xmax', 'ymax']
            missing_columns = [col for col in required_columns if col not in gdf.columns]

            # If the bounding box columns are missing, generate them from geometry
            if missing_columns:
                if 'geometry' in gdf.columns:
                    # Generate xmin, ymin, xmax, ymax from geometry bounds
                    gdf['xmin'] = gdf.geometry.bounds.minx
                    gdf['ymin'] = gdf.geometry.bounds.miny
                    gdf['xmax'] = gdf.geometry.bounds.maxx
                    gdf['ymax'] = gdf.geometry.bounds.maxy
                else:
                    raise KeyError(f"Missing geometry column in {shp}. Cannot generate bounding boxes.")

            # Add the 'label' column if it doesn't exist
            if 'label' not in gdf.columns:
                if isinstance(label, dict):
                    image_name = get_image_name(shp)  # Extract image name from shapefile
                    if image_name in label:
                        gdf['label'] = label[image_name]  # Assign label from dictionary
                    else:
                        raise ValueError(f"No label found for {image_name} in the label dictionary.")
                else:
                    gdf['label'] = label  # Single value assignment

            # Save the updated data to CSV (assuming that's what you want to do)
            csv_file_path = shp.replace('.shp', '_updated.csv')
            gdf.to_csv(csv_file_path, index=False)
            print(f"Saved updated annotations to {csv_file_path}")

    # Proceed with other preprocessing steps...
    if not os.path.isdir(directory_to_save_crops):
        os.makedirs(directory_to_save_crops)
        print(f"Created directory: {directory_to_save_crops}")

    # Process images and annotations
    if isinstance(image_path, list):  # Handle list of images
        assert isinstance(annotations, list), "If 'image_path' is a list, 'annotations' must also be a list"
        for img, shp in zip(image_path, annotations):
            print(f"Processing image: {img}, with annotations: {shp}")
            preprocess_data(directory_to_save_crops, img, shp, patch_size, patch_overlap, max_empty)
    else:  # Handle single image
        if image_path.endswith('.tif'):
            preprocess_data(directory_to_save_crops, image_path, annotations, patch_size, patch_overlap, max_empty)
        else:
            owd = os.getcwd()
            os.chdir(image_path)
            tif_files = glob.glob('*.tif')
            os.chdir(owd)

            if annotations.endswith('.shp'):
                for tif in tif_files:
                    print(f"Processing image: {tif} with annotations: {annotations}")
                    preprocess_data(directory_to_save_crops, image_path + "\\" + tif, annotations, patch_size, patch_overlap,max_empty)
            else:
                os.chdir(annotations)
                shp_files = glob.glob('*.shp')
                assert len(tif_files) == len(shp_files), "Number of .shp-files does not match .tif-files"
                for tif, shp in zip(tif_files, shp_files):
                    print(f"Processing image: {tif} with annotations: {shp}")
                    preprocess_data(directory_to_save_crops, image_path + "\\" + tif, shp, patch_size, patch_overlap, max_empty)

    # Merging annotations
    if merge_name is None:
        merge_name = "csv_ref_merged.csv"
    print(f"Merging annotations into: {merge_name}")
    annotations_merge(directory_to_save_crops, merge_name)

    # Split annotations into train/test
    if split > 0:
        print(f"Splitting annotations into train/test with split ratio: {split}")
        annotations_split(directory_to_save_crops, merge_name, split, seed=seed)

