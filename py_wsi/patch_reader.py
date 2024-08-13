'''

These functions acess the .svs files via the openslide-python API and perform patch
sampling as is typically needed for deep learning.

Author: @ysbecca
'''

import numpy as np
import openslide
from openslide import open_slide  
from openslide.deepzoom import DeepZoomGenerator
from glob import glob
from xml.dom import minidom
from shapely.geometry import Polygon, Point
import logging
from skimage.color import rgb2gray, rgb2hed
from skimage.filters import threshold_otsu
import math
import tifffile as tiff
from skimage import exposure

from .store import *


def check_label_exists(label, label_map):
    ''' Checking if a label is a valid label. 
    '''
    if label in label_map:
        return True
    else:
        print("py_wsi error: provided label " + str(label) + " not present in label map.")
        print("Setting label as -1 for UNRECOGNISED LABEL.")
        print(label_map)
        return False

def generate_label(regions, region_labels, point, label_map):
    ''' Generates a label given an array of regions.
        - regions               array of vertices
        - region_labels         corresponding labels for the regions
        - point                 x, y tuple
        - label_map             the label dictionary mapping string labels to integer labels
    '''
    for i in range(len(region_labels)):
        poly = Polygon(regions[i])
        if poly.contains(Point(point[0], point[1])):
            if check_label_exists(region_labels[i], label_map):
                return label_map[region_labels[i]]
            else:
                return -1
    # By default, we set to "Normal" if it exists in the label map.
    if check_label_exists('Normal', label_map):
        return label_map['Normal']
    else:
        return -1

def get_regions(path):
    ''' Parses the xml at the given path, assuming annotation format importable by ImageScope. '''
    xml = minidom.parse(path)
    # The first region marked is always the tumour delineation
    regions_ = xml.getElementsByTagName("Region")
    regions, region_labels = [], []
    for region in regions_:
        vertices = region.getElementsByTagName("Vertex")
        attribute = region.getElementsByTagName("Attribute")
        if len(attribute) > 0:
            r_label = attribute[0].attributes['Value'].value
        else:
            r_label = region.getAttribute('Text')
        region_labels.append(r_label)

        # Store x, y coordinates into a 2D array in format [x1, y1], [x2, y2], ...
        coords = np.zeros((len(vertices), 2))

        for i, vertex in enumerate(vertices):
            coords[i][0] = vertex.attributes['X'].value
            coords[i][1] = vertex.attributes['Y'].value

        regions.append(coords)
    return regions, region_labels

def patch_to_tile_size(patch_size, overlap):
    return int(patch_size - overlap*2)

def sample_and_store_patches(file_name,
                             file_dir,
                             pixel_overlap,
                             env=False,
                             meta_env=False,
                             patch_size=512,
                             dz_level=None,
                             magnification=20,
                             ignore_bg_percent=None,
                             color_deconv=False,
                             xml_dir=False,
                             label_map={},
                             limit_bounds=False,
                             rows_per_txn=20,
                             db_location='',
                             prefix='',
                             storage_option='lmdb'):
    ''' Sample patches of specified size from .svs file.
        - file_name             name of whole slide image to sample from
        - file_dir              directory file is located in
        - pixel_overlap         pixels overlap on each side
        - env, meta_env         for LMDB only; environment variables
        - level                 0 is lowest resolution; level_count - 1 is highest
        - xml_dir               directory containing annotation XML files
        - label_map             dictionary mapping string labels to integers
        - rows_per_txn          how many patches to load into memory at once
        - storage_option        the patch storage option              

        Note: patch_size is the dimension of the sampled patches, NOT equivalent to openslide's definition
        of tile_size. This implementation was chosen to allow for more intuitive usage.
    '''

    filepath = os.path.join(file_dir, file_name)
    logging.info(f"-------------- | Tiling {filepath} | --------------")
    slide = open_slide(filepath)
    # Get appropriate level for the magnification
    if dz_level is None:
        # level = get_level_for_magnification(slide, magnification)
        # logging.info("Level was not specified, magnification of {} used to choose level {}.".format(magnification, level))
        desired_downsample = get_downsample_factor(slide, magnification)
        logging.info(f"This is desired_downsampling {desired_downsample}")
        # patch_size = int(np.ceil(patch_size / df))
        # if math.log2(patch_size) % 1 != 0:
        #     logging.error("Patch size is not a power of 2, will not tile.")
        #     KeyError()
        # pixel_overlap = int(np.ceil(pixel_overlap / df))
         # Get the approp level in DeepZoomGenerator world, using desired slide level
        # Calculate the corresponding Deep Zoom level
        dz_level = round(math.log2(desired_downsample))
        logging.info(f"This is desired dz level {dz_level}")

    tile_size = patch_to_tile_size(patch_size, pixel_overlap)
    logging.info(f"Tile size {tile_size} given patch size {patch_size} and pixel overlap {pixel_overlap}")

    tiles = DeepZoomGenerator(slide,
                              tile_size=tile_size,
                              overlap=pixel_overlap,
                              limit_bounds=limit_bounds)

    if xml_dir:
        # Expect filename of XML annotations to match SVS file name
        regions, region_labels = get_regions(xml_dir + file_name[:-4] + ".xml")

    if dz_level >= tiles.level_count:
        print("[py_wsi error]: requested level does not exist. Number of slide levels: " + str(tiles.level_count))
        return 0

    # Account for the reversed order in Deep Zoom level world
    dz_level = tiles.level_count - 1 - dz_level
    
    logging.info(f"this is dz_level {dz_level}")
    x_tiles, y_tiles = tiles.level_tiles[dz_level]
    logging.info(f"This is level_tiles {tiles.level_tiles}")
    logging.info(f"x_tiles: {x_tiles}, y_tiles: {y_tiles}")

    x, y = 0, 0
    count, batch_count = 0, 0
    patches_deconv, patches_RGB, coords, labels = [], [], [], []
    while y < y_tiles:
        while x < x_tiles:
            logging.info(f"x: {x}, y: {y}")
            new_tile = np.array(tiles.get_tile(dz_level, (x, y)), dtype=np.uint8)
            # BG subtract here before adding to patch
            is_fg = True
            if ignore_bg_percent:
                is_fg = is_foreground(new_tile, threshold=ignore_bg_percent)
                logging.info(f"Foreground? {is_fg}")

            # OpenSlide calculates overlap in such a way that sometimes depending on the dimensions, edge
            # patches are smaller than the others. We will ignore such patches.
            if np.shape(new_tile) == (patch_size, patch_size, 3):

                # skip mostly bg patches
                if ignore_bg_percent and not is_fg:
                    x += 1
                    logging.info(f"New patch is > {ignore_bg_percent * 100}% background, SAVE = FALSE.")
                    continue

                # Save tiff of RGB
                patch_path = os.path.join(db_location, prefix + "_RGB_" + file_name[:-4] + "_X" + str(x) + "_Y" + str(y) + ".tiff")
                tiff.imwrite(patch_path, new_tile)
                patches_RGB.append(new_tile)

                # Color deconv
                if color_deconv:
                    new_tile = rgb2hed(new_tile)
                    # Rescale each channel of the HED image
                    new_tile_norm = np.zeros_like(new_tile)
                    for i in range(new_tile_norm.shape[-1]):
                        if i != 2:  # DAB channel stays 0
                            # Rescale the channel to range [0, 255] for puposes of saving to tiff successfully
                            new_tile_norm[:, :, i] = exposure.rescale_intensity(new_tile[:, :, i], in_range=(np.min(new_tile[:, :, i]), np.max(new_tile[:, :, i])), out_range=(0, 255))
                    # Convert to uint8
                    new_tile = new_tile_norm.astype(np.uint8)
                    patch_path = os.path.join(db_location, prefix + "_HED_" +file_name[:-4] + "_X" + str(x) + "_Y" + str(y) + ".tiff")
                    tiff.imwrite(patch_path, new_tile)
                    patches_deconv.append(new_tile)
                coords.append(np.array([x, y]))
                logging.info(f"New patch shape is {np.shape(new_tile)}, SAVE = TRUE")
                count += 1

                # Calculate the patch label based on centre point.
                if xml_dir:
                    converted_coords = tiles.get_tile_coordinates(dz_level, (x, y))[0]
                    labels.append(generate_label(regions, region_labels, converted_coords, label_map))
            else:
                logging.info(f"SAVE = FALSE")
            x += 1

        # To save memory, we will save data into the dbs every rows_per_txn rows. i.e., each transaction will commit
        # rows_per_txn rows of patches. Write after last row regardless. HDF5 does NOT follow
        # this convention due to efficiency.
        if (y % rows_per_txn == 0 and y != 0) or y == y_tiles-1:
            if storage_option == 'disk':
                save_to_disk(db_location, patches, coords, file_name[:-4], labels)
            elif storage_option == 'lmdb':
                # LMDB by default.
                save_in_lmdb(env, patches, coords, file_name[:-4], labels)
            if storage_option != 'hdf5':
                del patches_RGB
                del patches
                del coords
                del labels
                patches, patches_RGB, coords, labels = [], [], [], [] # Reset right away.

        y += 1
        x = 0

    # Write to HDF5 files all in one go.
    if storage_option == 'hdf5':
        save_to_hdf5(db_location, prefix + "_RGB", patches_RGB, coords, file_name[:-4], labels)
        if color_deconv:
            save_to_hdf5(db_location, prefix + "_HED", patches_deconv, coords, file_name[:-4], labels)
        
        

    # Need to save tile dimensions if LMDB for retrieving patches by key.
    if storage_option == 'lmdb':
        save_meta_in_lmdb(meta_env, file_name[:-4], [x_tiles, y_tiles])

    return count


def bg_subtract(image, bg_val=255):
    """
    Background subtract from the original WSI image

    :param image: Original image array
    :type image: numpy array, as returned from read_svs
    :return: no_bg_image
    :rtype: numpy array
    """
    # Convert RGB to gray
    image_gray = rgb2gray(image)
    # Otsu Threshold
    thresh = threshold_otsu(image_gray)
    msk_background = image_gray > thresh
    # Select background pixels, set to 255 (white)
    image[msk_background] = 255
    return image


def is_foreground(tile, threshold=.5, std_threshold = .05):
    """
    Determines if the tile is majority background or not, returning a True if majority bg, False if majority tissue.

    :param tile: Background-subtracted / previously thresholded tile
    :type tile: Numpy array of RGB PIL. Image
    :param threshold: The percent of pixels at which the tile is considered a 'background' tile. Should be 0 <= threshold <= 1
    :type threshold: float
    :param bg_value: The value for which that and any greater value up to 255 is considered 'background'. This can be set via background subtraction in a previous step.
    :type bg_value: int
    :return:
    :rtype: numpy.bool_ (a Numpy array of Boolean values)
    """
    # Convert tile to numpy
    tile_arr = np.array(tile)
    # Convert RGB to gray
    image_gray = rgb2gray(tile_arr)
    # Get rid of empty patches
    if np.std(image_gray) < std_threshold:
        return False
    # Otsu Threshold
    otsu_thresh = threshold_otsu(image_gray)
    background_msk = image_gray > otsu_thresh

    bg_count = np.sum(background_msk)
    if (bg_count / background_msk.size) >= threshold:
        fg_status = False
    else:
        fg_status = True
    return fg_status


def get_level_for_magnification(slide, desired_magnification):
    ds_factor = get_downsample_factor(slide, desired_magnification)
    true_level = slide.get_best_level_for_downsample(ds_factor)
    logging.info(f"Closest level for desired magnification {desired_magnification} is {true_level}")
    return true_level


def get_downsample_factor(slide, desired_magnification):
    """
    Get the downsample factor for the desired magnification.

    :param slide:
    :type slide:
    :param desired_magnification:
    :type desired_magnification:
    :return:
    :rtype:
    """
    try:
        objective_power = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        logging.info(f"objective power is {objective_power}")
    except:
        objective_power = 40
        logging.warning(f"Objective power not located in slide metadata. Using default objective power of {objective_power} to calculate downsample factor.")
    downsample_factor = objective_power / desired_magnification
    if downsample_factor < 1:
        logging.warning(f"Objective power is {objective_power}, which is lower than requested magnification {desired_magnification}. Setting downsampling factor to 1; level 0 will be used")
        downsample_factor = 1
    elif math.log2(downsample_factor) % 1 != 0:
        ValueError("Downsample factor must be a power of 2")
    return downsample_factor
