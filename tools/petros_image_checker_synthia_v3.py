from PIL import Image
import numpy as np
import os
import logging
from mmcv.utils import get_logger

logger_error = get_logger(name='error_synthia', log_file='image_checker_errors_synthia.log', log_level=logging.INFO)
logger = get_logger(name='petros', log_file='image_checker_synthia.log', log_level=logging.INFO)

def check_images(root_folder, target_subdir):
    corrupted_images = []
    total_files_checked = 0
    logger.info(f'Checking images in folder: {root_folder}')

    for subdir, dirs, files in os.walk(root_folder):
        if subdir.endswith(target_subdir):
            logger.info(f'Checking subdir: {subdir}')
            for filename in files:
                file_path = os.path.join(subdir, filename)
                if filename.endswith('.png'):  # Checking only PNG files
                    total_files_checked += 1
                    try:
                        with Image.open(file_path) as img:
                            np_img = np.array(img)
                            if np_img.shape == ():  # Checking if the image shape is unexpected
                                raise ValueError("Invalid image shape")
                    except (IOError, SyntaxError, ValueError) as e:
                        logger_error.info(f'Error occurred with image {filename}: {e}')
                        corrupted_images.append(file_path)

    return corrupted_images, total_files_checked

root_folder = 'data/synthia'
target_subdir = 'RGB'
corrupted_images, total_files_checked = check_images(root_folder, target_subdir)
print(f"Found {len(corrupted_images)} potentially corrupted images.")
print(f"Number of images checked: {total_files_checked}")
