from PIL import Image
import os

import logging
from mmcv.utils import get_logger

logger_error = get_logger(name='error_synthia', log_file='image_checker_errors_synthia.log', log_level=logging.INFO) # added by Petros
logger = get_logger(name='petros', log_file='image_checker_synthia.log', log_level=logging.INFO) # added by Petros

def check_images(root_folder):
    corrupted_images = []
    logger.info(f'loading file - 20.11.2023 - PETROS DEBUG - ROOT] {root_folder}')  # added by Petros
    for subdir, dirs, files in os.walk(root_folder):
        logger.info(f'loading file - 20.11.2023 - PETROS DEBUG - SUBDIR] {subdir}')     # added by Petros
        logger.info(f'loading file - 20.11.2023 - PETROS DEBUG - DIRS] {dirs}')         # added by Petros
        logger.info(f'loading file - 20.11.2023 - PETROS DEBUG - FILES] {files}')    
        for filename in files:
            logger.info(f'loading file - 20.11.2023 - PETROS DEBUG - FILENAME] {filename}') # added by Petros
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add other image formats if needed
                file_path = os.path.join(subdir, filename)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify the integrity of the file
                except (IOError) as e:
                # except (IOError, SyntaxError) as e:
                    logger_error.info(f'Error occurred: {e}')
                    logger_error.info(f'Corrupted image: {file_path}')
                    corrupted_images.append(file_path)
    return corrupted_images

# Usage
root_folder = 'data/synthia' # 'data/synthia/panoptic-labels-crowdth-0-for-daformer/synthia_panoptic' for gt panoptic check
corrupted_images = check_images(root_folder)
print(f"Found {len(corrupted_images)} potentially corrupted images.")




# def check_images(root_folder):
#     corrupted_images = []
#     for subdir, dirs, files in os.walk(root_folder):
#         for filename in files:
#             if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add other image formats if needed
#                 try:
#                     with Image.open(os.path.join(root_folder, filename)) as img:
#                         img.verify()  # Verify the integrity of the file
#                 except (IOError, SyntaxError) as e:
#                     print(f'Error occurred: {e}')
#                     print(f'Corrupted image: {filename}')
#                     corrupted_images.append(filename)
#     return corrupted_images