from PIL import Image
import os

import logging
from mmcv.utils import get_logger

logger = get_logger(name='petros', log_file='image_checker_vnc.log', log_level=logging.INFO) # added by Petros
logger_error = get_logger(name='error', log_file='image_checker_errors.log', log_level=logging.INFO) # added by Petros

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
                     #except (IOError, SyntaxError) as e:
                    logger_error.info(f'Error occurred: {e}')
                    logger_error.info(f'Corrupted image: {file_path}')
                    corrupted_images.append(file_path)
    return corrupted_images

# Usage
root_folder = 'data'  # or data_150_old
corrupted_images = check_images(root_folder)
print(f"Found {len(corrupted_images)} potentially corrupted images.")


