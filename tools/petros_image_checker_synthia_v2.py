from PIL import Image
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
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    total_files_checked += 1
                    try:
                        with Image.open(file_path) as img:
                            img.verify()
                            img.load()  # Load image data to check for corruption
                    except Exception as e:  # Catching a broader range of exceptions
                        logger_error.info(f'Error occurred: {e}')
                        logger_error.info(f'Corrupted image: {file_path}')
                        corrupted_images.append(file_path)

    return corrupted_images, total_files_checked

root_folder = 'data/synthia'
target_subdir = 'RGB'
corrupted_images, total_files_checked = check_images(root_folder, target_subdir)
print(f"Found {len(corrupted_images)} potentially corrupted images.")
print(f"Number of images checked: {total_files_checked}")



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