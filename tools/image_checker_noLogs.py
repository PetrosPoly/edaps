from PIL import Image
import os

import logging
from mmcv.utils import get_logger

def check_images(root_folder):
    corrupted_images = []
    for subdir, dirs, files in os.walk(root_folder): 
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add other image formats if needed
                file_path = os.path.join(subdir, filename)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify the integrity of the file
                except (IOError, SyntaxError) as e:
                    print('Error occurred:', e)
                    print('Corrupted image:', file_path)
                    corrupted_images.append(file_path)
    return corrupted_images

# Usage
root_folder = 'data'  # or data_150_old
corrupted_images = check_images(root_folder)
print("The number of potentially corrupted images is:", len(corrupted_images))


