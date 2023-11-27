from PIL import Image
import os
import logging
from mmcv.utils import get_logger
import os
import mmcv
from mmseg.datasets.pipelines.loading import LoadImageFromFile

# Assuming LoadImageFromFile is defined as in your provided code
image_loader = LoadImageFromFile(to_float32=False, color_type='color')

root_folder = 'data/synthia/RGB'
file_names = os.listdir(root_folder)

corrupted_images = []

for file_name in file_names:
    if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        image_path = os.path.join(root_folder, file_name)
        image_info = {
            'img_prefix': None,
            'img_info': {'filename': image_path}
        }

        try:
            loaded_image = image_loader(image_info)
            # Additional checks can be performed here if necessary
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            corrupted_images.append(image_path)

print(f"Found {len(corrupted_images)} potentially corrupted images.")
