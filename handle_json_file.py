import pandas as pd
import json

import json

# Load the JSON file
with open('data/cityscapes/gtFine_panoptic/cityscapes_panoptic_val_original.json', 'r') as file:
    json_data = json.load(file)

# Define the target filename
target_filename = ['frankfurt_000000_000294_gtFine_leftImg8bit.png', 
                   'frankfurt_000001_011835_gtFine_leftImg8bit.png', 
                   'lindau_000000_000019_gtFine_leftImg8bit.png', 
                   'lindau_000018_000019_gtFine_leftImg8bit.png', 
                   'munster_000000_000019_gtFine_leftImg8bit.png', 
                   'munster_000052_000019_gtFine_leftImg8bit.png'] 

target_indexes_list=[]
for target in target_filename:
    # Find the index of the item with the target filename
    target_index = next((i for i, item in enumerate(json_data['images']) if item['file_name'] == target), None) # we use next() as inside we have a generator expression
    if target_index is not None:
        target_indexes_list.append(target_index)
        
# Convert the list of indices into a list of ranges
index_ranges = [(target_indexes_list[i], target_indexes_list[i + 1]) for i in range(0, len(target_indexes_list), 2)]

# Filter json_data['images'] to keep only items within these ranges
filtered_images = []
filtered_annotations = []

for i, item in enumerate(json_data['images']):
    if any(start <= i < end for start, end in index_ranges):
        filtered_images.append(item)

for i, item in enumerate(json_data['annotations']):
    if any(start <= i < end for start, end in index_ranges):
        filtered_annotations.append(item)
        
# Update json_data['images'] with the filtered list
json_data['images'] = filtered_images
json_data['annotations'] = filtered_annotations

# Specify the path for the new JSON file
new_json_file = 'data/cityscapes/gtFine_panoptic/cityscapes_panoptic_val.json'

# Write the updated json_data to a new file
with open(new_json_file, 'w') as file:
    json.dump(json_data, file, indent=4)
    
# Load the JSON file
with open('data/cityscapes/gtFine_panoptic/cityscapes_panoptic_val.json', 'r') as file:
    json_data = json.load(file)

print(f"Updated JSON data saved to {new_json_file}")

# json_data now contains the updated images list