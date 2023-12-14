# --------------------------------------------------------------------------------
# Petros Polydorou - 2023 ETH Zurich
# Semester Project - Domain Adaptation
# Name - new_panoptic_and_rois_visualize.py
# Description - Load a gt_panoptic image and visualize it with colors
# --------------------------------------------------------------------------------

import argparse
import os

from functions import load_annotations_bbox_panoptic, subrois_per_bbox, save_and_plot_synthia_gt_pan


if __name__ == '__main__':
    
    # # Define and parse command-line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--subrois', action='store_true', help='rois to be generated for each bounding box') # If python3 new_panoptic_and_rois_vis.py --subrois then subrois is True elseif python3 new_panoptic_and_rois_vis.py (without --subrois) then subrois is False
    # args = parser.parse_args() 
    # print('subrois =', args.subrois)
    # print('RoIs for bounding box will be generated' if args.subrois else 'RoIs for bounding box will NOT be generated')
    
    subrois = False
    
    # Filenames
    data_root = 'data/synthia'
    ann_dir='panoptic-labels-crowdth-0-for-daformer/synthia_panoptic'
    img_dir = 'RGB'
    image_filename = '0000000_panoptic.png'
    outdir='w_Petros/bbox_visualization_per_instance/output_visualizations/synthia/panoptic_gt/'

    # Create the output directory & check if the directory exists, and raise an AssertionError if not
    os.makedirs(outdir, exist_ok=True)
    assert os.path.exists(outdir), f"Directory '{outdir}' does not exist."
    debug = False

    # Load all ground truth panoptic meta info
    img_infos = load_annotations_bbox_panoptic(data_root, ann_dir, img_dir)
    print('done')
    # # Divide the bounding boxes into 4 rois
    roi_dicts = subrois_per_bbox(img_infos) if subrois else None
    
    # Plot the RGB image with the bounding boxes
    save_and_plot_synthia_gt_pan(data_root, ann_dir, image_filename, outdir, roi_dicts, subrois ,debug, img_infos)