import numpy as np
import torch

def divide_bounding_box_v1(box):
    x1, y1, x2, y2 = box
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    # No overlapping of sub - boxes
    roi1 = (x1, y1, x1 + width // 2 - 1, y1 + height // 2 - 1)
    roi2 = (x1 + width // 2, y1, x2, y1 + height // 2 - 1)
    roi3 = (x1, y1 + height // 2 , x1 + width // 2 - 1, y2)
    roi4 = (x1 + width // 2, y1 + height // 2, x2, y2)
    return roi1, roi2, roi3, roi4

def divide_bounding_box_v2(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    # Overlapping of sub - boxes
    roi1 = (x1, y1, x1 + width // 2, y1 + height // 2)
    roi2 = (x1 + width // 2, y1, x2, y1 + height // 2)
    roi3 = (x1, y1 + height // 2, x1 + width // 2, y2)
    roi4 = (x1 + width // 2, y1 + height // 2, x2, y2)
    return roi1, roi2, roi3, roi4 

def isValidBox(box):
    isValid = False
    x1, y1, x2, y2 = box
    if x1 < x2 and y1 < y2:
        isValid = True
    return isValid

def get_bbox_coord(mask):
    # bbox computation for a segment
    hor = np.sum(mask, axis=0)
    hor_idx = np.nonzero(hor)[0]
    x = hor_idx[0]
    width = hor_idx[-1] - x + 1
    vert = np.sum(mask, axis=1)
    vert_idx = np.nonzero(vert)[0]
    y = vert_idx[0]
    height = vert_idx[-1] - y + 1
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + int(width) - 1
    y2 = y1 + int(height) - 1
    bbox = [x1, y1, x2, y2]
    return bbox

if __name__ == '__main__':
    
    version = True # if version 1 or 2 will be used 
    
    # Sample data
    image = np.array([
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        [[17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32]],
        [[33, 34, 35, 36], [37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]]
    ])

    # using a ramdom mask
    mask = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 1]
    ])

    # Produce the box and sub-boxes
    bbox = get_bbox_coord(mask)
    if isValidBox(bbox):
        rpn1, rpn2, rpn3, rpn4 = divide_bounding_box_v1(bbox) if version else divide_bounding_box_v2(bbox)
    rpn_list = [rpn1, rpn2, rpn3, rpn4]
    
    # Extraxt region of interest 
    submask = []
    features = []
    for rpn in rpn_list:
        x1, y1, x2, y2 = rpn
        
        # focus working with rpn1 rpn2 rpn3 rpn4
        submask_rpn = mask[y1:y2+1, x1:x2+1]  # the slice operation start:end includes the start index but excludes the end index.
        submask.append(submask_rpn)
        
        roi = image[:, y1:y2+1, x1:x2+1]
        # Extract features using submaskr
        feature_rpn = roi * submask_rpn
        features.append(feature_rpn)

    print(features)