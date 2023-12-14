# --------------------------------------------------------------------------------
# Petros Polydorou - 2023 ETH Zurich 
# Semester Project - Domain Adaptation
# Description - Divide bounding boxes
# --------------------------------------------------------------------------------

def divide_bounding_box(x1, y1, x2, y2):
    width = x2 - x1
    height = y2 - y1
    
    roi1 = (x1, y1, x1 + width // 2, y1 + height // 2)
    roi2 = (x1 + width // 2, y1, x2, y1 + height // 2)
    roi3 = (x1, y1 + height // 2, x1 + width // 2, y2)
    roi4 = (x1 + width // 2, y1 + height // 2, x2, y2)
    
    return roi1, roi2, roi3, roi4

#Define the bounding box coordinates

x1, y1 = 0, 0
x2, y2 = 100, 80

# Divide the bounding box into ROIs

roi1, roi2, roi3, roi4 = divide_bounding_box(x1, y1, x2, y2)

print("ROI 1:", roi1)
print("ROI 2:", roi2)
print("ROI 3:", roi3)
print("ROI 4:", roi4)

# save them with dictionary a

roi_dict ={'rois': [roi1, roi2, roi3, roi4], 'labels': [label1, label2, label3, label4]}