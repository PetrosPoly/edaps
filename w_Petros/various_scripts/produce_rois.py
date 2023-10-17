import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL
# Load your RGB image (assuming you have it)
image = PIL.open('data/synthia/panoptic-labels-crowdth-0-for-daformer/synthia_panoptic/0000000_panoptic.png')

# Create a Matplotlib figure and axis
fig, ax = plt.subplots(1)
ax.imshow(image)

# Loop through the ROI dictionaries and plot bounding boxes and sublabels
for roi_dict in roi_dicts:
    rois = roi_dict['rois']
    sublabels = roi_dict['sublabels']
    
    for roi, sublabel in zip(rois, sublabels):
        x1, y1, x2, y2 = roi
        label_text = f"Sublabel: {sublabel}"
        
        # Create a rectangle patch for the bounding box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add sublabel text
        ax.text(x1, y1, label_text, color='r', backgroundcolor='w')

# Show the image with bounding boxes and sublabels
plt.axis('off')  # Turn off axis labels
plt.show()
