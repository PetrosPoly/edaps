# -----------------------------------------------------------------------------------------
# Petros Polydorou - 2023 ETH Zurich 
# Semester Project - Domain Adaptation
# Description - Load annotations - Visualize annotated images
# Description - Divide bounding boxes into 4 rois - produce sublabels for instance classes
# -----------------------------------------------------------------------------------------

import numpy as np
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import os.path as osp
import json

from mmcv.utils import print_log
# from mmdet.core import BitmapMasks
from mmseg.utils.visualize_pred import prep_gt_pan_for_vis, subplotimgV2
from tools.panoptic_deeplab.utils import rgb2id
from cityscapesscripts.helpers.labels import id2label

def load_annotations_bbox_panoptic(data_root, ann_dir, img_dir, seg_map_suffix='.png'):
    img_infos = []
    img_dir = osp.join(data_root,img_dir)
    ann_dir = osp.join(data_root,ann_dir)
    json_filename = 'data/cityscapes/gtFine_panoptic/cityscapes_panoptic_train_trainId.json'
    #json_filename = ann_dir + '.json'
    print_log(f'Loaded annotations from : {json_filename}')
    dataset = json.load(open(json_filename))
    for ano in dataset['annotations']:
        img_info = {}
        if 'synthia' in data_root:
            ano_fname = ano['file_name']
            seg_fname = ano['image_id'] + seg_map_suffix
        elif 'cityscapes' in data_root:
            ano_fname = ano['image_id']
            str1 = ano_fname.split('_')[0] + '/' + ano_fname
            ano_fname = str1 + '_leftImg8bit.png'
            seg_fname = str1 + seg_map_suffix
        img_info['filename'] = ano_fname
        img_info['ann'] = {}
        img_info['ann']['seg_map'] = seg_fname
        img_info['ann']['segments_info'] = ano['segments_info']
        img_infos.append(img_info)
    print_log(f'Loaded {len(img_infos)} images from {ann_dir}')
    return img_infos

def subrois_per_bbox(img_infos):  # for each extract the points []
    roi_dicts = []
    for image in img_infos[:1]:
        for i , area in enumerate(image['ann']['segments_info']):
            #Label of the area
            label = area['id']
            if label > 1000:
                print(f'bbox for the area is the following', area['bbox'])
                #Define the bounding box coordinates
                x1 = area['bbox'][0]
                y1 = area['bbox'][1]  #  y1 = area['bbox'][2] 
                x2 = x1 + area['bbox'][2] - 1
                y2 = y1 + area['bbox'][3] - 1 
                bbox = [x1, y1, x2, y2]
                # Divide the bounding box into ROIs
                roi1, roi2, roi3, roi4 = divide_bounding_box_v1(bbox)
                # Define the sublabels
                slab_roi1 = area['id'] * 10 + 1
                slab_roi2 = area['id'] * 10 + 2
                slab_roi3 = area['id'] * 10 + 3
                slab_roi4 = area['id'] * 10 + 4
                # save them within a dictionary
                roi_dict = {}
                roi_dict['rois'] = [roi1, roi2, roi3, roi4]
                roi_dict['sublabels'] = [slab_roi1, slab_roi2, slab_roi3, slab_roi4]
                roi_dicts.append(roi_dict)
    return roi_dicts

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

# added by Petros
def divide_box(box):
    x1, y1, x2, y2 = box
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    # No overlapping of sub - boxes
    roi1 = (x1, y1, x1 + width // 2 - 1, y1 + height // 2 - 1)
    roi2 = (x1 + width // 2, y1, x2, y1 + height // 2 - 1)
    roi3 = (x1, y1 + height // 2 , x1 + width // 2 - 1, y2)
    roi4 = (x1 + width // 2, y1 + height // 2, x2, y2)
    return roi1, roi2, roi3, roi4
# added by Petros for contrastive

def contrastive_labels(img_infos, panoptic):  # for each extract the points []
    # panoptic = rgb2id(panoptic)
    contrastive = panoptic
    # for image in img_infos[:1]:
    image = img_infos[3]
    for i , seg in enumerate(image['ann']['segments_info']):
        #Label of the area
        label = seg['id'] 
        if label > 1000:
            mask = (panoptic == seg["id"])
            if not mask.sum() == 0:
                box = get_bbox_coord(mask)
                subboxes = [None, None, None, None]
                if isValidBox(box):
                    # build the subregion label
                    subboxes[0], subboxes[1], subboxes[2], subboxes[3] = divide_box(box)
                    for i, subbox in enumerate(subboxes):
                        x1, y1, x2, y2 = subbox
                        for y in range(y1, y2+1):
                            for x in range(x1, x2+1):
                                if panoptic[y, x] == label:
                                    contrastive [y,x] = panoptic[y,x] * 10 + i
    return contrastive

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

def save_and_plot_synthia_gt_pan(data_root, ann_dir, image_filename, outdir, roi_dicts, subrois, debug, img_infos):
    # Input filenames
    input_filename = os.path.join(data_root, ann_dir, image_filename)
    # open GT panoptic label                                
    TrgPanGT = Image.open(input_filename)
    assert TrgPanGT is not None, "file could not be read, check with os.path.exists()"
    TrgPanGT_ar = np.array(TrgPanGT).astype(np.uint32)
    TrgPanGT_ar_id = rgb2id(TrgPanGT_ar)
    # TrgPanGT_ar_id = contrastive_labels(img_infos, TrgPanGT_ar_id)
    # plot the panoptic
    fig, ax = plt.subplots(figsize=(24, 24), constrained_layout=True)
    if subrois: 
        filename = image_filename.replace('_panoptic', '_output_panoptic_with_1stcar_RoIs') 
        output_filename = os.path.join(outdir, f'{filename}')
        # for roi_dict in roi_dicts:
        # for roi_dict in roi_dicts[56:63]:
        roi_dict = roi_dicts[56]
        print(roi_dict)
        rois = roi_dict['rois'] 
        sublabels = roi_dict['sublabels']
        for roi, sublabel in zip(rois, sublabels):
            x1, y1, x2, y2 = roi
            label_text = f"{sublabel}"
            # Create a rectangle patch for the bounding box
            # TODO: Different labels with different colors and text
            # rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=6, edgecolor='yellow', facecolor='none')
            # ax.add_patch(rect)
            # Add sublabel text
            # ax.text(x1+(x2-x1)/2, y1+(y2-y1)/2, label_text, color='r', backgroundcolor='w', weight='bold')
    else: 
        filename = image_filename.replace('_panoptic', '_output_panoptic') 
        output_filename = os.path.join(outdir, f'{filename}')
    subplotimgV2(ax, prep_gt_pan_for_vis(TrgPanGT_ar_id, dataset_name=data_root, debug=debug, blend_ratio=1.0, img=None, runner_mode='val', ax=ax, label_divisor=1000), 'TrgPanGT')
    ax.axis('off')
    plt.axis('off')
    plt.show()
    plt.savefig(output_filename)
    plt.close()
    # print 
    print(f'visual saved at {output_filename}')
    
    
# def divide_isntance_masks(self, results):  
#         panoptic = results['gt_panoptic_seg']
#         segments = results['ann_info']['segments_info']
#         panoptic = rgb2id(panoptic)
#         height, width = panoptic.shape[0], panoptic.shape[1]
#         semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label
#         panoptic_only_thing_classes = np.zeros(panoptic.shape)
#         max_inst_per_class = np.zeros(len(self.thing_list))
#         class_id_tracker = {}
#         for cid in self.thing_list:
#             class_id_tracker[cid] = 1
#         gt_masks = []
#         gt_labels = []
#         gt_bboxes = []
#         gt_bboxes_ignore = np.empty([0, 4], dtype=np.float32)
#         for seg in segments:
#             cat_id = seg["category_id"]
#             if self.mode == 'val':
#                 labelInfo = id2label[cat_id]
#                 cat_id = labelInfo.trainId
#             if self.ignore_crowd_in_semantic:
#                 if not seg['iscrowd']:
#                     semantic[panoptic == seg["id"]] = cat_id
#             else:
#                 semantic[panoptic == seg["id"]] = cat_id
#             mask = (panoptic == seg["id"])
#             if not mask.sum() == 0:
#                 if self.ignore_crowd_in_instance:
#                     if not seg['iscrowd']:
#                         # gt_masks_all.append(mask.astype(np.uint8))
#                         if cat_id in self.thing_list:
#                             box = get_bbox_coord(mask)
#                             if isValidBox(box):
#                                 gt_masks.append(mask.astype(np.uint8))
#                                 gt_labels.append(self._map_instance_class_ids(cat_id))
#                                 gt_bboxes.append(box)
#                                 panoptic_only_thing_classes[panoptic == seg["id"]] = cat_id * self.label_divisor + class_id_tracker[cat_id]
#                                 class_id_tracker[cat_id] += 1
#                 else:
#                     if cat_id in self.thing_list:
#                         box = get_bbox_coord(mask)
#                         if isValidBox(box):
#                             gt_masks.append(mask.astype(np.uint8))
#                             gt_labels.append(self._map_instance_class_ids(cat_id))
#                             gt_bboxes.append(box)
#                             panoptic_only_thing_classes[panoptic == seg["id"]] = cat_id * self.label_divisor + class_id_tracker[cat_id]
#                             class_id_tracker[cat_id] += 1
#         # Divide the bounding box into ROIs
#         for gt_bbox in gt_bboxes:
#             # define the sub_bboxes
#             roi1, roi2, roi3, roi4 = divide_bounding_box(gt_bbox)
#             x1, y1, x2, y2 = map(int, roi1)
#             width = x2 - x1 + 1
#             height = y2 - y1 + 1
#             # Define the sub masks 
#             submask1 = mask[0:y1+height//2, 0:x1+width//2]
#             submask2 = mask[0:y1+height//2, x1+width//2:x2]
#             submask3 = mask[y1+height//2:y2,0:x1+width//2]
#             submask4 = mask[y1+height//2:y2,x1+width//2:x2]
#             # Define the sub labels
#             slab_roi1 = seg['id'] * 10 + 1
#             slab_roi2 = seg['id'] * 10 + 2
#             slab_roi3 = seg['id'] * 10 + 3
#             slab_roi4 = seg['id'] * 10 + 4
#             # save them within a dictionary
#             roi_dict = {}
#             roi_dict['rois'] = [roi1, roi2, roi3, roi4]
#             roi_dict['sublabels'] = [slab_roi1, slab_roi2, slab_roi3, slab_roi4]
#             roi_dicts.append(roi_dict)
#         for cid in list(class_id_tracker.keys()):
#             max_inst_per_class[self._map_instance_class_ids(cid)] = class_id_tracker[cid]
#         gt_masks = BitmapMasks(gt_masks, height, width)
#         results['gt_masks'] = gt_masks
#         results['gt_semantic_seg'] = semantic.astype('long')
#         results['gt_panoptic_only_thing_classes'] = panoptic_only_thing_classes.astype('long')
#         results['gt_labels'] = np.asarray(gt_labels).astype('long')
#         results['max_inst_per_class'] = max_inst_per_class.astype('long')
#         results['gt_bboxes'] = np.asarray(gt_bboxes).astype(np.float32)
#         results['gt_bboxes_ignore'] = gt_bboxes_ignore
#         # adding the fields
#         results['bbox_fields'] = ['gt_bboxes_ignore', 'gt_bboxes']
#         results['mask_fields'] = ['gt_masks']
#         results['seg_fields'] = ['gt_semantic_seg']
#         results['pan_fields'] = ['gt_panoptic_only_thing_classes']
#         results['maxinst_fields'] = ['max_inst_per_class']
#         return results