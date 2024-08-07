import json
from pycocotools import mask as maskUtils

input_file = 'instances_default.json'
output_file = 'overlapping_annotations.json'
category_id = 3
iou_threshold = 0.5

with open(input_file) as f:
    data = json.load(f)

annotations = data['annotations']
images = data['images']

def segmentation_to_rle(segmentation, height, width):
    if type(segmentation) == list:
        rles = maskUtils.frPyObjects(segmentation, height, width)
        rle = maskUtils.merge(rles)
    elif type(segmentation['counts']) == list:
        rle = maskUtils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return rle

def compute_iou_from_segmentation(segA, segB, height, width):
    rleA = segmentation_to_rle(segA, height, width)
    rleB = segmentation_to_rle(segB, height, width)
    iou = maskUtils.iou([rleA], [rleB], [0])[0][0]
    return iou

overlapping_annotations = []

# Create a dictionary for image dimensions
image_dimensions = {image['id']: (image['height'], image['width']) for image in images}

for i in range(len(annotations)):
    for j in range(i + 1, len(annotations)):
        ann1 = annotations[i]
        ann2 = annotations[j]

        if ann1['image_id'] != ann2['image_id'] or ann1['category_id'] != category_id or ann2['category_id'] != category_id:
            continue
        
        # Extract image dimensions
        height, width = image_dimensions[ann1['image_id']]

        # Extract segmentations
        segA = ann1['segmentation']
        segB = ann2['segmentation']

        # Compute IoU
        iou = compute_iou_from_segmentation(segA, segB, height, width)

        if iou > iou_threshold:
            overlapping_annotations.append(ann1)
            overlapping_annotations.append(ann2)

data['annotations'] = overlapping_annotations

with open(output_file, 'w') as f:
    json.dump(data, f)