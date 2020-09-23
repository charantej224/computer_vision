import copy
import json
import os

test_split = 0.2
directory = "/home/charan/Documents/research/dataset/ms_coco/annotations_trainval2017/annotations/"

train_json = os.path.join(directory, 'train_2017.json')
test_json = os.path.join(directory, 'test_2017.json')

with open(os.path.join(directory, 'instances_val2017.json')) as f:
    val_dict = json.load(f)
    f.close()

original_images = len(val_dict['images'])

test_images_count = round(original_images * 0.2)
train_images_count = original_images - test_images_count

train_dict = copy.deepcopy(val_dict)
test_dict = copy.deepcopy(val_dict)
train_dict['images'], train_dict['annotations'] = [], []
test_dict['images'], test_dict['annotations'] = [], []
print("splitting function for coco dataset")

test_image_ids = []
train_image_ids = []

counter = 0
for each in val_dict['images']:
    if counter < test_images_count:
        test_image_ids.append(each['id'])
        test_dict['images'].append(each)
    else:
        train_image_ids.append(each['id'])
        train_dict['images'].append(each)
    counter += 1

for each in val_dict['annotations']:
    if each['image_id'] in test_image_ids:
        test_dict['annotations'].append(each)
    else:
        train_dict['annotations'].append(each)

print("images")

with open(train_json, 'w') as f:
    json.dump(train_dict, f, indent=2)

with open(test_json, 'w') as f:
    json.dump(train_dict, f, indent=2)
