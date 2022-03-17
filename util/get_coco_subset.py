import json
import random


def random_coco_subset(coco_path, subset_num):
    with open(coco_path, 'r') as f:
        coco = json.load(f)

    all_elem = list(range(len(coco['images'])))
    random.shuffle(all_elem)
    random_choice = sorted(all_elem[:subset_num])
    selected_idx = [coco['images'][idx]['id'] for idx in random_choice]
    sub_coco = {'info': coco['info'], 'licenses': coco['licenses'], 'categories': coco['categories'],
                'images': [elem for elem in coco['images'] if elem['id'] in selected_idx],
                'annotations': [elem for elem in coco['annotations'] if elem['image_id'] in selected_idx]}

    with open(coco_path.replace('.json', '_{}.json'.format(subset_num)), 'w') as f:
        json.dump(sub_coco, f)
