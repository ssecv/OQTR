import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from collections import defaultdict


class COCOAnalyze:
    def __init__(self, coco_path, normalize=False, output_dir=None):
        self.coco = COCO(coco_path)
        self.normalize = normalize
        self.dataset_name = coco_path.split('/')[-1].split('.')[0]
        self.output_dir = os.path.join(os.path.dirname(coco_path), self.dataset_name) \
            if output_dir is None else output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _get_box_center_rate(self, ann_id):
        anns = self.coco.anns[ann_id]
        img = self.coco.imgs[anns['image_id']]
        x = anns['bbox'][0] + 0.5 * anns['bbox'][2]
        y = anns['bbox'][1] + 0.5 * anns['bbox'][3]
        img_x = img['width']
        img_y = img['height']
        return x / img_x, y / img_y

    def _plot_with_dict(self, data, title, xlabel, ylabel, bar_plot=False):
        if isinstance(data, dict):
            x = sorted(data.keys())
            y = [data[elem] for elem in x]
        else:
            x, y = data

        fig, ax = plt.subplots(figsize=(7, 7))
        if bar_plot:
            ax.bar(x, y)
        else:
            ax.plot(x, y)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.legend(['如果有多条线，则是每条线代表的意思'])

        return fig

    def average_instance_num(self):
        """
        Average instance number of a single image
        """
        return len(self.coco.anns) / len(self.coco.imgs)

    def instance_distribute(self):
        """
        Instance number ratio distribution
        """
        num_dict = defaultdict(float)
        total_num = len(self.coco.dataset['images'])
        for ins in self.coco.imgToAnns.values():
            num = len(ins)
            num_dict[num] += 1

        for k in num_dict.keys():
            num_dict[k] = num_dict[k] / total_num

        return num_dict

    def box_center_map(self, map_shape=(16, 16)):
        """
        Box center map of all instances
        """
        map_arr = np.zeros(map_shape)
        for img_id in self.coco.anns.keys():
            x, y = self._get_box_center_rate(img_id)
            map_arr[int(y * map_shape[0])][int(x * map_shape[1])] += 1

        return map_arr / map_arr.max()

    def size_distribute(self):
        """
        Instance size distribution, normalized by the area of total image
        """
        size_list = []
        for ins in self.coco.anns.values():
            image_dict = self.coco.imgs[ins['image_id']]
            image_size = image_dict['width'] * image_dict['height']
            size_list.append(ins['area'] / image_size)

        size_list = sorted(size_list, reverse=True)
        x = np.linspace(0, 1, num=len(size_list))

        return x, size_list

    def analyze(self):
        avg_ins_num = self.average_instance_num()

        ins_dist = self.instance_distribute()
        ins_fig = self._plot_with_dict(ins_dist, 'Instance Number Distribution',
                                       'instance number', 'image number', bar_plot=True)
        ins_fig.savefig(os.path.join(self.output_dir, 'instance_number_dist.png'))

        center_map = self.box_center_map()
        int_center_map = (center_map * 255).astype('uint8')
        heat_map = cv2.applyColorMap(int_center_map, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(self.output_dir, 'center_map.png'), int_center_map)
        cv2.imwrite(os.path.join(self.output_dir, 'center_heat_map.png'), heat_map)

        size_dist = self.size_distribute()
        size_fig = self._plot_with_dict(size_dist, 'Instance Size Distribution',
                                        'instance size', 'image number')
        size_fig.savefig(os.path.join(self.output_dir, 'instance_size_dist.png'))

        print('Analyze {} Result'.format(self.dataset_name))
        print('Average Instance Number: {}'.format(avg_ins_num))
        print('Instance Number Dict: {}'.format(ins_dist))


if __name__ == '__main__':
    # analyzer = COCOAnalyze('/Users/patrick/Desktop/Li_train500.json', normalize=True)
    analyzer = COCOAnalyze('/Users/patrick/Desktop/SIS10K_train.json', normalize=True)
    analyzer.analyze()
