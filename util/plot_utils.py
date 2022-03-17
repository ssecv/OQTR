"""
Plotting utilities to visualize training logs.
"""
import os
import cv2
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path, PurePath


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'),
              ewm_col=0, log_name='log.txt'):
    """
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    """
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if 'mAP' in field:
                if '50' in field:
                    map_idx = 1
                elif '75' in field:
                    map_idx = 2
                elif 'mAPS' in field:
                    map_idx = 3
                elif 'mAPM' in field:
                    map_idx = 4
                elif 'mAPL' in field:
                    map_idx = 5
                else:
                    map_idx = 0

                if field == 'mAP-mask':
                    coco_eval = pd.DataFrame(
                        np.stack(df.test_coco_eval_masks.dropna().values)[:, map_idx]
                    ).ewm(com=ewm_col).mean()
                else:
                    coco_eval = pd.DataFrame(
                        np.stack(df.test_coco_eval_bbox.dropna().values)[:, map_idx]
                    ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)

    return fig, axs


def save_plot(log_path_list, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), save_name='result.png'):
    if isinstance(log_path_list, list):
        log = [Path(log_path) for log_path in log_path_list]
    else:
        log = Path(log_path_list)
    fig, _ = plot_logs(log, fields)
    fig.savefig(os.path.join(log_path_list[0], save_name))


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def plot_results(img_path, prob, boxes, seg=None, save_path=None, show=False,
                 im=None, classes=('N/A', 'saliency')):
    """
    Plot boxes and masks to the original images.
    """
    pil_img = im if im is not None else Image.open(img_path)
    if seg is not None:
        all_seg = seg
        img_size = pil_img.size
        pil_img = np.asarray(pil_img) * 0.7
        for idx in range(all_seg.shape[0]):
            seg = all_seg[idx]
            seg = cv2.resize(seg, img_size)
            if len(seg.shape) < 3:
                seg = np.stack([seg] * 3, axis=-1)
            r_seg = seg[:, :, 0]
            g_seg = seg[:, :, 1]
            b_seg = seg[:, :, 2]
            r_seg[r_seg > 0] = COLORS[idx % 6][0] * 100
            g_seg[g_seg > 0] = COLORS[idx % 6][1] * 100
            b_seg[b_seg > 0] = COLORS[idx % 6][2] * 100
            seg = np.stack([r_seg, g_seg, b_seg], axis=-1)
            pil_img += seg * 0.3
        pil_img = pil_img.astype('int')
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{classes[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        output_path = 'tmp.png'
        plt.savefig(output_path,
                    bbox_inches='tight', pad_inches=0)
        print('Save {} successfully.'.format(output_path))
    if show:
        plt.show()
    plt.close()


def save_segmentation(img_path, seg, save_path):
    """
    Visualize saliency maps only.
    """
    im = Image.open(img_path)
    composite = np.zeros(seg[0].shape, dtype=np.uint8)
    for idx, mask in enumerate(seg):
        composite[mask == 255] = idx + 1

    composite = Image.fromarray(composite)
    composite = composite.convert('P')
    composite.putpalette(
        [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 255, 255, 255, 0, 255, 255, 255, 0, 255, 255, 255])
    # composite.show()
    composite = composite.resize(im.size)
    save_img_path = 'tmp.png'
    composite.save(save_img_path)
    print('Save {} successfully.'.format(save_img_path))
