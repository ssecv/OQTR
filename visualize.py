import torchvision.transforms as T

from models import build_model
from run_utils import *


parser = argparse.ArgumentParser('running script', parents=[get_args_parser()])
parser.add_argument('--input', required=True, type=str, help='input image')
parser.add_argument('--output', required=False, type=str, default='./', help='output path')
args = parser.parse_args()
args.num_queries = 10
args.slim = True
args.dataset_file = 'sis'
args.no_aux_loss = True
args.masks = True
args.saliency_query = True
device = torch.device('cpu')

# fix the seed for reproducibility
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model_without_ddp, _, postprocessor = build_model(args)
model_without_ddp.to(device)
model_without_ddp.eval()

transform = T.Compose([
    Resize([320], max_size=480),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

checkpoint = torch.load("oqtr_r50.pth", map_location='cpu')
res = model_without_ddp.load_state_dict(checkpoint['model'])
print(res)


def eval_single_image(image_path, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    im = Image.open(image_path)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    scores, boxes, seg = detect(im, model_without_ddp, transform, postprocessor,
                                device=device, masks=args.masks)
    if args.visualize_type == 'seg':
        save_segmentation(image_path, seg, out_dir)
    else:
        if args.visualize_type == 'box':
            seg = None
        plot_results(image_path, scores, boxes, seg=seg,
                     save_path=out_dir, im=im)


if __name__ == '__main__':
    eval_single_image(args.input, args.output_dir)
