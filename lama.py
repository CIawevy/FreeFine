import os
import sys

sys.path.append('./lama-with-refiner/')
import torch
from torch.utils.data._utils.collate import default_collate
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2
from saicinpainting.evaluation.refinement import refine_predict
from saicinpainting.training.trainers import load_checkpoint
from omegaconf import OmegaConf
def get_inpaint_model(device):
    predict_config = OmegaConf.load('./lama-with-refiner/configs/prediction/default.yaml')
    predict_config.model.path = './models/big-lama/'
    predict_config.refiner.gpu_ids = '0'

    # device = torch.device(predict_config.device)
    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')

    train_config = OmegaConf.load(train_config_path)
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(predict_config.model.path,
                                   'models',
                                   predict_config.model.checkpoint)

    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)
    return model,predict_config


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


# seg_model = get_seg_model()
class lama_with_refine():
    #ClawerMadeLama
    def __init__(self,device):
        self.inpaint_model, self.predict_config = get_inpaint_model(device)
    def __call__(self,img,masks, *args, **kwargs):
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR
        # predictions, visualized_output = seg_model.run_on_image(img)
        img = img.astype('float32') / 255
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))

        batch = dict(image=img, mask=masks[None, ...])

        batch['unpad_to_size'] = [torch.tensor([batch['image'].shape[1]]), torch.tensor([batch['image'].shape[2]])]
        batch['image'] = torch.tensor(pad_img_to_modulo(batch['image'], self.predict_config.dataset.pad_out_to_modulo))[
            None].to(
            self.predict_config.device)
        batch['mask'] = torch.tensor(pad_img_to_modulo(batch['mask'], self.predict_config.dataset.pad_out_to_modulo))[
            None].float().to(self.predict_config.device)

        cur_res = refine_predict(batch, self.inpaint_model, **self.predict_config.refiner)
        cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')

        return cur_res



# def inference(img, class_name, confidence_score, sigma, mask_threshold):
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#     # predictions, visualized_output = seg_model.run_on_image(img)
#
#     img = img.astype('float32') / 255
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = np.transpose(img, (2, 0, 1))
#
#     preds = predictions['instances'].get_fields()
#
#     masks = preds['pred_masks'][
#         torch.logical_and(preds['pred_classes'] == className[class_name], preds['scores'] > confidence_score)]
#     masks = torch.max(masks, axis=0)
#     masks = masks.values.cpu().numpy()
#     masks = gaussian_filter(masks, sigma=sigma)
#     masks = (masks > mask_threshold) * 255
#
#     batch = dict(image=img, mask=masks[None, ...])
#
#     batch['unpad_to_size'] = [torch.tensor([batch['image'].shape[1]]), torch.tensor([batch['image'].shape[2]])]
#     batch['image'] = torch.tensor(pad_img_to_modulo(batch['image'], predict_config.dataset.pad_out_to_modulo))[None].to(
#         predict_config.device)
#     batch['mask'] = torch.tensor(pad_img_to_modulo(batch['mask'], predict_config.dataset.pad_out_to_modulo))[
#         None].float().to(predict_config.device)
#
#     cur_res = refine_predict(batch, inpaint_model, **predict_config.refiner)
#     cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
#
#     cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
#
#     return cur_res