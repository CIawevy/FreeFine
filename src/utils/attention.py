import math
from typing import Union, Tuple, List, Callable, Dict, Optional
import torch
import torch.nn.functional as nnf
from diffusers import DiffusionPipeline
import numpy as np
import time
# from IPython.display import display
from PIL import Image
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import abc
import torch

import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

# import ptp_utils
# import seq_aligner

# from FPE
# def register_attention_control_new(model, controller):
#     def ca_forward(self, place_in_unet):
#         to_out = self.to_out
#         if type(to_out) is torch.nn.modules.container.ModuleList:
#             to_out = self.to_out[0]
#         else:
#             to_out = self.to_out
#
#         def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
#             x = hidden_states
#             context = encoder_hidden_states
#             mask = attention_mask
#             batch_size, sequence_length, dim = x.shape
#             h = self.heads
#             q = self.to_q(x)
#             is_cross = context is not None
#             context = context if is_cross else x
#             k = self.to_k(context)
#             v = self.to_v(context)
#             q = self.head_to_batch_dim(q)
#             k = self.head_to_batch_dim(k)
#             v = self.head_to_batch_dim(v)
#
#             sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
#
#             if mask is not None:
#                 mask = mask.reshape(batch_size, -1)
#                 max_neg_value = -torch.finfo(sim.dtype).max
#                 mask = mask[:, None, :].repeat(h, 1, 1)
#                 sim.masked_fill_(~mask, max_neg_value)
#
#             # attention, what we cannot get enough of
#             attn = sim.softmax(dim=-1)
#             # if is_cross:
#             #     ssss = []
#             attn = controller(attn, is_cross, place_in_unet)
#             out = torch.einsum("b i j, b j d -> b i d", attn, v)
#             out = self.batch_to_head_dim(out)
#             return to_out(out)
#
#         return forward
#
#     class DummyController:
#
#         def __call__(self, *args):
#             return args[0]
#
#         def __init__(self):
#             self.num_att_layers = 0
#
#     if controller is None:
#         controller = DummyController()
#
#     def register_recr(net_, count, place_in_unet):
#         # print(net_.__class__.__name__)
#         if net_.__class__.__name__ == 'Attention':
#             net_.forward = ca_forward(net_, place_in_unet)
#             return count + 1
#         elif hasattr(net_, 'children'):
#             for net__ in net_.children():
#                 count = register_recr(net__, count, place_in_unet)
#         return count
#
#     cross_att_count = 0
#     sub_nets = model.unet.named_children()
#     # print(sub_nets)
#     for net in sub_nets:
#         if "down" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "down")
#         elif "up" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "up")
#         elif "mid" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "mid")
#
#     controller.num_att_layers = cross_att_count
import torch.nn as nn


class MaskDropBlock(nn.Module):
    def __init__(self, block_size, drop_prob):
        super(MaskDropBlock, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, mask):
        if not self.training:
            return mask

        n, c, h, w = mask.shape
        # 计算每个维度上的patch数量
        patch_h = (h + self.block_size - 1) // self.block_size
        patch_w = (w + self.block_size - 1) // self.block_size

        # 生成patch维度的伯努利分布，使用1 - drop_prob，因为我们希望得到的是保持的比例
        patch_drop_mask = torch.bernoulli(torch.full((n, c, patch_h, patch_w), 1 - self.drop_prob, device=mask.device))

        # 将patch mask扩展到完整的mask尺寸
        full_drop_mask = patch_drop_mask.repeat_interleave(self.block_size, dim=2).repeat_interleave(self.block_size, dim=3)
        if full_drop_mask.shape[2] > h:
            full_drop_mask = full_drop_mask[:, :, :h]
        if full_drop_mask.shape[3] > w:
            full_drop_mask = full_drop_mask[:, :, :, :w]

        # 应用dropout mask
        return mask * full_drop_mask
def override_forward(self):

    def forward(
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        h_sample: Optional[torch.FloatTensor] = None,
        copy: Optional[torch.FloatTensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        last_up_block_idx: int = None,
    ):
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        down_block_samples = []


        # for downsample_block in self.down_blocks:
        for i, downsample_block in enumerate(self.down_blocks):

            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            # replace downsample
            # if h_sample is not None:
            #     if i == 2:
            #         sample = h_sample


            down_block_res_samples += res_samples
            down_block_samples.append(sample)
            # down_block_samples += sample

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        copy_down_block = down_block_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # replace bootleneck
        if h_sample is not None:
            if sample.shape[0]==4:#[uncon_edit,uncon_ref,con_edit,con_ref]
                #only replace edit streams' h_sample
                sample = torch.cat((h_sample[0,None],sample[1,None],h_sample[1,None],sample[3,None]))
            else:
                sample = h_sample

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
        # only difference from diffusers:
        # save the intermediate features of unet upsample blocks
        # the 0-th element is the mid-block output
        all_intermediate_features = [sample]
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

            # replace upsample
            # if h_sample is not None:
            #     if i == 2:
            #         sample = h_sample

            all_intermediate_features.append(sample)

            # return early to save computation time if needed
            if last_up_block_idx is not None and i == last_up_block_idx:
                return all_intermediate_features

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # only difference from diffusers, return intermediate results_0
        if return_intermediates:
            if copy is not None:
                return sample, all_intermediate_features, copy_down_block
            else:
                return sample, all_intermediate_features
        else:
            return sample

    return forward
#from p2p
def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, ):
            is_cross = encoder_hidden_states is not None

            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)  #attention mask can be passed

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)


            # SDSA Controller
            controller.heads = self.heads
            controller.upcast_attention = self.upcast_attention
            controller.scale = self.scale
            controller.upcast_softmax=self.upcast_softmax
            if place_in_unet in ['up'] and not is_cross and controller.share_attn:
            # if not is_cross and controller.share_attn:
                hidden_states = controller.mutual_self_attention(query,key,value,is_cross, place_in_unet)
            elif controller.local_edit and is_cross:
                hidden_states = controller.modulate_local_cross_attn(query,key,value,is_cross, place_in_unet)
            elif  place_in_unet in ['up'] and is_cross and controller.bidirectional_loc:#Expansion INV Process
                hidden_states = controller.loc_cross_attn_mask_expansion(query, key, value, is_cross, place_in_unet)
            else:
                query = self.head_to_batch_dim(query)
                key = self.head_to_batch_dim(key)
                value = self.head_to_batch_dim(value)

                attention_probs = self.get_attention_scores(query, key, attention_mask)

                attention_probs = controller(attention_probs, is_cross, place_in_unet) #layer step mask log

                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            #feature injection modulation
            if place_in_unet in ['up'] and not is_cross and controller.DIFT_FEATURE_INJ:
                hidden_states = controller.feature_injection(hidden_states, is_cross, place_in_unet)

            return hidden_states

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.LOW_RESOURCE = False
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)

        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):

                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

class Mask_Expansion_SELF_ATTN(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}
        # return {"down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.log_mask:
            if self.use_cfg:#single img, double img,later reference
                h = attn.shape[0]
                bs = h // self.heads
                if bs > 1:#[[uncon_edit,uncon_ref,con_edit,con_ref]]
                    #1.Log con edit mask
                    idx = h // 2 + h // 4
                    attn[h // 2:idx] = self.forward(attn[h // 2:idx], is_cross, place_in_unet)
                else:#[uncon_edit,con_edit]
                    attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                bs = h // self.heads
                if bs > 1:# [edit,ref]
                    #log edit
                    attn[:h // 2] = self.forward(attn[:h // 2], is_cross, place_in_unet)
                else:#[img]
                    #log img
                    attn = self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    def mask_logging(self, attn, is_cross: bool, place_in_unet: str):
        if self.log_mask:
            if self.use_cfg:#single img, double img,later reference
                h = attn.shape[0]
                bs = h // self.heads
                if bs == 4:#[[uncon_edit,uncon_ref,con_edit,con_ref]]
                    #1.Log con edit mask
                    idx = h // 2 + h // 4
                    attn[h // 2:idx] = self.forward(attn[h // 2:idx], is_cross, place_in_unet)
                else:#[uncon_edit,con_edit]
                    attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                bs = h // self.heads
                if bs > 1:# [edit,ref]
                    #log edit
                    attn[:h // 2] = self.forward(attn[:h // 2], is_cross, place_in_unet)
                else:#[img]
                    #log img
                    attn = self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if is_cross:
            pass
        elif self.log_mask:
            key = f"{place_in_unet}_{'self'}"
            self.step_store[key].append(self.get_correlation_mask(attn))
        return attn

    def temp_view(self, mask, title='Mask', name=None):
        """
        显示输入的mask图像

        参数:
        mask (torch.Tensor): 要显示的mask图像，类型应为torch.bool或torch.float32
        title (str): 图像标题
        """
        # 确保输入的mask是float类型以便于显示
        if isinstance(mask, np.ndarray):
            mask_new = mask
        else:
            mask_new = mask.float()
            mask_new = mask_new.detach().cpu()
            mask_new = mask_new.numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(mask_new, cmap='gray')
        plt.title(title)
        plt.axis('off')  # 去掉坐标轴
        # plt.savefig(name+'.png')
        plt.show()
    def view(self,mask, title='Mask'):
        """
        显示输入的mask图像

        参数:
        mask (torch.Tensor): 要显示的mask图像，类型应为torch.bool或torch.float32
        title (str): 图像标题
        """
        # 确保输入的mask是float类型以便于显示
        mask_new = mask.float()
        mask_new = mask_new.detach().cpu()
        plt.figure(figsize=(6, 6))
        plt.imshow(mask_new.numpy(), cmap='gray')
        plt.title(title)
        plt.axis('off')  # 去掉坐标轴
        plt.show()

    def dilation_operation(self, mask, kernel_size=5):
        mask = mask.float()
        # 创建结构元素
        kernel = self.get_structuring_element(kernel_size)
        # 膨胀
        opened = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        opened = (opened > 0).float()
        return opened

    def closing_operation(self, mask, kernel_size=5):
        mask = mask.float()
        # 创建结构元素
        kernel = self.get_structuring_element(kernel_size)
        # 先膨胀
        dilated = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        dilated = (dilated > 0).float()
        # 后腐蚀
        closed = F.conv2d(dilated.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        closed = (closed == torch.max(closed)).float()
        return closed
    def normalize_expansion_mask(self, mask, exp_mask, roi_expansion,local_box=None):
        if local_box is not None:
            candidate_mask = torch.zeros_like(local_box)
            box_index = (local_box>125).bool()

            box_value = exp_mask[box_index]
            ma, mi = box_value.max(), box_value.min()
            box_norm = (box_value - mi) / (ma - mi)
            candidate_mask[box_index] = box_norm
        else:
            if roi_expansion:
                candidate_mask = torch.ones_like(exp_mask)
                expansion_loc = mask < 125
                average_expansion_masks_of_interested = exp_mask[expansion_loc]
                ma, mi = average_expansion_masks_of_interested.max(), average_expansion_masks_of_interested.min()
                average_expansion_masks_norm = (average_expansion_masks_of_interested - mi) / (ma - mi)
                candidate_mask[expansion_loc] = average_expansion_masks_norm
            else:
                ma, mi = exp_mask.max(), exp_mask.min()
                average_expansion_masks_norm = (exp_mask - mi) / (ma - mi)
                candidate_mask = average_expansion_masks_norm
        return candidate_mask
    def find_edge_pixels(self,mask):
        # 使用binary_dilation找到边缘
        kernel = torch.tensor([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]], device=mask.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mask = mask.float().unsqueeze(0).unsqueeze(0)
        dilated_mask = F.conv2d(mask, kernel, padding=1) > 0
        # 确保dilated_mask也是布尔类型
        dilated_mask = dilated_mask.bool()
        edge_pixels = dilated_mask & ~mask.bool()
        return edge_pixels.squeeze(0).squeeze(0)

    def bfs_distance_transform(self, expand_mask_bool, edge_pixels):
        # 初始化距离映射为无穷大
        distance_map = torch.full_like(expand_mask_bool, float('inf'), dtype=torch.float32)

        # 将边缘像素的距离设为0
        queue = [(x.item(), y.item()) for x, y in torch.nonzero(edge_pixels, as_tuple=False)]
        for x, y in queue:
            distance_map[x, y] = 0

        # 使用BFS计算距离，仅对expand_mask_bool为True的点计算距离
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            x, y = queue.pop(0)
            current_distance = distance_map[x, y]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < expand_mask_bool.shape[0] and 0 <= ny < expand_mask_bool.shape[1]:
                    if expand_mask_bool[nx, ny] and distance_map[nx, ny] > current_distance + 1:
                        distance_map[nx, ny] = current_distance + 1
                        queue.append((nx, ny))

        return distance_map
    def Mask_post_process(self, binary_mask, ref_mask, type):

        ref_mask_bool = ref_mask > 0
        # 找到原始mask的边缘像素
        edge_pixels = self.find_edge_pixels(ref_mask_bool)
        edge_coords = torch.nonzero(edge_pixels, as_tuple=False)


        expand_mask_bool = binary_mask

        # 腐蚀
        expand_mask_bool = self.erode_operation(expand_mask_bool, kernel_size=10)
        expand_regions = expand_mask_bool.bool() & ~ref_mask_bool.bool()

        # 不连通的位置直接去掉
        if type == 'hard':
            # 初始化访问标记 DFS搜索通路
            visited = set()

            # 从边缘像素开始进行DFS
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for coord in edge_coords:
                x, y = coord[0].item(), coord[1].item()
                self.dfs(x, y, expand_regions, visited, directions)

            # 构建refined mask
            refined_mask = torch.zeros_like(expand_regions, dtype=torch.bool)
            for (x, y) in visited:
                refined_mask[x, y] = True

            # 确保包含原始mask的像素
            refined_mask |= ref_mask_bool

        refined_mask = self.dilation_operation(refined_mask, kernel_size=15)

        return refined_mask.to(torch.uint8) * 255

    def erode_operation(self, mask, kernel_size=5):
        mask = mask.float()
        # 创建结构元素
        kernel = self.get_structuring_element(kernel_size)
        # 腐蚀
        eroded = F.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding='same').squeeze(0).squeeze(0)
        eroded = (eroded == torch.max(eroded)).float()

        return eroded

    def get_structuring_element(self, kernel_size):
        # 创建一个正方形的结构元素
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)
        # 将kernel移到和mask相同的设备上
        kernel = kernel.to(self.device)
        return kernel
    def dfs(self,x, y, mask, visited, directions):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if self.is_valid(nx, ny, mask) and (nx, ny) not in visited:
                    stack.append((nx, ny))

    def is_valid(self, x, y, mask):
        # 检查像素是否在mask范围内且值为True
        return 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x, y]
    def normalize_and_binarize(self,mask_logits,ref_mask,roi_expansion,mask_threshold,otsu=False,local_box=None):
        # norm_exp_mask = self.normalize_expansion_mask(ref_mask, mask_logits, roi_expansion,local_box )
        ma, mi = mask_logits.max(), mask_logits.min()
        norm_exp_mask = (mask_logits - mi) / (ma - mi)
        if otsu:
            binary_mask = self.mask_with_otsu_pytorch(norm_exp_mask)
        else:
            norm_exp_mask = norm_exp_mask > mask_threshold
            binary_mask = norm_exp_mask.to(mask_logits.dtype)
        return binary_mask
    def fetch_expansion_mask_from_store(self, expansion_masks, obj_mask,dfs_mask, roi_expansion=True, post_process='hard', mask_threshold=0.15,otsu=False,local_box=None):
        dtype = self.obj_mask.dtype
        binary_exp_mask = self.normalize_and_binarize(expansion_masks,obj_mask,roi_expansion,mask_threshold,otsu,local_box)
        if post_process is not None:
            assert post_process in ['hard'], f'not implement method: {post_process}'
            # post process based on distance and init thr
            final_mask = self.Mask_post_process(binary_exp_mask, dfs_mask, post_process,)
        else:
            final_mask = binary_exp_mask * 255
        return final_mask.to(dtype)

    def otsu_thresholding_torch(self,image):

        hist = torch.histc(image.float(), bins=256, min=0, max=255).to(image.device)


        pixel_sum = torch.sum(hist)
        weight1 = torch.cumsum(hist, 0)
        weight2 = pixel_sum - weight1

        bin_edges = torch.arange(256).float().to(image.device)


        mean1 = torch.cumsum(hist * bin_edges, 0) / (weight1 + (weight1 == 0))
        mean2 = (torch.cumsum(hist.flip(0) * bin_edges.flip(0), 0) / (weight2.flip(0) + (weight2.flip(0) == 0))).flip(0)


        inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        threshold = torch.argmax(inter_class_variance)
        # print(f'threshold:{threshold/255}')


        binary_image = torch.where(image <= threshold, 0, 255)
        return binary_image.type(torch.uint8)

    def mask_with_otsu_pytorch(self,tensor: torch.Tensor):
        image = (tensor * 255).to(torch.uint8)

        binary_image = self.otsu_thresholding_torch(image)
        return (binary_image / 255).to(tensor.dtype)
    def between_steps(self):
        self.step_num += 1

        if self.SDSA_REF_MASK is not None:
            # Dropout ref mask for each step without gradients
            with torch.no_grad():
                # SDSA_REF_MASK_STEP = F.dropout(self.SDSA_REF_MASK, p=self.drop_out)
                SDSA_REF_MASK_STEP = self.drop_block(self.SDSA_REF_MASK.unsqueeze(0).unsqueeze(0)).squeeze(0, 1)
                SDSA_REF_MASK_STEP[SDSA_REF_MASK_STEP > 0] = 1
                self.SDSA_REF_MASK_STEP = SDSA_REF_MASK_STEP
        if self.log_mask:
            if not hasattr(self, 'device'):
                self.device = self.obj_mask.device
            mask_=torch.zeros_like(self.obj_mask)
            h,w = mask_.shape
            attention_map_length = 0
            for k,v in self.step_store.items():
                if self.is_locating:
                    weight = 1.0 if 'cross' in k else 0.0
                else:
                    weight = 1.0 if 'cross' in k else 1.0
                # weight = 1.0 if 'self' in k else 0.0
                # weight = 1.0 if 'cross' in k else 0.0
                # weight = 0.5 if 'self' in k else 5.0
                for f_t in v:
                    fh,fw = f_t.shape
                    d_ratio =2**int(math.log2(h/fh)+0.5)
                    if d_ratio > 16 :
                        continue
                        # pass
                    if mask_.shape != f_t.shape:
                        f_t = F.interpolate(f_t.unsqueeze(0).unsqueeze(0), size=mask_.shape,
                                                    mode='nearest').squeeze(0).squeeze(0)
                    mask_ += f_t * weight
                    attention_map_length+=1
            mask_ /= attention_map_length
            # self.expansion_mask_store[f'step{self.cur_step}'] = mask_
            if self.expansion_mask_on_the_fly is None:
                self.expansion_mask_on_the_fly = mask_
            else:
                self.expansion_mask_on_the_fly += mask_

            expansion_masks = self.expansion_mask_on_the_fly / self.step_num

            if self.forbit_expand_area is None:
                self.forbit_expand_area = self.obj_mask

            # expansion_masks = self.expansion_mask_on_the_fly
            # if self.step_num==1 and self.is_locating:
            if self.step_num==1:
                self.original_obj_mask = self.obj_mask
            if self.is_locating: #enhance init ca
                # auto threshold but avoid mask shrinking
                candidate_mask = self.fetch_expansion_mask_from_store(expansion_masks, self.original_obj_mask, self.original_obj_mask,
                                                                      roi_expansion=False,mask_threshold=0.15,otsu=True,)
                #expand_mask -> expand + obj mask
                expand_mask = torch.where(~(self.forbit_expand_area > 0).bool(), candidate_mask, 0)
                if expand_mask.sum() == 0:  # first step enhance
                    self.is_locating = True
                else:
                    self.is_locating = False
                self.obj_mask = expand_mask
            else:
                candidate_mask = self.fetch_expansion_mask_from_store(expansion_masks, self.original_obj_mask,
                                                                      self.original_obj_mask,
                                                                      roi_expansion=False, mask_threshold=0.15,
                                                                      otsu=True, )
                # expand_mask -> expand+obj mask
                expand_mask = torch.where(~(self.forbit_expand_area > 0).bool(), candidate_mask, 0)
                if not self.last_step:
                    self.obj_mask = expand_mask
                else:
                    self.obj_mask = candidate_mask

            # self.expansion_mask_on_the_fly=None
            self.step_store = self.get_empty_store()


    def contrast_operation(self,mask,contrast_beta,clamp=True,min_v=0,max_v=1,dim=-1,local_box=None):
        if local_box is not None:
            if dim is not None:
                mean_value = mask.mean(dim=dim)[:, :, None]
            else:
                mean_value = mask.mean()
            contrast_beta = 5.0
            # increase varience
            contrasted_mask = mask.clone()
            in_box_value = contrasted_mask[local_box.bool()]
            contrasted_mask_in_box = (in_box_value - mean_value) * contrast_beta + mean_value
            contrasted_mask[local_box.bool()] = contrasted_mask_in_box
            del mean_value,contrasted_mask_in_box,in_box_value
            if clamp:
                return contrasted_mask.clamp(min=min_v, max=max_v)
            return contrasted_mask
        else:
            if dim is not None:
                mean_value = mask.mean(dim=dim)[:,:,None]
            else:
                mean_value = mask.mean()

            #increase varience
            contrasted_mask = (mask - mean_value) * contrast_beta + mean_value
            del mean_value
            if clamp:
                return contrasted_mask.clamp(min=min_v,max=max_v)
            return contrasted_mask
    def get_down_h_w(self,d_ratio,h,w,seq):
        if not hasattr(self,'down_sampling_shape'):
            self.down_sampling_shape=dict()
        else:
            if d_ratio in self.down_sampling_shape.keys():
                h,w = self.down_sampling_shape[d_ratio]
                assert h * w == seq,f'{h} * {w} != {seq}'

                return h,w

        #cal dict
        ori_d_ratio = d_ratio
        d_ratio //= 8
        new_h,new_w = h//8,w//8
        while d_ratio!=1:
            d_ratio //= 2
            new_h = (new_h+1)//2 if new_h%2 else new_h//2
            new_w = (new_w + 1) // 2 if new_w % 2 else new_w // 2
        assert new_w*new_h == seq
        self.down_sampling_shape[ori_d_ratio]=[new_h,new_w]
        return new_h,new_w

    def get_correlation_mask_cross(self, attn,attn_h,attn_w,local_focus_box=None):
        # correlate_maps = attn.sum(dim=-1).sum(dim=-1) #seq_len
        correlate_maps = attn.sum(dim=-1)[:,:self.assist_len].sum(dim=-1)  # seq_len
        # if local_focus_box is not None:
        #     correlate_maps *= local_focus_box
        if self.use_contrast:
            correlate_maps = self.contrast_operation(correlate_maps, self.contrast_beta, clamp=True, min_v=0, max_v=1,
                                                     dim=None,local_box=local_focus_box)
        correlate_maps = correlate_maps.reshape(attn_h, attn_w)
        return correlate_maps
    def get_correlation_mask(self,attn):
        assert self.obj_mask is not None,"input obj mask please!"
        mask = self.obj_mask
        h,seq_len,_ = attn.shape #head
        m_h,m_w = mask.shape
        d_ratio = 2**int(math.log2((m_h * m_w // seq_len) ** 0.5)+0.5)
        attn_h,attn_w = self.get_down_h_w(d_ratio,m_h,m_w,seq_len)
        mask = mask.unsqueeze(0).unsqueeze(0)
        downsampled_mask = F.interpolate(mask, size=(attn_h, attn_w), mode='nearest')
        mask_index = (downsampled_mask > 0.5).squeeze(0).squeeze(0).flatten()#obj mask -> down sampled obj mask
        correlate_maps = attn[:,mask_index]
        if self.use_contrast:
            correlate_maps = self.contrast_operation(correlate_maps, self.contrast_beta, clamp=True, min_v=0, max_v=1, dim=-1)
        correlate_maps = correlate_maps.reshape(h,-1,attn_h,attn_w).sum(dim=0).sum(dim=0) #sum of all heads and all pixels for each obj latent position
        return correlate_maps

    def reset(self):
        super(Mask_Expansion_SELF_ATTN, self).reset()
        #reset all the stat and store except for self.obj_mask
        #which is defined outside this class
        self.step_store = self.get_empty_store()
        self.expansion_mask_store = {}
        self.expansion_mask_on_the_fly = None
        self.expansion_mask_on_the_fly_assist = None
        self.log_mask = False
        self.step_num = 0
        self.model_type = 'Inverse'
        self.use_cfg = False
        self.share_attn = False
        self.local_edit = False
        self.DIFT_FEATURE_INJ = False
        self.bidirectional_loc = False
        self.last_step = False #ddim inv last step
        self.down_sampling_shape = dict()
        self.forbit_expand_area=None
        self.assist_len = None
        self.is_locating=False
        self.local_focus_box = None



    def head_to_batch_dim(self, tensor, out_dim=3):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)

        return tensor
    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def get_local_attention_scores(self, query, key, local_region_mask):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        #LOCAL CFG MODULATE QK.T
        #4,8,4224,77
        #[uncon_e,uncon_r,con_e,con_r]
        #["","","car",""]
        local_region_mask[local_region_mask>0] = 1
        _,L1,L2 = attention_scores.shape
        attention_scores = attention_scores.view(-1,self.heads,L1,L2)
        uncon_edit_qk = attention_scores[0].permute(1,2,0)#4227,77,heads
        con_edit_qk = attention_scores[2].permute(1,2,0)#4227,77,heads
        local_focus_qk = con_edit_qk * local_region_mask[:,None,None] + uncon_edit_qk * (1-local_region_mask[:,None,None])
        local_focus_qk = local_focus_qk.permute(2,0,1)
        attention_scores[2] = local_focus_qk
        attention_scores = attention_scores.view(-1,L1,L2)
        del uncon_edit_qk,con_edit_qk,local_focus_qk


        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs
    def cross_attention_mask_log(self, attention_probs, local_region_mask, place_in_unet, attn_h, attn_w, d_ratio, local_focus_box):
        # attention_probs=attention_scores.softmax(dim=-2) #different from CA need to softmax on image dim
        if local_focus_box is not None:
            local_focus_box[local_focus_box>0] = 1
        local_region_mask[local_region_mask > 0] = 1
        _, L1, L2 = attention_probs.shape
        attention_probs = attention_probs.view(-1, self.heads, L1, L2)
        assist_prompt_probs = attention_probs[1].permute(1, 2, 0).softmax(dim=0)  # 4227,77,heads
        key = f"{place_in_unet}_{'cross'}"
        self.step_store[key].append(self.get_correlation_mask_cross(assist_prompt_probs,attn_h,attn_w,local_focus_box))

    def get_local_attention_scores_with_log(self, query, key, local_region_mask, place_in_unet, attn_h, attn_w, d_ratio, local_focus_box=None):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()
        self.cross_attention_mask_log(attention_scores, local_region_mask, place_in_unet, attn_h, attn_w, d_ratio, local_focus_box)  # log key and Exp mask
        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores
        attention_probs = attention_probs.to(dtype)

        return attention_probs


    def process_mask_before_mutual(self,mask,seq,skip_eight=False):
        mask[mask>0] = 1
        # PROCESS MASK
        h, w = mask.shape
        d_ratio = 2**int(math.log2((h * w // seq) ** 0.5)+0.5)
        if skip_eight:
            d_ratio *=8
        attn_h, attn_w = self.get_down_h_w(d_ratio, h, w,seq)
        # down sample
        mask=F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(attn_h,attn_w),
                                 mode='nearest').squeeze(0, 1)
        return mask,d_ratio
    def post_process_attn_mask(self,mask):
        mask.masked_fill_(mask == 0, torch.finfo(mask.dtype).min)
        mask.masked_fill_(mask == 1, 0)
        mask= mask.repeat(self.heads, 1, 1)
        return mask


    def prepare_mutual_attention_mask(self,b,seq,value):
        edit_mask_ori,d_ratio = self.process_mask_before_mutual(self.obj_mask,seq)
        # SDSA_REF_MASK_STEP = self.SDSA_REF_MASK
        # if self.SDSA_REF_MASK_STEP is not None:
        #     SDSA_REF_MASK_STEP = self.SDSA_REF_MASK_STEP
        # ref_mask_ori,_ = self.process_mask_before_mutual(SDSA_REF_MASK_STEP,seq)
        ref_mask_ori,_ = self.process_mask_before_mutual(self.SDSA_REF_MASK,seq,skip_eight=True)
        edit_mask = edit_mask_ori.flatten()[None,]
        ref_mask = ref_mask_ori.flatten()
        mask = torch.ones((seq, seq), dtype=value.dtype,device=value.device)
        FG_mask = (mask*  ref_mask[None,])[None,]
        BG_mask = (mask* (1-ref_mask)[None,])[None,]
        mask = mask[None,]
        #post process -inf
        FG_mask = torch.cat((FG_mask,mask,FG_mask,mask))
        BG_mask = torch.cat((BG_mask,mask,BG_mask,mask))
        FG_mask = self.post_process_attn_mask(FG_mask)
        BG_mask = self.post_process_attn_mask(BG_mask)
        mask_ = torch.ones(edit_mask.shape, dtype=value.dtype,device=value.device)
        final_mask = torch.cat((edit_mask,mask_,edit_mask,mask_)).repeat(self.heads,  1)
        return FG_mask,BG_mask,final_mask,d_ratio
    def prepare_shared_self_attention_mask(self,b,seq,value):
        if self.SDSA_REF_MASK_STEP is None:
            with torch.no_grad():
                # SDSA_REF_MASK_STEP = F.dropout(self.SDSA_REF_MASK, p=self.drop_out)
                SDSA_REF_MASK_STEP = self.drop_block(self.SDSA_REF_MASK.unsqueeze(0).unsqueeze(0)).squeeze(0,1)
                SDSA_REF_MASK_STEP[SDSA_REF_MASK_STEP>0] = 1
        else:
            SDSA_REF_MASK_STEP = self.SDSA_REF_MASK_STEP
        # edit_mask = self.SDSA_EDIT_MASK
        half_seq = seq // 2
        h,w = SDSA_REF_MASK_STEP.shape
        d_ratio = 2**int(math.log2((h * w // half_seq) ** 0.5)+0.5)
        attn_h, attn_w = self.get_down_h_w(d_ratio, h, w,half_seq)
        #down sample
        ref_mask = F.interpolate(SDSA_REF_MASK_STEP.unsqueeze(0).unsqueeze(0), size=(attn_h,attn_w), mode='nearest').squeeze(0,1).flatten()

        mask = torch.zeros((b, seq, seq), dtype=value.dtype,device=value.device)
        mask[:, :half_seq, :half_seq] = 1
        mask[:, half_seq:, half_seq:] = 1
        mask[:, :half_seq, half_seq:] = ref_mask.repeat(b,half_seq,1)
        # epsilon = torch.finfo(value.dtype).eps
        # mask = torch.log(mask + epsilon)
        mask.masked_fill_(mask == 0, -1e9)
        mask.masked_fill_(mask == 1, 0)
        mask = mask.repeat(self.heads,1,1)
        del ref_mask
        return mask

    def share_attention_prepare(self,tensor):
        # 将形状(B, L, C)的张量concat为形状  (2, L*2, C)
        chunks = torch.chunk(tensor, 2, dim=0) #[uncon,con]
        return torch.cat(chunks, dim=1)

    def inverse_share_attention(self, tensor, batch_size, sequence_length):
        # 将形状 (2, L*2, C) 的张量还原为形状 (B, L, C)
        C = tensor.shape[2]
        split_tensor = torch.split(tensor, sequence_length, dim=1)
        return torch.cat(split_tensor, dim=0).reshape(batch_size, sequence_length, C)

    def subject_driven_self_attention(self,query,key,value,is_cross, place_in_unet):
        # TODO:1.share attention condition : SDSA,place_in_unet==UP (decoder)
        #      2. controller control , different stream
        #      3. concat
        #      4. prepare SDSA mask, with dropout for each step
        #      5. encoder & mid used for mask update on the fly

        # if self.SDSA_EDIT_MASK is None:
        #     self.SDSA_EDIT_MASK = self.obj_mask #dilated expansion target
        # else:
        #     pass
            # self.SDSA_EDIT_MASK = self.controller.expansion_mask_on_the_fly / self.controller.step_num nan avoid
            # “Otsu’s method”

        ref_mask = self.SDSA_REF_MASK
        ref_mask[ref_mask>0] = 1
        self.SDSA_REF_MASK = ref_mask

        batch_size, sequence_length, _ = (
            query.shape
        )

        query = self.share_attention_prepare(query)
        dim = query.shape[-1]
        key = self.share_attention_prepare(key)
        value = self.share_attention_prepare(value)

        attention_mask = self.prepare_shared_self_attention_mask(batch_size//2,sequence_length*2,value)

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        attention_probs = self.get_attention_scores(query, key, attention_mask)

        _ = self.mask_logging(attention_probs[:, :sequence_length, :sequence_length], is_cross, place_in_unet)#mask logging
        del _
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)
        hidden_states = self.inverse_share_attention(hidden_states, batch_size, sequence_length)
        return hidden_states
        # """
        # SLICE ATTENTION TO SAVE MEMORY
        # """
        # self.slice_size = self.heads // 2 #auto
        # batch_size_attention, query_tokens, _ = query.shape
        #
        # hidden_states = torch.zeros(
        #     (batch_size_attention, query_tokens, dim // self.heads), device=query.device, dtype=query.dtype
        # )
        # attention_map_log = torch.zeros(
        #     (batch_size_attention, query_tokens, query_tokens), device=query.device, dtype=query.dtype
        # )
        #
        # for i in range(batch_size_attention // self.slice_size):
        #     start_idx = i * self.slice_size
        #     end_idx = (i + 1) * self.slice_size
        #
        #     query_slice = query[start_idx:end_idx]
        #     key_slice = key[start_idx:end_idx]
        #     attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None
        #
        #     attn_slice = self.get_attention_scores(query_slice, key_slice, attn_mask_slice)
        #     attention_map_log[start_idx:end_idx] = attn_slice
        #
        #     attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])
        #     hidden_states[start_idx:end_idx] = attn_slice
        # _ = self.mask_logging(attention_map_log[:, :sequence_length, :sequence_length], is_cross, place_in_unet)#mask logging
        # del _,attention_map_log
        # hidden_states = self.batch_to_head_dim(hidden_states)
        # hidden_states = self.inverse_share_attention(hidden_states,batch_size,sequence_length)
        # return hidden_states
    def cross_manner_attention_modulate(self,q):
        #[uncon_edit,uncon_ref,con_edit,con_ref]
        _, qu_r, _, qc_r = q.chunk(4)
        return torch.cat((qu_r,qu_r,qc_r,qc_r))
    def mask_attention(self,query,key,value,attention_mask):
        attention_probs = self.get_attention_scores(query, key, attention_mask)
        return torch.bmm(attention_probs, value)

    def mutual_self_attention(self, query, key, value, is_cross, place_in_unet):
        batch_size, sequence_length, _ = (
            query.shape
        )
        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        if self.cur_att_layer//2 not in self.layer_idx:
            hidden_states = self.mask_attention(query,key,value,None)
            self.cur_att_layer += 1
            if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
                self.cur_att_layer = 0
                self.cur_step += 1
                self.between_steps()
            return self.batch_to_head_dim(hidden_states)


        key = self.cross_manner_attention_modulate(key) # Kr -> Ke ->:Replace
        value = self.cross_manner_attention_modulate(value) # Vr -> Ve
        #/16 do mutual self attn
        #prepare BG FG attn mask
        FG_mask,BG_mask,mask,d_ratio = self.prepare_mutual_attention_mask(batch_size,sequence_length,value)
        attn_out_fg = self.mask_attention(query,key,value,FG_mask)
        attn_out_bg = self.mask_attention(query,key,value,BG_mask)
        hidden_states = mask[:,:,None] * attn_out_fg + (1-mask)[:,:,None]* attn_out_bg
        hidden_states = self.batch_to_head_dim(hidden_states)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return hidden_states


    def loc_cross_attn_mask_expansion(self, query, key, value, is_cross, place_in_unet):
        #[uncon_assist,con_assist]
        #[normal ca , expand mask]
        batch_size, sequence_length, _ = (
            query.shape
        )
        local_region = self.obj_mask
        h, w = local_region.shape
        d_ratio = 2 ** int(math.log2((h * w // sequence_length) ** 0.5) + 0.5)
        attn_h, attn_w = self.get_down_h_w(d_ratio, h, w, sequence_length)
        # down sample
        local_region = F.interpolate(local_region.unsqueeze(0).unsqueeze(0), size=(attn_h, attn_w),
                                     mode='nearest').squeeze(0, 1).flatten()
        if self.is_locating:
            local_focus_box = F.interpolate(self.local_focus_box.unsqueeze(0).unsqueeze(0), size=(attn_h, attn_w),
                                            mode='nearest').squeeze(0, 1).flatten()
        else:
            local_focus_box = None

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)
        attention_probs = self.get_local_attention_scores_with_log(query, key, local_region, place_in_unet, attn_h, attn_w, d_ratio, local_focus_box)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)
        del attention_probs
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
            # self.between_steps_assist_hook()

        return hidden_states
    def modulate_local_cross_attn(self,query,key,value,is_cross, place_in_unet):
        batch_size, sequence_length, _ = (
            query.shape
        )
        local_region = self.local_edit_region
        h, w = local_region.shape
        d_ratio = 2**int(math.log2((h * w // sequence_length) ** 0.5)+0.5)
        attn_h, attn_w = self.get_down_h_w(d_ratio, h, w,sequence_length)
        # down sample
        local_region = F.interpolate(local_region.unsqueeze(0).unsqueeze(0), size=(attn_h,attn_w),
                                 mode='nearest').squeeze(0, 1).flatten()

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        attention_probs = self.get_local_attention_scores(query, key,local_region)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)
        del attention_probs
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

        return hidden_states

    def set_FI_allow(self,):
        self.DIFT_FEATURE_INJ= True

    def set_FI_forbid(self, ):
        self.DIFT_FEATURE_INJ = False
    def inter_mask(self,hw,mask,skip_eight=False):
        mask[mask>0] = 1
        h, w = mask.shape
        d_ratio = 2**int(math.log2((h * w // hw) ** 0.5)+0.5)
        if skip_eight: #down sample mask matched
            d_ratio *= 8
        attn_h, attn_w = self.get_down_h_w(d_ratio, h, w,hw)
        h, w = attn_h,attn_w
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bicubic').squeeze(0, 1)
        return mask,h,w

    def feature_injection(self, hidden_states, is_cross, place_in_unet, alpha=0.8):
        # start_time = time.time()
        # if self.cur_att_layer // 2 not in self.layer_idx:
        #     return hidden_states
        feature_map_layers = self.correspondence_map
        batch, hw, c = hidden_states.shape

        edit_mask, _, _ = self.inter_mask(hw, self.obj_mask)
        ref_mask, h, w = self.inter_mask(hw, self.SDSA_REF_MASK,True)
        try:
            H_list = {feature_map_layers[k].shape[0]:v for k,v in feature_map_layers.items()}
            feature_map = H_list[h]
        except:
            #in case some layers are not used for FI,cannot find matched resolution map in feature_map_layers
            return hidden_states

        chunks = hidden_states.chunk(2, dim=0)


        feature_map = feature_map.view(-1, 2)
        edit_mask = edit_mask.flatten()
        ref_mask = ref_mask.flatten()
        match_indices = feature_map[:, 0] * w + feature_map[:, 1]



        matched_obj_index= (match_indices > 0 ) & (edit_mask > 0.5)   #1.similarity thr 2. ori index in edit mask
        matched_obj_index_valid = ref_mask[match_indices[matched_obj_index]] > 0.5   # 3. target index in ref_mask
        matched_obj_index_final = matched_obj_index.clone()
        matched_obj_index_final[matched_obj_index] = matched_obj_index_valid #final index valid
        del feature_map,matched_obj_index_valid,matched_obj_index

        final_hidden = []
        for chunk in chunks:
            edit_hidden,ref_hidden = chunk
            blended_hidden = edit_hidden.clone()
            blended_hidden[matched_obj_index_final] = ref_hidden[match_indices[matched_obj_index_final]]
            blended_hidden = (1-alpha)*edit_hidden + alpha * blended_hidden
            final_hidden.append(torch.cat((blended_hidden[None,],ref_hidden[None,]),dim=0))
        final_hidden = torch.cat(final_hidden,dim=0)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"FI executed in: {elapsed_time:.6f} seconds")
        del blended_hidden,edit_hidden,ref_hidden,match_indices
        return final_hidden





    def __init__(self,block_size=8,drop_rate=0.5,layer_idx=None,start_layer=None):
        super(Mask_Expansion_SELF_ATTN, self).__init__()
        self.step_store = self.get_empty_store()
        self.expansion_mask_store={}
        self.expansion_mask_on_the_fly=None
        self.expansion_mask_on_the_fly_assist = None
        self.log_mask = False
        self.step_num = 0
        self.model_type='Inverse' #'Sample'
        self.use_cfg=False
        self.share_attn=False
        self.local_edit=False
        self.SDSA_EDIT_MASK = None #on the fly
        self.SDSA_REF_MASK = None
        self.SDSA_REF_MASK_STEP = None
        self.obj_mask =None
        self.drop_out = drop_rate
        self.drop_block = MaskDropBlock(block_size, drop_rate)
        self.DIFT_FEATURE_INJ = False
        self.correspondence_map = None
        self.layer_idx = list(range(start_layer, 16))
        self.bidirectional_loc = False
        self.last_step = False
        self.down_sampling_shape = dict()
        self.forbit_expand_area=None
        self.assist_len=None
        self.is_locating=False
        # MODEL_TYPE = {
        #     "SD": 16,
        #     "SDXL": 70
        # }


class SelfAttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(SelfAttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                pass
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 self_replace_steps: Union[float, Tuple[float, float]]):
        super(SelfAttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])

