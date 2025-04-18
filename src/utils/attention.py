import math
import torch.nn.functional as nnf
import numpy as np
import math
import matplotlib.pyplot as plt
import abc
import torch

import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
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
def register_attention_control_4bggen(model, controller):
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


            # TCA Controller
            controller.heads = self.heads
            controller.upcast_attention = self.upcast_attention
            controller.scale = self.scale
            controller.upcast_softmax=self.upcast_softmax

            if place_in_unet in ['up'] and not is_cross and controller.use_tca:
                hidden_states = controller.Temporal_contextal_attention_bg(query, key, value, is_cross, place_in_unet)
            elif controller.local_edit and is_cross:
                # if controller.method_type=='1':
                #     hidden_states = controller.modulate_local_cross_attn_bg_1(query,key,value,is_cross, place_in_unet)
                # else:
                hidden_states = controller.modulate_local_cross_attn_bg(query, key, value, is_cross,
                                                                              place_in_unet)
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


            # TCA Controller
            controller.heads = self.heads
            controller.upcast_attention = self.upcast_attention
            controller.scale = self.scale
            controller.upcast_softmax=self.upcast_softmax
            if not is_cross and controller.use_style_align and place_in_unet in controller.style_align_scope:
                hidden_states = controller.style_align_share_attention(query, key, value, is_cross, place_in_unet)
            elif not is_cross and controller.use_tca and place_in_unet in controller.tca_scope:
                hidden_states = controller.Temporal_contextal_attention(query, key, value, is_cross, place_in_unet)
            elif controller.local_edit and is_cross:
                hidden_states = controller.modulate_local_cross_attn(query,key,value,is_cross, place_in_unet)
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

def register_attention_control_compose(model, controller):
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


            # TCA Controller
            controller.heads = self.heads
            controller.upcast_attention = self.upcast_attention
            controller.scale = self.scale
            controller.upcast_softmax=self.upcast_softmax
            # if not is_cross and controller.use_style_align and place_in_unet in controller.style_align_scope:
            #     hidden_states = controller.style_align_share_attention(query, key, value, is_cross, place_in_unet)
            if not is_cross and controller.use_tca and place_in_unet in controller.tca_scope:
                hidden_states = controller.Temporal_contextal_attention_compose(query, key, value, is_cross, place_in_unet)
            elif controller.local_edit and is_cross:
                hidden_states = controller.modulate_local_cross_attn_compose(query,key,value,is_cross, place_in_unet)
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

class Attention_Modulator(AttentionControl):

    def __init__(self, start_layer=None):
        super(Attention_Modulator, self).__init__()
        # self.step_store = self.get_empty_store()
        self.step_num = 0
        self.model_type = 'Inverse'  # 'Sample'
        self.use_cfg = False
        self.use_style_align=False
        self.use_tca = False
        self.local_edit = False
        self.fg_retain_mask = None
        self.fg_ref_mask = None
        self.obj_mask = None
        self.local_edit_region = None
        if start_layer is not None:
            self.layer_idx = list(range(start_layer, 16)) #for mmsa , start layer = 10 for SD-V15
        else:
            self.layer_idx = list(range(16))
        self.down_sampling_shape = dict()
        self.method = None
        self.context_guidance = None
        self.tca_scope = ['up']
        self.style_align_scope = ['down','mid','up']
        self.src_masks = None
        self.tgt_masks = None


    # @staticmethod
    # def get_empty_store():
    #     return {"down_cross": [], "mid_cross": [], "up_cross": [],
    #             "down_self": [],  "mid_self": [],  "up_self": []}
    #     # return {"down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if is_cross:
            pass
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

    def between_steps(self):
        pass

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



    def reset(self):
        super(Attention_Modulator, self).reset()
        #reset all the stat and store except for self.obj_mask
        #which is defined outside this class
        # self.step_store = self.get_empty_store()
        # self.log_mask = False
        self.step_num = 0
        self.model_type = 'Inverse'
        self.use_cfg = False
        self.use_tca = False
        self.use_style_align=False
        self.local_edit = False
        self.down_sampling_shape = dict()
        self.style_align = False
        self.method = None
        self.context_guidance = None
        self.tca_scope = ['up']
        self.style_align_scope = ['down', 'mid', 'up']
     


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

    def get_cross_hidden_state(self, query, key, value, local_region_mask):
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

        hidden_states = torch.bmm(attention_scores.softmax(dim=-1).to(dtype), value)

        return hidden_states



    def process_mask_before_attention(self, mask, seq, skip_eight=False):
        if mask.max() > 1:
            m_dtype = mask.dtype
            mask = (mask / mask.max()).to(m_dtype)

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

    def prepare_various_attention_mask(self, b, seq, value):
        fg_retain_mask, d_ratio = self.process_mask_before_attention(self.fg_retain_mask, seq,skip_eight=False)

        ref_mask_ori, _ = self.process_mask_before_attention(self.fg_ref_mask , seq, skip_eight=False)

        fg_retain_mask = fg_retain_mask.flatten()[None,]


        ref_mask = ref_mask_ori.flatten()
        mask = torch.ones((seq, seq), dtype=value.dtype, device=value.device)
        FG_mask = (mask * ref_mask[None,])[None,]
        BG_mask = (mask * (1 - ref_mask)[None,])[None,]
        mask = mask[None,]
        # post process -inf
        FG_mask = torch.cat((FG_mask, mask, FG_mask, mask))
        BG_mask = torch.cat((BG_mask, mask, BG_mask, mask))
        FG_mask = self.post_process_attn_mask(FG_mask)
        BG_mask = self.post_process_attn_mask(BG_mask)
        mask_ = torch.ones(fg_retain_mask.shape, dtype=value.dtype, device=value.device)
        final_mask_fg = torch.cat((fg_retain_mask, mask_, fg_retain_mask, mask_)).repeat(self.heads, 1)
        if self.method == 'tca':
            fg_retain_mask_st2, _ = self.process_mask_before_attention(self.fg_retain_mask_st2, seq, skip_eight=False)
            fg_retain_mask_st2 = fg_retain_mask_st2.flatten()[None,]
            final_mask_fg_st2 = torch.cat((fg_retain_mask_st2, mask_, fg_retain_mask_st2, mask_)).repeat(self.heads, 1)
        else:
            final_mask_fg_st2 = None

        return FG_mask, BG_mask, final_mask_fg, d_ratio,final_mask_fg_st2

    def prepare_various_attention_mask_compose(self, src_mask,tgt_mask, seq, value):
        fg_retain_mask, d_ratio = self.process_mask_before_attention(tgt_mask, seq, skip_eight=False)

        ref_mask_ori, _ = self.process_mask_before_attention(src_mask, seq, skip_eight=False)

        fg_retain_mask = fg_retain_mask.flatten()[None,]

        ref_mask = ref_mask_ori.flatten()
        mask = torch.ones((seq, seq), dtype=value.dtype, device=value.device)
        FG_mask = (mask * ref_mask[None,])[None,]
        # post process -inf
        # FG_mask = torch.cat((FG_mask, mask, FG_mask, mask))
        attention_mask = self.post_process_attn_mask(FG_mask)
        feature_mask = (fg_retain_mask).repeat(self.heads, 1)

        return attention_mask,feature_mask
    def prepare_attention_mask_for_bggen(self, b, seq, value):
        # fg_retain_mask==obj_Mask==background inpainting area
        fg_retain_mask, d_ratio = self.process_mask_before_attention(self.fg_retain_mask, seq, skip_eight=False)
        fg_retain_mask = fg_retain_mask.flatten()[None,]

        mask = torch.ones((seq, seq), dtype=value.dtype, device=value.device)
        FG_mask = (mask * fg_retain_mask)[None,]
        BG_mask = (mask * (1 - fg_retain_mask))[None,]
        mask = mask[None,]
        # post process -inf
        FG_mask = torch.cat((FG_mask,mask, FG_mask,mask ))
        BG_mask = torch.cat((BG_mask,mask, BG_mask,mask))
        FG_mask = self.post_process_attn_mask(FG_mask)
        BG_mask = self.post_process_attn_mask(BG_mask)
        mask_ = torch.ones(fg_retain_mask.shape, dtype=value.dtype, device=value.device)
        final_mask_fg = torch.cat((fg_retain_mask,mask_, fg_retain_mask,mask_)).repeat(self.heads, 1)


        return FG_mask, BG_mask, final_mask_fg, d_ratio
    def prepare_sdsa_mask_for_bggen(self, seq, value):
        # fg_retain_mask==obj_Mask==background inpainting area
        fg_retain_mask, d_ratio = self.process_mask_before_attention(self.fg_retain_mask, seq, skip_eight=False)
        fg_retain_mask = fg_retain_mask.flatten()[None,]
        fg_retain_mask = torch.cat([torch.ones_like(fg_retain_mask), fg_retain_mask], dim=-1)

        mask = torch.ones((seq, seq * 2), dtype=value.dtype, device=value.device)
        BG_mask = (mask * (1 - fg_retain_mask))[None,]
        mask = mask[None,]
        # post process -inf
        BG_mask = torch.cat((BG_mask, mask, BG_mask, mask))
        attn_mask = self.post_process_attn_mask(BG_mask)

        return attn_mask
    def prepare_sdsa_mask(self, seq, value):
        fg_ref_mask, d_ratio = self.process_mask_before_attention(self.fg_ref_mask, seq, skip_eight=False)
        fg_ref_mask = fg_ref_mask.flatten()[None,]
        fg_ref_mask = torch.cat([torch.ones_like(fg_ref_mask),fg_ref_mask],dim=-1)

        mask = torch.ones((seq, seq*2), dtype=value.dtype, device=value.device)
        FG_mask = (mask * fg_ref_mask)[None,]
        mask = mask[None,]
        # post process -inf
        FG_mask = torch.cat((FG_mask,mask, FG_mask,mask ))
        FG_mask = self.post_process_attn_mask(FG_mask)
        return FG_mask

  
    def prepare_attention_mask_for_bggen_1(self, b, seq, value):
        fg_retain_mask, d_ratio = self.process_mask_before_attention(self.fg_retain_mask, seq, skip_eight=False)
        fg_retain_mask = fg_retain_mask.flatten()[None,]

        mask = torch.ones((seq, seq), dtype=value.dtype, device=value.device)
        FG_mask = (mask * fg_retain_mask)[None,]
        BG_mask = (mask * (1 - fg_retain_mask))[None,]
        mask = mask[None,]
        # post process -inf
        FG_mask = torch.cat((FG_mask, FG_mask, ))
        BG_mask = torch.cat((BG_mask, BG_mask,))
        FG_mask = self.post_process_attn_mask(FG_mask)
        BG_mask = self.post_process_attn_mask(BG_mask)
        mask_ = torch.ones(fg_retain_mask.shape, dtype=value.dtype, device=value.device)
        final_mask_fg = torch.cat((fg_retain_mask, fg_retain_mask)).repeat(self.heads, 1)


        return FG_mask, BG_mask, final_mask_fg, d_ratio
    def prepare_mutual_attention_mask(self,b,seq,value):
        edit_mask_ori,d_ratio = self.process_mask_before_attention(self.obj_mask, seq)
        # SDSA_REF_MASK_STEP = self.SDSA_REF_MASK
        # if self.SDSA_REF_MASK_STEP is not None:
        #     SDSA_REF_MASK_STEP = self.SDSA_REF_MASK_STEP
        # ref_mask_ori,_ = self.process_mask_before_mutual(SDSA_REF_MASK_STEP,seq)
        ref_mask_ori,_ = self.process_mask_before_attention(self.SDSA_REF_MASK, seq, skip_eight=True)
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

    def share_attention_prepare(self,tensor):
        # 将形状(B, L, C)的张量concat为形状  (2, L*2, C)
        chunks = torch.chunk(tensor, 2, dim=0) #[uncon,con]
        return torch.cat(chunks, dim=1)

    def inverse_share_attention(self, tensor, batch_size, sequence_length):
        # 将形状 (2, L*2, C) 的张量还原为形状 (B, L, C)
        C = tensor.shape[2]
        split_tensor = torch.split(tensor, sequence_length, dim=1)
        return torch.cat(split_tensor, dim=0).reshape(batch_size, sequence_length, C)
    def seperate_tokens(self,query):
        # total_length = query.shape[0] // (self.heads * 2)
        # qu_e, qu_r, qc_e, qc_r = query[:self.heads], query[self.heads:self.heads * total_length], \
        #     query[self.heads * total_length:self.heads * (total_length + 1)], query[self.heads * (total_length + 1):]
        # return qu_e, qu_r, qc_e, qc_r
        total_length = (query.shape[0] // self.heads)-1
        qu_e, qu_r, qc_e = query[:self.heads], query[self.heads:self.heads * total_length], \
        query[self.heads * total_length:self.heads * (total_length + 1)],
        return qu_e, qu_r, qc_e
    def seperate_tokens_compose_cross(self,query):
        # total_length = query.shape[0] // (self.heads * 2)
        # qu_e, qu_r, qc_e, qc_r = query[:self.heads], query[self.heads:self.heads * total_length], \
        #     query[self.heads * total_length:self.heads * (total_length + 1)], query[self.heads * (total_length + 1):]
        # return qu_e, qu_r, qc_e, qc_r
        total_length = (query.shape[0] // self.heads)-self.prompt_length
        qu, qc_e = query[:self.heads * total_length], \
        query[self.heads * total_length:self.heads * (total_length + self.prompt_length)],
        return qu,  qc_e

    def seperate_tokens_compose_cross_query(self, query):
        # total_length = query.shape[0] // (self.heads * 2)
        # qu_e, qu_r, qc_e, qc_r = query[:self.heads], query[self.heads:self.heads * total_length], \
        #     query[self.heads * total_length:self.heads * (total_length + 1)], query[self.heads * (total_length + 1):]
        # return qu_e, qu_r, qc_e, qc_r
        total_length = (query.shape[0] // self.heads) - 1
        qu, qc_e = query[:self.heads * total_length], \
            query[self.heads * total_length:self.heads * (total_length + 1)],
        return qu, qc_e

    def cross_manner_attention_modulate(self, q):
        _, qu_r, _, qc_r = q.chunk(4)
        return torch.cat((qu_r,qu_r,qc_r,qc_r))



    def mask_attention(self,query,key,value,attention_mask):
        attention_probs = self.get_attention_scores(query, key, attention_mask)
        return torch.bmm(attention_probs, value)

    def Temporal_contextal_attention(self, query, key, value, is_cross, place_in_unet):
        batch_size, sequence_length, _ = (
            query.shape
        )
        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        if self.cur_att_layer // 2 not in self.layer_idx:
            hidden_states = self.mask_attention(query, key, value, None)
            self.cur_att_layer += 1
            if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
                self.cur_att_layer = 0
                self.cur_step += 1
                self.between_steps()
            return self.batch_to_head_dim(hidden_states)

        cross_key = self.cross_manner_attention_modulate(key)  # Kr -> Ke ->:Replace
        cross_value = self.cross_manner_attention_modulate(value)  # Vr -> Ve
        # /16 do mutual self attn
        # prepare BG FG attn mask
        src_FG_mask, src_BG_mask, tgt_FG_mask, d_ratio, tgt_FG_mask_st2 = self.prepare_various_attention_mask(
            batch_size, sequence_length, value)
        attn_out_fg = self.mask_attention(query, cross_key, cross_value, src_FG_mask)
        attn_out_bg = self.mask_attention(query, cross_key, cross_value, src_BG_mask)
        if self.method == 'mmsa':
            hidden_states = tgt_FG_mask[:, :, None] * attn_out_fg + (1 - tgt_FG_mask)[:, :, None] * attn_out_bg
        elif self.method == 'tca':
            tgt_FG_mask[tgt_FG_mask > 0] = 1
            # if self.sep_region:
            #     ref_hidden_states_non_completion = tgt_FG_mask_st2[:, :, None] * attn_out_fg + (1 - tgt_FG_mask)[:, :,
            #                                                                                    None] * attn_out_bg
            #     self_hidden_states = self.mask_attention(query, key, value, None)
            #     completion_region = (tgt_FG_mask - tgt_FG_mask_st2)[:, :, None]
            #     hidden_states_non_completion = (1 - self.context_guidance) * (
            #                 1 - completion_region) * self_hidden_states + self.context_guidance * ref_hidden_states_non_completion
            #     hidden_states = hidden_states_non_completion + self_hidden_states * completion_region
            # else: #sep region is not necessary, we come to the code bellow
            ref_hidden = tgt_FG_mask[:, :, None] * attn_out_fg + (1 - tgt_FG_mask)[:, :, None] * attn_out_bg
            self_hidden = self.mask_attention(query, key, value, None)
            hidden_states = ref_hidden * (self.context_guidance) + self_hidden * (1 - self.context_guidance)

        hidden_states = self.batch_to_head_dim(hidden_states)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return hidden_states
    def Temporal_contextal_attention_compose(self, query, key, value, is_cross, place_in_unet):
        batch_size, sequence_length, _ = (
            query.shape
        )
        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        if self.cur_att_layer // 2 not in self.layer_idx:
            hidden_states = self.mask_attention(query, key, value, None)
            self.cur_att_layer += 1
            if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
                self.cur_att_layer = 0
                self.cur_step += 1
                self.between_steps()
            return self.batch_to_head_dim(hidden_states)

        total_length = (query.shape[0] // self.heads )-1
        self_hidden = self.mask_attention(query, key, value, None)
        hu_e,hu_r,hc_e, = self.seperate_tokens(self_hidden)
        qu_e, _, qc_e, = self.seperate_tokens(query)
        _, ku_r, _,  = self.seperate_tokens(key)  # Kr -> Ke ->:Replace
        _, vu_r, _,   = self.seperate_tokens(value)  # Vr -> Ve
        # /16 do mutual self attn
        # prepare BG FG attn mask

        hu_e_new, hc_e_new = torch.zeros_like(hu_e), torch.zeros_like(hc_e)
        ku_r = ku_r.chunk(total_length-1)
        vu_r = vu_r.chunk(total_length-1)
        for i in range(total_length-1):
            attn_mask,feat_mask = self.prepare_various_attention_mask_compose(
                self.src_masks[i],self.tgt_masks[i], sequence_length, value)
            feat_mask = feat_mask[:,:,None]
            hu_e_new +=  feat_mask * self.mask_attention(qu_e, ku_r[i], vu_r[i], attn_mask)
            hc_e_new +=  feat_mask * self.mask_attention(qc_e, ku_r[i], vu_r[i], attn_mask)
        if self.method == 'mmsa':
            hu_e,hc_e = hu_e_new,hc_e_new
        elif self.method == 'tca':
            hu_e = hu_e_new * (self.context_guidance) + hu_e* (1 - self.context_guidance)
            hc_e = hc_e_new * (self.context_guidance) + hc_e * (1 - self.context_guidance)

        hidden_states = torch.cat([hu_e,hu_r,hc_e])
        hidden_states = self.batch_to_head_dim(hidden_states)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return hidden_states

    def style_align_share_attention(self, query, key, value, is_cross, place_in_unet):
        def concat_first(feat, dim=2, scale=1.):
            feat_style = expand_first(feat, scale=scale)
            return torch.cat((feat, feat_style), dim=dim)

        def expand_first(feat, scale=1., ):  # MODIFIED FOR OUR TASK
            b = feat.shape[0]
            # feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
            # [uncon_edit,uncon_ref,con_edit,con_ref] total is 2
            assert scale == 1., 'not implemente error'
            feat_style = torch.stack((feat[1], feat[b // 2 + 1])).unsqueeze(1)
            if scale == 1:
                feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
            else:
                feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
                feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
            return feat_style.reshape(*feat.shape)

        batch_size, sequence_length, _ = (
            query.shape
        )

        key = concat_first(key, -2, scale=1)
        value = concat_first(value, -2)



        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)


        if self.method == 'sdsa':
            attn_mask = self.prepare_sdsa_mask(sequence_length, value)
        else:
            attn_mask=None

        # key: [uncon_edit*8,uncon_ref*8,con_edit*8,con_ref*8]

        hidden_states = nnf.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask , dropout_p=0.0, is_causal=False
        )

        hidden_states = self.batch_to_head_dim(hidden_states)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return hidden_states
    def style_align_share_attention_bg(self, query, key, value, is_cross, place_in_unet):
        def concat_first(feat, dim=2, scale=1.):
            feat_style = expand_first(feat, scale=scale)
            return torch.cat((feat, feat_style), dim=dim)

        def expand_first(feat, scale=1., ):  # MODIFIED FOR OUR TASK
            b = feat.shape[0]
            # feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
            # [uncon_edit,uncon_ref,con_edit,con_ref] total is 2
            assert scale == 1., 'not implemente error'
            feat_style = torch.stack((feat[1], feat[b // 2 + 1])).unsqueeze(1)
            if scale == 1:
                feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
            else:
                feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
                feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
            return feat_style.reshape(*feat.shape)

        batch_size, sequence_length, _ = (
            query.shape
        )

        key = concat_first(key, -2, scale=1)
        value = concat_first(value, -2)

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)
        if self.method == 'sdsa':
            attn_mask = self.prepare_sdsa_mask_for_bggen(sequence_length, value)
        else:
            attn_mask=None
        # key: [uncon_edit*8,uncon_ref*8,con_edit*8,con_ref*8]

        hidden_states = nnf.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = self.batch_to_head_dim(hidden_states)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return hidden_states


    def calc_mean_std_2d(self,feat, eps=1e-5, mask=None):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 2)
        C = size[0]
        if mask is not None:
            feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
            feat_std = feat_var.sqrt().view(C, 1)
            feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1)
        else:
            feat_var = feat.view(C, -1).var(dim=1) + eps
            feat_std = feat_var.sqrt().view(C, 1)
            feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1)

        return feat_mean, feat_std

    def adain(self,content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)
        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def calc_mean_std(self,feat, eps=1e-5, mask=None):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        if len(size) == 2:
            return self.calc_mean_std_2d(feat, eps, mask)

        assert (len(size) == 3)
        C = size[0]
        if mask is not None:
            feat_var = feat.view(C, -1)[:, mask.view(-1) == 1].var(dim=1) + eps
            feat_std = feat_var.sqrt().view(C, 1, 1)
            feat_mean = feat.view(C, -1)[:, mask.view(-1) == 1].mean(dim=1).view(C, 1, 1)
        else:
            feat_var = feat.view(C, -1).var(dim=1) + eps
            feat_std = feat_var.sqrt().view(C, 1, 1)
            feat_mean = feat.view(C, -1).mean(dim=1).view(C, 1, 1)

        return feat_mean, feat_std

    def Temporal_contextal_attention_bg(self, query, key, value, is_cross, place_in_unet):
        batch_size, sequence_length, _ = (
            query.shape
        )

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        if self.cur_att_layer // 2 not in self.layer_idx:
            hidden_states = self.mask_attention(query, key, value, None)
            self.cur_att_layer += 1
            if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
                self.cur_att_layer = 0
                self.cur_step += 1
                self.between_steps()
            return self.batch_to_head_dim(hidden_states)

        cross_key = self.cross_manner_attention_modulate(key)  # Kr -> Ke ->:Replace
        cross_value = self.cross_manner_attention_modulate(value)  # Vr -> Ve
        # /16 do mutual self attn
        # prepare BG FG attn mask
        src_FG_mask, src_BG_mask, tgt_FG_mask, d_ratio= self.prepare_attention_mask_for_bggen(batch_size, sequence_length, value)
        # attn_out_fg = self.mask_attention(query, key, value, src_FG_mask)

        attn_out_bg = self.mask_attention(query, cross_key, cross_value, src_BG_mask)

        if self.method=='mmsa':
            hidden_states = attn_out_bg
        elif self.method =='tca':
            tgt_FG_mask[tgt_FG_mask>0] = 1
            self_hidden_states = self.mask_attention(query, key, value, None)
            hidden_states = self_hidden_states*(1-self.context_guidance) + attn_out_bg*self.context_guidance

        hidden_states = self.batch_to_head_dim(hidden_states)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return hidden_states

    def modulate_local_cross_attn_bg(self,query,key,value,is_cross, place_in_unet):
        batch_size, sequence_length, _ = (
            query.shape
        )
        local_region = self.local_edit_region
        h, w = local_region.shape
        d_ratio = 2**int(math.log2((h * w // sequence_length) ** 0.5)+0.5)
        attn_h, attn_w = self.get_down_h_w(d_ratio, h, w,sequence_length)
        # down sample
        local_region = F.interpolate(local_region.unsqueeze(0).unsqueeze(0), size=(attn_h,attn_w),
                                 mode='nearest').squeeze(0, 1)
        local_region  = local_region.flatten()
        # local_region[local_region > 0] = 1
        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        hidden_states = self.get_cross_hidden_state(query, key, value, local_region)

        hidden_states = self.batch_to_head_dim(hidden_states)
        _, L1, L2 = hidden_states.shape #2,4096,320 [uncon_edit,con_edit]
        uncon_edit,uncon_ref,con_edit,_ = hidden_states
        modified_con_edit = local_region[:,None]*con_edit+(1-local_region)[:,None]*uncon_edit
        hidden_states = torch.stack([uncon_edit,uncon_ref,modified_con_edit,uncon_ref],dim=0)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

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
                                 mode='nearest').squeeze(0, 1)
        local_region  = local_region.flatten()
        # local_region[local_region > 0] = 1
        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        hidden_states = self.get_cross_hidden_state(query, key, value, local_region)

        hidden_states = self.batch_to_head_dim(hidden_states)
        _, L1, L2 = hidden_states.shape #4,4096,320 [uncon_edit,uncon_ref,con_edit,con_ref]
        uncon_edit,uncon_ref,con_edit,_ = hidden_states
        modified_con_edit = local_region[:,None]*con_edit+(1-local_region)[:,None]*uncon_edit
        hidden_states = torch.stack([uncon_edit,uncon_ref,modified_con_edit,uncon_ref],dim=0)



        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

        return hidden_states
    def modulate_local_cross_attn_compose(self,query,key,value,is_cross, place_in_unet):
        batch_size, sequence_length, _ = (
            query.shape
        )
        local_cross_attn_masks = []
        for local_region in self.tgt_masks:
            # local_region = self.local_edit_region
            h, w = local_region.shape
            d_ratio = 2**int(math.log2((h * w // sequence_length) ** 0.5)+0.5)
            attn_h, attn_w = self.get_down_h_w(d_ratio, h, w,sequence_length)
            # down sample
            local_region = F.interpolate(local_region.unsqueeze(0).unsqueeze(0), size=(attn_h,attn_w),
                                     mode='nearest').squeeze(0, 1)
            local_region  = local_region.flatten()
            local_cross_attn_masks.append(local_region)
        # local_region[local_region > 0] = 1
        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        qu,qc_e = self.seperate_tokens_compose_cross_query(query)
        ku, kc_e = self.seperate_tokens_compose_cross(key)
        vu, vc_e = self.seperate_tokens_compose_cross(value)
        hu = self.get_cross_hidden_state(qu, ku, vu,None)
        kc_e = kc_e.chunk(len(local_cross_attn_masks))
        vc_e = vc_e.chunk(len(local_cross_attn_masks))
        hc_e = torch.zeros_like(qc_e)
        for i,mask in enumerate(local_cross_attn_masks):
            hc_e += mask[:,None]*self.get_cross_hidden_state(qc_e, kc_e[i], vc_e[i],None)
        hidden_states = torch.cat([hu,hc_e])
        hidden_states = self.batch_to_head_dim(hidden_states)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

        return hidden_states
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



