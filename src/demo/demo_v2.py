import gradio as gr
import numpy as np
from src.demo.utils import get_point, store_img, get_point_move, store_img_move, clear_points, upload_image_move, segment_with_points, segment_with_points_paste, fun_clear, paste_with_mask_and_offset,draw_inpaint_area
from lora.lora_utils import  train_lora
def train_lora_interface(original_image,
                         prompt,
                         # model_path,
                         # vae_path,
                         lora_path,
                         lora_step,
                         lora_lr,
                         lora_batch_size,
                         lora_rank,
                         progress=gr.Progress()):
    train_lora(
        original_image,
        prompt,
        # model_path,
        # vae_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_batch_size,
        lora_rank,
        progress)
    return "Training LoRA Done!"
# MyExamples
examples_CPIG_FULL_3D = [
    [
        "examples/move/001.png",
        "examples/move/001.png",
        'a photo of a cup',
        'empty scene',
    ],
    [
        "examples/move/002.png",
        "examples/move/002.png",
        'a photo of apples',
        'empty scene',
    ],
    [
        "examples/move/003.png",
        "examples/move/003.png",
        'a photo of a table',
        'empty scene',
    ],
    [
        "examples/move/004.png",
        "examples/move/004.png",
        'Astronauts play football on the moon',
        'empty scene',
    ],
    [
        "examples/move/005.png",
        "examples/move/005.png",
        'sun',
        'empty scene',
    ],
    ["examples/appearance/004_base.jpg",
     "examples/appearance/004_base.jpg",
     'car',
     'empty scene',],
    [
        "examples/drag/003.png",
        "examples/drag/003.png",
        'oil painting',
        'empty scene',],
]
# MyExamples
# MyExamples
examples_CPIG_FULL_2D = [
    [
        "examples/move/001.png",
        "examples/move/001.png",
        "examples/move/001.png",
        'a photo of a cup',
        'empty scene',
    ],
    [
        "examples/move/002.png",
        "examples/move/002.png",
        "examples/move/002.png",
        'a photo of apples',
        'empty scene',
    ],
    [
        "examples/move/003.png",
        "examples/move/003.png",
        "examples/move/003.png",
        'a photo of a table',
        'empty scene',
    ],
    [
        "examples/move/004.png",
        "examples/move/004.png",
        "examples/move/004.png",
        'Astronauts play football on the moon',
        'empty scene',
    ],
    [
        "examples/move/005.png",
        "examples/move/005.png",
        "examples/move/005.png",
        'sun',
        'empty scene',
    ],
    ["examples/appearance/004_base.jpg",
     "examples/appearance/004_base.jpg",
     "examples/appearance/004_base.jpg",
     'car',
     'empty scene',],
    [
        "examples/drag/003.png",
        "examples/drag/003.png",
        "examples/drag/003.png",
        'oil painting',
        'empty scene',],
]
# MyExamples
examples_CPIG = [
    [
        "examples/move/001.png",
        "examples/move/001.png",
        'a photo of a cup',
    ],
    [
        "examples/move/002.png",
        "examples/move/002.png",
        'a photo of apples',
    ],
    [
        "examples/move/003.png",
        "examples/move/003.png",
        'a photo of a table',
    ],
    [
        "examples/move/004.png",
        "examples/move/004.png",
        'Astronauts play football on the moon',
    ],
    [
        "examples/move/005.png",
        "examples/move/005.png",
        'sun',
    ],
    ["examples/appearance/004_base.jpg",
     "examples/appearance/004_base.jpg",
     'car'],
    [
        "examples/drag/003.png",
        "examples/drag/003.png",
        'oil painting',],
]



def create_my_demo_full_SV3D_magic(runner):
    DESCRIPTION = """
    ## Baseline Demo with simple copy-paste and inpainting
    Usage:
    - Upload a source image, and then draw a box to generate the mask corresponding to the editing object(optional).
    - Label the object's movement path on the source image.
    - Label reference region. (optional)
    - Add a text description to the image and click the `Edit` button to start editing."""

    with gr.Blocks() as demo:
        original_image = gr.State(value=None)  # store original image
        mask_ref = gr.State(value=None)
        selected_points = gr.State([])
        global_points = gr.State([])
        global_point_label = gr.State([])
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    gr.Markdown("## 1. Draw box to mask target object(original)")
                    img_draw_box = gr.Image(source='upload', label="Draw box", interactive=True, type="numpy")

                    gr.Markdown("## 2. 1. Draw box to mask target object(transformed)")
                    img_ref = gr.Image(source='upload', label="Draw box", interactive=True, type="numpy")

                    gr.Markdown("## 3. Prompt")
                    prompt = gr.Textbox(label="Prompt")

                    gr.Markdown("## 3.Assist Prompt")
                    assist_prompt = gr.Textbox(label="Prompt")

                    gr.Markdown("## 4.Inpaint Prompt")
                    INP_prompt = gr.Textbox(label="INP_Prompt")

                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

                    with gr.Box():
                        guidance_scale = gr.Slider(
                            label="Classifier-free guidance strength",
                            minimum=0,
                            maximum=100,
                            step=0.2,
                            value=7.5,
                            interactive=True)
                        eta = gr.Slider(
                            label="eta setting in DDIM denoising process 0:DDIM 1:DDPM",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=1.0,
                            interactive=True)
                        num_step = gr.Slider(
                            label="number of diffusion steps",
                            minimum=0,
                            maximum=1000,
                            step=1,
                            value=50,
                            interactive=True)
                        start_step = gr.Slider(
                            label="number of start step of num_step",
                            minimum=0,
                            maximum=1000,
                            step=1,
                            value=25,
                            interactive=True)
                        end_step = gr.Slider(
                            label="number of start step of end_step",
                            minimum=0,
                            maximum=1000,
                            step=1,
                            value=0,
                            interactive=True)
                        mask_threshold = gr.Slider(
                            label=" mask_threshold",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.5,
                            interactive=True)
                        mode = gr.Slider(
                            label=" inpainting mode selection 1:laMa 2:sd-inpaint",
                            minimum=1,
                            maximum=2,
                            step=1,
                            value=1,
                            interactive=True)
                        use_mask_expansion = gr.Slider(
                            label="use mask expansion module to contain more semantic areas",
                            minimum=0,
                            maximum=1,
                            step=1,
                            value=0,
                            interactive=True)

                        max_resolution = gr.Slider(label="Resolution", value=512, minimum=428, maximum=1024, step=1)
                        dilate_kernel_size = gr.Slider(
                            label="dilate_kernel_size for inpainting mask dilation",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=30,
                            interactive=True)
                        contrast_beta = gr.Slider(
                            label="contrast_beta for contrast operation in attention map store >1:focus <1:sparse",
                            minimum=0.1,
                            maximum=100,
                            step=1,
                            value=1.67,
                            interactive=True)
                        with gr.Accordion('Advanced options', open=False):
                            seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                            # 模拟用户输入
                            # tx, ty, tz = 0, 0, 0  # 相对平移量 定义在三维坐标系上
                            # rx, ry, rz = 0, -40, 0  # 旋转角度（度数）
                            # sx, sy, sz = 1, 1, 1  # 缩放比例 >1为缩小
                            # splatting_radius = 0.015, splatting_tau = 0.0, splatting_points_per_pixel = 30,

                            feature_injection=gr.Slider(
                                label="feature_injection",
                                minimum=0,
                                maximum=1,
                                step=1,
                                value=1,
                                interactive=True)

                            sim_thr = gr.Slider(
                                label="sim_thr",
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.5,
                                interactive=True)
                            use_sdsa = gr.Slider(
                                label="use_sdsa",
                                minimum=0,
                                maximum=1,
                                step=1,
                                value=1,
                                interactive=True)
                            FI_range = gr.Slider(
                                label="FI_range",
                                value=(682, 640),
                                interactive=False)
                            DIFT_LAYER_IDX=gr.Slider(
                                label="DIFT_LAYER_IDX",
                                value=[0,1,2,3],
                                interactive=False)

                            mask_threshold_target = gr.Slider(
                                label=" mask_threshold_target",
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.5,
                                interactive=True)

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    mask = gr.Image(source='upload', label="Mask of object(original)", interactive=True, type="numpy")
                    trans_mask = gr.Image(source='upload', label="Mask of object(trans)", interactive=True, type="numpy")

                    gr.Markdown("<h5><center>EditResults</center></h5>")
                    output_edit = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>EditRefer</center></h5>")
                    refer_edit = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>referenceIMG</center></h5>")
                    INP_IMG = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>InpaintingMask</center></h5>")
                    INP_Mask = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>TargetMask</center></h5>")
                    TGT_MSK = gr.Gallery().style(grid=1, height='auto')



                    # im_w_mask_ref = gr.Image(label="Mask of inpaint region", interactive=True, type="numpy")

            img_draw_box.select(
                segment_with_points,
                inputs=[img_draw_box, original_image, global_points, global_point_label],
                outputs=[img_draw_box, original_image, mask_ref, global_points, global_point_label, mask]
            )

        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(
                examples=examples_CPIG_FULL_3D,
                inputs=[img_draw_box, img_ref, prompt, INP_prompt]
            )


        run_button.click(fn=runner,
                         inputs=[original_image, img_ref, prompt, INP_prompt, seed, guidance_scale, num_step,
                                 max_resolution, mode, dilate_kernel_size,
                                 start_step, eta, use_mask_expansion,
                                contrast_beta, mask_threshold, mask_threshold_target,end_step,feature_injection,FI_range,sim_thr,DIFT_LAYER_IDX,use_sdsa,mask_ref,assist_prompt ],
                         outputs=[output_edit, refer_edit,INP_IMG, INP_Mask, TGT_MSK])
        clear_button.click(fn=fun_clear,
                           inputs=[original_image, mask, prompt, INP_prompt, ],
                           outputs=[original_image, mask, prompt, INP_prompt, ])
    return demo

def create_my_demo_full_SV3D_multi_obj_case(runner):
    DESCRIPTION = """
    ## Baseline Demo with simple copy-paste and inpainting
    Usage:
    - Upload a source image, and then draw a box to generate the mask corresponding to the editing object(optional).
    - Label the object's movement path on the source image.
    - Label reference region. (optional)
    - Add a text description to the image and click the `Edit` button to start editing."""

    with gr.Blocks() as demo:
        original_image = gr.State(value=None)  # store original image
        mask_ref = gr.State(value=[])
        selected_points = gr.State([])
        global_points = gr.State([])
        global_point_label = gr.State([])
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    gr.Markdown("## 1. Draw box to mask target object(original)")
                    img_draw_box = gr.Image(source='upload', label="Draw box", interactive=True, type="numpy")

                    gr.Markdown("## 2. 1. Draw box to mask target object(transformed)")
                    img_ref = gr.Image(source='upload', label="Draw box", interactive=True, type="numpy")

                    gr.Markdown("## 3. Prompt")
                    prompt = gr.Textbox(label="Prompt")

                    gr.Markdown("## 4.Inpaint Prompt")
                    INP_prompt = gr.Textbox(label="INP_Prompt")

                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

                    with gr.Box():
                        guidance_scale = gr.Slider(
                            label="Classifier-free guidance strength",
                            minimum=0,
                            maximum=100,
                            step=0.2,
                            value=7.5,
                            interactive=True)
                        eta = gr.Slider(
                            label="eta setting in DDIM denoising process 0:DDIM 1:DDPM",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0,
                            interactive=True)
                        num_step = gr.Slider(
                            label="number of diffusion steps",
                            minimum=0,
                            maximum=1000,
                            step=1,
                            value=50,
                            interactive=True)
                        start_step = gr.Slider(
                            label="number of start step of num_step",
                            minimum=0,
                            maximum=1000,
                            step=1,
                            value=25,
                            interactive=True)
                        end_step = gr.Slider(
                            label="number of start step of end_step",
                            minimum=0,
                            maximum=1000,
                            step=1,
                            value=0,
                            interactive=True)
                        mask_threshold = gr.Slider(
                            label=" mask_threshold",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.5,
                            interactive=True)
                        mode = gr.Slider(
                            label=" inpainting mode selection 1:laMa 2:sd-inpaint",
                            minimum=1,
                            maximum=2,
                            step=1,
                            value=2,
                            interactive=True)
                        use_mask_expansion = gr.Slider(
                            label="use mask expansion module to contain more semantic areas",
                            minimum=0,
                            maximum=1,
                            step=1,
                            value=1,
                            interactive=True)

                        max_resolution = gr.Slider(label="Resolution", value=512, minimum=428, maximum=1024, step=1)
                        dilate_kernel_size = gr.Slider(
                            label="dilate_kernel_size for inpainting mask dilation",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=30,
                            interactive=True)
                        contrast_beta = gr.Slider(
                            label="contrast_beta for contrast operation in attention map store >1:focus <1:sparse",
                            minimum=0.1,
                            maximum=100,
                            step=1,
                            value=1.67,
                            interactive=True)
                        with gr.Accordion('Advanced options', open=False):
                            seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                            # 模拟用户输入
                            # tx, ty, tz = 0, 0, 0  # 相对平移量 定义在三维坐标系上
                            # rx, ry, rz = 0, -40, 0  # 旋转角度（度数）
                            # sx, sy, sz = 1, 1, 1  # 缩放比例 >1为缩小
                            # splatting_radius = 0.015, splatting_tau = 0.0, splatting_points_per_pixel = 30,

                            feature_injection=gr.Slider(
                                label="feature_injection",
                                minimum=0,
                                maximum=1,
                                step=1,
                                value=1,
                                interactive=True)

                            sim_thr = gr.Slider(
                                label="sim_thr",
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.5,
                                interactive=True)
                            use_sdsa = gr.Slider(
                                label="use_sdsa",
                                minimum=0,
                                maximum=1,
                                step=1,
                                value=1,
                                interactive=True)
                            FI_range = gr.Slider(
                                label="FI_range",
                                value=(682, 640),
                                interactive=False)
                            DIFT_LAYER_IDX=gr.Slider(
                                label="DIFT_LAYER_IDX",
                                value=[0,1,2,3],
                                interactive=False)

                            mask_threshold_target = gr.Slider(
                                label=" mask_threshold_target",
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.5,
                                interactive=True)

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    mask = gr.Image(source='upload', label="Mask of object(original)", interactive=True, type="numpy")

                    gr.Markdown("<h5><center>EditResults</center></h5>")
                    output_edit = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>EditRefer</center></h5>")
                    refer_edit = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>referenceIMG</center></h5>")
                    INP_IMG = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>InpaintingMask</center></h5>")
                    INP_Mask = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>TargetMask</center></h5>")
                    TGT_MSK = gr.Gallery().style(grid=1, height='auto')

                    # im_w_mask_ref = gr.Image(label="Mask of inpaint region", interactive=True, type="numpy")

            img_draw_box.select(
                segment_with_points,
                inputs=[img_draw_box, original_image, global_points, global_point_label],
                outputs=[img_draw_box, original_image, mask, global_points, global_point_label, ]
            )
            # img_ref.edit(
            #     draw_inpaint_area,
            #     [img_ref],
            #     [original_image, im_w_mask_ref, mask_ref]
            # )
            # img_ref.select(
            #     segment_with_points,
            #     inputs=[img_ref, original_image, global_points, global_point_label],
            #     outputs=[img_ref, original_image, trans_mask, global_points, global_point_label, ]
            # )
        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(
                examples=examples_CPIG_FULL_3D,
                inputs=[img_draw_box, img_ref, prompt, INP_prompt]
            )


        run_button.click(fn=runner,
                         inputs=[original_image, img_ref, prompt, INP_prompt, seed, guidance_scale, num_step,
                                 max_resolution, mode, dilate_kernel_size,
                                 start_step, eta, use_mask_expansion,
                                contrast_beta, mask_threshold, mask_threshold_target,end_step,feature_injection,FI_range,sim_thr,DIFT_LAYER_IDX,use_sdsa,mask ],
                         outputs=[output_edit, refer_edit,INP_IMG, INP_Mask, TGT_MSK])
        clear_button.click(fn=fun_clear,
                           inputs=[original_image, mask, prompt, INP_prompt, ],
                           outputs=[original_image, mask, prompt, INP_prompt, ])
    return demo

def create_my_demo_full_2D_magic(runner):
    DESCRIPTION = """
    ## Baseline Demo with simple copy-paste and inpainting & continuous editing
    Usage:
    - Upload a source image, and then draw a box to generate the mask corresponding to the editing object(optional).
    - Label the object's movement path on the source image.
    - Label reference region. (optional)
    - Add a text description to the image and click the `Edit` button to start editing."""

    with gr.Blocks() as demo:
        original_image = gr.State(value=None)  # store original image
        mask_ref = gr.State(value=None)
        selected_points = gr.State([])
        global_points = gr.State([])
        global_point_label = gr.State([])
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    gr.Markdown("## 1. Draw box to mask target object(optional)")
                    img_draw_box = gr.Image(source='upload', label="Draw box", interactive=True, type="numpy")

                    gr.Markdown("## 2. Draw arrow to describe the movement")
                    img_ref = gr.Image(source='upload', label="Original image", interactive=True, type="numpy")

                    gr.Markdown("## 3. Prompt")
                    prompt = gr.Textbox(label="Prompt")

                    gr.Markdown("## 3.Assist Prompt")
                    assist_prompt = gr.Textbox(value='shadow',label="Assist-Prompt")

                    gr.Markdown("## 4.Inpaint Prompt")
                    INP_prompt = gr.Textbox(value='a photo of a background, a photo of an empty place',label="INP_Prompt")
                    gr.Markdown("## 5.Lora Path")
                    lora_path = gr.Textbox(value="./lora_tmp", label="LoRA path")
                    # general parameters
                    # with gr.Row():
                    #     lora_path = gr.Textbox(value="./lora_tmp", label="LoRA path")
                    #     lora_status_bar = gr.Textbox(label="display LoRA training status")
                    #     train_lora_button = gr.Button(value='Train LoRA', scale=0.3)


                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

                    # with gr.Tab("LoRA Parameters"):
                    #     with gr.Row():
                    #         lora_step = gr.Number(value=60, label="LoRA training steps", precision=0)
                    #         lora_lr = gr.Number(value=0.0005, label="LoRA learning rate")
                    #         lora_batch_size = gr.Number(value=4, label="LoRA batch size", precision=0)
                    #         lora_rank = gr.Number(value=16, label="LoRA rank", precision=0)

                    with gr.Box():
                        guidance_scale = gr.Slider(
                            label="Classifier-free guidance strength",
                            minimum=0,
                            maximum=100,
                            step=0.1,
                            value=7.5,
                            interactive=True)
                        eta= gr.Slider(
                            label="eta setting in DDIM denoising process 0:DDIM 1:DDPM",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=1.0,
                            interactive=True)
                        num_step = gr.Slider(
                            label="number of diffusion steps",
                            minimum=0,
                            maximum=1000,
                            step=1,
                            value=50,
                            interactive=True)
                        start_step = gr.Slider(
                            label="number of start step of num_step",
                            minimum=0,
                            maximum=1000,
                            step=1,
                            value=25,
                            interactive=True)
                        exp_step = gr.Slider(
                            label="Mask expansion step",
                            minimum=0,
                            maximum=1000,
                            step=1,
                            value=4,
                            interactive=True)
                        end_step = gr.Slider(
                            label="h-feature end injection steps",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0,
                            interactive=True)
                        use_mask_expansion = gr.Slider(
                            label="use mask expansion module to contain more semantic areas",
                            minimum=0,
                            maximum=1,
                            step=1,
                            value=1,
                            interactive=True)
                        mode = gr.Slider(
                            label=" inpainting mode selection 1:laMa 2:sd-inpaint",
                            minimum=1,
                            maximum=2,
                            step=1,
                            value=2,
                            interactive=True)

                        max_resolution = gr.Slider(label="Resolution", value=512, minimum=428, maximum=1024, step=1)
                        contrast_beta = gr.Slider(
                            label="contrast_beta for contrast operation in attention map store >1:focus <1:sparse",
                            minimum=0.1,
                            maximum=100,
                            step=1,
                            value=1.67,
                            interactive=True)
                        with gr.Accordion('Advanced options', open=False):
                            seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                            resize_scale = gr.Slider(
                                label="Object resizing scale",
                                minimum=0,
                                maximum=10,
                                step=0.1,
                                value=1,
                                interactive=True)
                            rotation_angle = gr.Slider(
                                label="Object clock-wise rotation angle [-180,180]",
                                minimum=-180,
                                maximum=180,
                                step=10,
                                value=0,
                                interactive=True)
                            flip_horizontal= gr.Slider(
                                label="flip_horizontal",
                                minimum=0,
                                maximum=1,
                                step=1,
                                value=0,
                                interactive=True)
                            flip_vertical = gr.Slider(
                                label="flip_horizontal",
                                minimum=0,
                                maximum=1,
                                step=1,
                                value=0,
                                interactive=True)
                            feature_injection = gr.Slider(
                                label="feature_injection",
                                minimum=0,
                                maximum=1,
                                step=1,
                                value=1,
                                interactive=True)
                            sim_thr = gr.Slider(
                                label="sim_thr",
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.5,
                                interactive=True)
                            use_mtsa = gr.Slider(
                                label="use_mtsa",
                                minimum=0,
                                maximum=1,
                                step=1,
                                value=1,
                                interactive=True)
                            FI_range = gr.Slider(
                                label="FI_range",
                                value=(682, 640),
                                interactive=False)
                            DIFT_LAYER_IDX = gr.Slider(
                                label="DIFT_LAYER_IDX",
                                value=[0, 1, 2, 3],
                                interactive=False)

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    mask = gr.Image(source='upload', label="Mask of object(original)", interactive=True, type="numpy")

                    gr.Markdown("<h5><center>EditResults</center></h5>")
                    output_edit = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>EditRefer</center></h5>")
                    refer_edit = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>referenceIMG</center></h5>")
                    INP_IMG = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>InpaintingMask</center></h5>")
                    INP_Mask = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>TargetMask</center></h5>")
                    TGT_MSK = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>inpaint BG</center></h5>")
                    inp_bg = gr.Gallery().style(grid=1, height='auto')

                img_draw_box.select(
                    segment_with_points,
                    inputs=[img_draw_box, original_image, global_points, global_point_label],
                    outputs=[img_draw_box, original_image, mask_ref, global_points, global_point_label, mask]
                )
                img_ref.select(
                    get_point_move,
                    [original_image, img_ref, selected_points],
                    [img_ref, original_image, selected_points, ],
                )
                # train_lora_button.click(
                #     train_lora_interface,
                #     [original_image,
                #      prompt,
                #      # model_path,
                #      # vae_path,
                #      lora_path,
                #      lora_step,
                #      lora_lr,
                #      lora_batch_size,
                #      lora_rank],
                #     [lora_status_bar]
                # )

        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(
                examples=examples_CPIG_FULL_3D,
                inputs=[img_draw_box, img_ref, prompt, INP_prompt]
            )

        run_button.click(fn=runner,
                         inputs=[ original_image,prompt, INP_prompt,selected_points,seed, guidance_scale, num_step, max_resolution, mode, start_step, resize_scale,
                                  rotation_angle, flip_horizontal,flip_vertical,eta, use_mask_expansion,exp_step,contrast_beta, end_step,
                                feature_injection, FI_range,sim_thr, DIFT_LAYER_IDX, use_mtsa,mask_ref,assist_prompt,lora_path],
                         outputs=[output_edit, refer_edit,INP_IMG, INP_Mask, TGT_MSK,inp_bg])
        clear_button.click(fn=fun_clear,
                           inputs=[original_image, mask, prompt, INP_prompt, ],
                           outputs=[original_image, mask, prompt, INP_prompt, ])
    return demo
