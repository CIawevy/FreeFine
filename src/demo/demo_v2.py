import gradio as gr
import numpy as np
from src.demo.utils import get_point, store_img, get_point_move, store_img_move, clear_points, upload_image_move, segment_with_points, segment_with_points_paste, fun_clear, paste_with_mask_and_offset,draw_inpaint_area



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
                            value=15,
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
                            value=0.2,
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
                        strong_inpaint = gr.Slider(
                            label="Strong inpaint area 0:False 1:True",
                            minimum=1,
                            maximum=1,
                            step=1,
                            value=1,
                            interactive=False)
                        cross_enhance = gr.Slider(
                            label="Cross_enhance 0:False 1:True",
                            minimum=0,
                            maximum=1,
                            step=1,
                            value=0,
                            interactive=False)
                        standard_drawing = gr.Slider(
                            label="select the box draw or casual draw to upload mask 0:casual draw 1:standard box draw",
                            minimum=0,
                            maximum=1,
                            step=1,
                            value=1,
                            interactive=False)
                        blending_alpha = gr.Slider(
                            label="alpha_blending value for blending edited regions and original images",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=1.0,
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
                            splatting_radius = gr.Slider(
                                label="splatting_radius",
                                minimum=0,
                                maximum=1,
                                step=0.001,
                                value=0.015,
                                interactive=True)
                            splatting_tau = gr.Slider(
                                label="splatting_tau",
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.0,
                                interactive=True)
                            splatting_points_per_pixel = gr.Slider(
                                label="splatting_radius",
                                minimum=100,
                                maximum=1,
                                step=2,
                                value=30,
                                interactive=True)
                            focal_length = gr.Slider(
                                label="focal_length",
                                minimum=0,
                                maximum=10000,
                                step=1,
                                value=340,
                                interactive=True)
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
                            sx = gr.Slider(
                                label="sx:x axis scaleing factor",
                                minimum=0,
                                maximum=10,
                                step=0.1,
                                value=1,
                                interactive=True)
                            sy = gr.Slider(
                                label="sy:y axis scaleing factor",
                                minimum=0,
                                maximum=10,
                                step=0.1,
                                value=1,
                                interactive=True)
                            sz = gr.Slider(
                                label="sz:z axis scaleing factor",
                                minimum=0,
                                maximum=10,
                                step=0.1,
                                value=1,
                                interactive=True)
                            rx = gr.Slider(
                                label="rx:x axis rotation angle",
                                minimum=-180,
                                maximum=180,
                                step=10,
                                value=0,
                                interactive=True)
                            ry = gr.Slider(
                                label="ry:y axis rotation angle",
                                minimum=-180,
                                maximum=180,
                                step=10,
                                value=0,
                                interactive=True)
                            rz = gr.Slider(
                                label="rz:z axis rotation angle",
                                minimum=-180,
                                maximum=180,
                                step=10,
                                value=0,
                                interactive=True)
                            tx = gr.Slider(
                                label="tx:x axis translation factor [-1,1]",
                                minimum=-1,
                                maximum=1,
                                step=0.1,
                                value=0,
                                interactive=True)
                            ty = gr.Slider(
                                label="ty:y axis translation factor [-1,1]",
                                minimum=-1,
                                maximum=1,
                                step=0.1,
                                value=0,
                                interactive=True)
                            tz = gr.Slider(
                                label="tz:z axis translation factor [-1,1]",
                                minimum=-1,
                                maximum=1,
                                step=0.1,
                                value=0,
                                interactive=True)
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
                         inputs=[img_draw_box, img_ref, prompt, INP_prompt, seed, guidance_scale, num_step,
                                 max_resolution, mode, dilate_kernel_size,
                                 start_step, tx, ty, tz, rx, ry, rz, sx, sy, sz, mask_ref, eta, use_mask_expansion,
                                 standard_drawing, contrast_beta, strong_inpaint, cross_enhance,
                                 mask_threshold, mask_threshold_target, blending_alpha, splatting_radius, splatting_tau,
                                 splatting_points_per_pixel, focal_length,end_step,feature_injection,FI_range,sim_thr,DIFT_LAYER_IDX,use_sdsa ],
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
                            value=15,
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
                            value=0.2,
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
                        strong_inpaint = gr.Slider(
                            label="Strong inpaint area 0:False 1:True",
                            minimum=1,
                            maximum=1,
                            step=1,
                            value=1,
                            interactive=False)
                        cross_enhance = gr.Slider(
                            label="Cross_enhance 0:False 1:True",
                            minimum=0,
                            maximum=1,
                            step=1,
                            value=0,
                            interactive=False)
                        standard_drawing = gr.Slider(
                            label="select the box draw or casual draw to upload mask 0:casual draw 1:standard box draw",
                            minimum=0,
                            maximum=1,
                            step=1,
                            value=1,
                            interactive=False)
                        blending_alpha = gr.Slider(
                            label="alpha_blending value for blending edited regions and original images",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=1.0,
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
                            splatting_radius = gr.Slider(
                                label="splatting_radius",
                                minimum=0,
                                maximum=1,
                                step=0.001,
                                value=0.015,
                                interactive=True)
                            splatting_tau = gr.Slider(
                                label="splatting_tau",
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.0,
                                interactive=True)
                            splatting_points_per_pixel = gr.Slider(
                                label="splatting_radius",
                                minimum=100,
                                maximum=1,
                                step=2,
                                value=30,
                                interactive=True)
                            focal_length = gr.Slider(
                                label="focal_length",
                                minimum=0,
                                maximum=10000,
                                step=1,
                                value=340,
                                interactive=True)
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
                            sx = gr.Slider(
                                label="sx:x axis scaleing factor",
                                minimum=0,
                                maximum=10,
                                step=0.1,
                                value=1,
                                interactive=True)
                            sy = gr.Slider(
                                label="sy:y axis scaleing factor",
                                minimum=0,
                                maximum=10,
                                step=0.1,
                                value=1,
                                interactive=True)
                            sz = gr.Slider(
                                label="sz:z axis scaleing factor",
                                minimum=0,
                                maximum=10,
                                step=0.1,
                                value=1,
                                interactive=True)
                            rx = gr.Slider(
                                label="rx:x axis rotation angle",
                                minimum=-180,
                                maximum=180,
                                step=10,
                                value=0,
                                interactive=True)
                            ry = gr.Slider(
                                label="ry:y axis rotation angle",
                                minimum=-180,
                                maximum=180,
                                step=10,
                                value=0,
                                interactive=True)
                            rz = gr.Slider(
                                label="rz:z axis rotation angle",
                                minimum=-180,
                                maximum=180,
                                step=10,
                                value=0,
                                interactive=True)
                            tx = gr.Slider(
                                label="tx:x axis translation factor [-1,1]",
                                minimum=-1,
                                maximum=1,
                                step=0.1,
                                value=0,
                                interactive=True)
                            ty = gr.Slider(
                                label="ty:y axis translation factor [-1,1]",
                                minimum=-1,
                                maximum=1,
                                step=0.1,
                                value=0,
                                interactive=True)
                            tz = gr.Slider(
                                label="tz:z axis translation factor [-1,1]",
                                minimum=-1,
                                maximum=1,
                                step=0.1,
                                value=0,
                                interactive=True)
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
                         inputs=[img_draw_box, img_ref, prompt, INP_prompt, seed, guidance_scale, num_step,
                                 max_resolution, mode, dilate_kernel_size,
                                 start_step, tx, ty, tz, rx, ry, rz, sx, sy, sz, mask_ref, eta, use_mask_expansion,
                                 standard_drawing, contrast_beta, strong_inpaint, cross_enhance,
                                 mask_threshold, mask_threshold_target, blending_alpha, splatting_radius, splatting_tau,
                                 splatting_points_per_pixel, focal_length,end_step,feature_injection,FI_range,sim_thr,DIFT_LAYER_IDX,use_sdsa,mask ],
                         outputs=[output_edit, refer_edit,INP_IMG, INP_Mask, TGT_MSK,])
        clear_button.click(fn=fun_clear,
                           inputs=[original_image, mask, prompt, INP_prompt, ],
                           outputs=[original_image, mask, prompt, INP_prompt, ])
    return demo