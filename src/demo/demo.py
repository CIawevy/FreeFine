import gradio as gr
import numpy as np
from src.demo.utils import get_point, store_img, get_point_move, store_img_move, clear_points, upload_image_move, segment_with_points, segment_with_points_paste, fun_clear, paste_with_mask_and_offset,draw_inpaint_area



# MyExamples
examples_edit = [
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

examples_remove = [
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
examples_app = [
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
examples_compose  = [
    [
        "examples/move/001.png",
        'a photo of a cup',
    ],
    [
        "examples/move/002.png",
        'a photo of apples',
    ],
    [
        "examples/move/003.png",
        'a photo of a table',
    ],
    [
        "examples/move/004.png",
        'Astronauts play football on the moon',
    ],
    [
        "examples/move/005.png",
        'sun',
    ],
]


def create_demo_remove(runner=None):
    DESCRIPTION = """
    # Object Removal

    ## Usage:

    - Step1:Upload a source image, and then click to make a box or draw with a brush to generate the mask indicating the editing object.
    - Step2:Adjust the configuration (e.g. dilation factor, seed, prompt...)
    - Step3:Click the run button to get the results!
    - You can also refer to the examples bellow by just one click on any of them
"""

    with gr.Blocks() as demo:
        original_image = gr.State(value=None)
        img_with_mask = gr.State(value=None)

        selected_points = gr.State([])
        global_points = gr.State([])
        global_point_label = gr.State([])

        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    # mask 0
                    gr.Markdown("## Draw box for Mask")
                    original_image_1 = gr.Image(source='upload', label="Original image (Mask 1)", interactive=True,
                                                type="numpy")
                    # mask 1
                    gr.Markdown("## Option: Draw box for Mask 2")
                    original_image_2 = gr.Image(source='upload', label="Original (Mask 2)", interactive=True,
                                                type="numpy")

                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# Mask")

                    gr.Markdown("## Removal Mask 1")
                    mask_1 = gr.Image(source='upload', label="Removal Mask 1", interactive=True, type="numpy")
                    gr.Markdown("## Option: Removal Mask 2")
                    mask_2 = gr.Image(source='upload', label="Removal Mask 2", interactive=True, type="numpy")
                    gr.Markdown("## Option: Removal Mask 3")
                    mask_3 = gr.Image(source='upload', label="Removal Mask 3", interactive=True, type="numpy")

                    gr.Markdown("## Option: Refine Mask to avoid artifacts:")
                    refine_mask = gr.Image(source='upload', label="Refine Mask", interactive=True, type="numpy")

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery(columns=1, height='auto')

            original_image_1.select(
                segment_with_points,
                inputs=[original_image_1, original_image, global_points, global_point_label],
                outputs=[original_image_1, original_image, mask_1, global_points, global_point_label]
            )
            original_image_2.select(
                segment_with_points,
                inputs=[original_image_2, original_image, global_points, global_point_label],
                outputs=[original_image_2, original_image, mask_2, global_points, global_point_label]
            )
            original_image_3.edit(
                store_img_move,
                [original_image_3],
                [original_image, img_with_mask, mask_3]
            )
            original_image_4.edit(
                store_img_move,
                [original_image_4, refine_mask],
                [original_image, img_with_mask, refine_mask]
            )

        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")
            # gr.Examples(
            #     examples=examples_remove,
            #     inputs=[
            #         original_image_1, mask_1,
            #         original_image_2, mask_2,
            #         original_image_3, mask_3,
            #         original_image_4, refine_mask]
            # )
        run_button.click(fn=runner, inputs=[original_image, mask_1, mask_2, mask_3, refine_mask,
                                            original_image_1, original_image_2, original_image_3], outputs=[output])
        clear_button.click(
            fn=fun_clear,
            inputs=[original_image, img_with_mask, selected_points, global_points, global_point_label, original_image_1,
                    original_image_2, original_image_3, original_image_4, mask_1, mask_2, mask_3, refine_mask],
            outputs=[original_image, img_with_mask, selected_points, global_points, global_point_label,
                     original_image_1, original_image_2, original_image_3, original_image_4, mask_1, mask_2, mask_3,
                     refine_mask]
        )
    return demo

