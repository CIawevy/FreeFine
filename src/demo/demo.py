import gradio as gr
import numpy as np
from src.demo.utils import get_point, store_img, get_point_move, store_img_move, clear_points, upload_image_move, segment_with_points, segment_with_points_paste, fun_clear, paste_with_mask_and_offset,draw_inpaint_area



# MyExamples
examples_CPIG_FULL = [
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
examples_move = [
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
examples_appearance = [
    [
        "examples/appearance/001_base.png",
        "examples/appearance/001_replace.png",
        'a photo of a cake',
        'a photo of a cake',
    ],
    [
        "examples/appearance/002_base.png",
        "examples/appearance/002_replace.png",
        'a photo of a doughnut',
        'a photo of a doughnut',
    ],
    [
        "examples/appearance/003_base.jpg",
        "examples/appearance/003_replace.png",
        'a photo of a Swiss roll',
        'a photo of a Swiss roll',
    ],
    [
        "examples/appearance/004_base.jpg",
        "examples/appearance/004_replace.jpeg",
        'a photo of a car',
        'a photo of a car',
    ],
    [
        "examples/appearance/005_base.jpeg",
        "examples/appearance/005_replace.jpg",
        'a photo of an ice-cream',
        'a photo of an ice-cream',
    ],
]
examples_drag = [
    [
        "examples/drag/001.png",
        'a photo of a mountain',
    ],
    [
        "examples/drag/003.png",
        'oil painting',
    ],
    [
        "examples/drag/004.png",
        'oil painting',
    ],
    [
        "examples/drag/005.png",
        'a dog',
    ],
    [
        "examples/drag/006.png",
        'a cat',
    ],
]
examples_face = [
    [
        "examples/face/001_base.png",
        "examples/face/001_reference.png",
    ],
    [
        "examples/face/002_base.png",
        "examples/face/002_reference.png",
    ],
    [
        "examples/face/003_base.png",
        "examples/face/003_reference.png",
    ],
    [
        "examples/face/004_base.png",
        "examples/face/004_reference.png",
    ],
    [
        "examples/face/005_base.png",
        "examples/face/005_reference.png",
    ],
]
examples_paste = [
    [
        "examples/move/001.png",
        "examples/paste/001_replace.png",
        'a photo of a croissant on the table',
        'a photo of a croissant',
        0,
        0,
        1
    ],
    [
        "examples/paste/002_base.png",
        "examples/paste/002_replace.png",
        'a red car on road',
        'a red car',
        0,
        110,
        1
    ],
    [
        "examples/paste/003_replace.jpg",
        "examples/paste/003_base.jpg",
        'a photo of a doughnut on the table',
        'a photo of a doughnut',
        0,
        0,
        1
    ],
    [
        "examples/paste/004_base.png",
        "examples/paste/004_replace.png",
        'a burger in a plate',
        'a burger',
        0,
        -150,
        0.8
    ],
    [
        "examples/paste/005_base.png",
        "examples/paste/005_replace.png",
        'a red apple on the hand',
        'a red apple',
        0,
        -140,
        0.8
    ],
]

def create_demo_move(runner):
    DESCRIPTION = """
    ## Object Moving & Resizing
    Usage:
    - Upload a source image, and then draw a box to generate the mask corresponding to the editing object.
    - Label the object's movement path on the source image.
    - Label reference region. (optional)
    - Add a text description to the image and click the `Edit` button to start editing."""
    
    with gr.Blocks() as demo:
        original_image = gr.State(value=None) # store original image
        mask_ref = gr.State(value=None)
        selected_points = gr.State([])
        global_points = gr.State([])
        global_point_label = gr.State([])
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    gr.Markdown("## 1. Draw box to mask target object")
                    img_draw_box = gr.Image(source='upload', label="Draw box", interactive=True, type="numpy")

                    gr.Markdown("## 2. Draw arrow to describe the movement")
                    img = gr.Image(source='upload', label="Original image", interactive=True, type="numpy")

                    gr.Markdown("## 3. Label reference region (Optional)")
                    img_ref = gr.Image(tool="sketch", label="Original image", interactive=True, type="numpy") 
        
                    gr.Markdown("## 4. Prompt")
                    prompt = gr.Textbox(label="Prompt")

                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

                    with gr.Box():
                        guidance_scale = gr.Slider(label="Classifier-free guidance strength", value=4, minimum=1, maximum=10, step=0.1)
                        energy_scale = gr.Slider(label="Classifier guidance strength (x1e3)", value=0.5, minimum=0, maximum=10, step=0.1)
                        max_resolution = gr.Slider(label="Resolution", value=768, minimum=428, maximum=1024, step=1)
                        with gr.Accordion('Advanced options', open=False):
                            seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                            resize_scale = gr.Slider(
                                        label="Object resizing scale",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=1,
                                        interactive=True)
                            w_edit = gr.Slider(
                                        label="Weight of moving strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=4,
                                        interactive=True)
                            w_content = gr.Slider(
                                        label="Weight of content consistency strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=6,
                                        interactive=True)
                            w_contrast = gr.Slider(
                                        label="Weight of contrast strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=0.2,
                                        interactive=True)
                            w_inpaint = gr.Slider(
                                        label="Weight of inpainting strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=0.8,
                                        interactive=True)
                            SDE_strength = gr.Slider(
                                        label="Flexibility strength",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.4,
                                        interactive=True)
                            ip_scale = gr.Slider(
                                        label="Image prompt scale",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.1,
                                        interactive=True)
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    mask = gr.Image(source='upload', label="Mask of object", interactive=True, type="numpy")
                    im_w_mask_ref = gr.Image(label="Mask of reference region", interactive=True, type="numpy")
                    
                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery().style(grid=1, height='auto')     

            img.select(
                get_point_move,
                [original_image, img, selected_points],
                [img, original_image, selected_points],
            )
            img_draw_box.select(
                segment_with_points, 
                inputs=[img_draw_box, original_image, global_points, global_point_label, img], 
                outputs=[img_draw_box, original_image, mask, global_points, global_point_label, img, img_ref]
            )
            img_ref.edit(
                store_img_move,
                [img_ref],
                [original_image, im_w_mask_ref, mask_ref]
            )
        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(
                examples=examples_move,
                inputs=[img_draw_box, prompt]
            )
                    
        run_button.click(fn=runner, inputs=[original_image, mask, mask_ref, prompt, resize_scale, w_edit, w_content, w_contrast, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale], outputs=[output])
        clear_button.click(fn=fun_clear, inputs=[original_image, global_points, global_point_label, selected_points, mask_ref, mask, img_draw_box, img, im_w_mask_ref], outputs=[original_image, global_points, global_point_label, selected_points, mask_ref, mask, img_draw_box, img, im_w_mask_ref])
    return demo

def create_demo_appearance(runner):
    DESCRIPTION = """
    ## Appearance Modulation
    Usage:
    - Upload a source image, and an appearance reference image.
    - Label object masks on these two image.
    - Add a text description to the image and click the `Edit` button to start editing."""

    with gr.Blocks() as demo:
        original_image_base = gr.State(value=None)
        global_points_base = gr.State([])
        global_point_label_base = gr.State([])
        global_points_replace = gr.State([])
        global_point_label_replace = gr.State([])
        original_image_replace = gr.State(value=None)
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    gr.Markdown("## 1. Upload image & Draw box to generate mask")
                    img_base = gr.Image(source='upload', label="Original image", interactive=True, type="numpy")
                    img_replace = gr.Image(tool="upload", label="Reference image", interactive=True, type="numpy")

                    gr.Markdown("## 2. Prompt")
                    prompt = gr.Textbox(label="Prompt")
                    prompt_replace = gr.Textbox(label="Prompt of reference image")

                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

                    with gr.Box():
                        guidance_scale = gr.Slider(label="Classifier-free guidance strength", value=5, minimum=1, maximum=10, step=0.1)
                        energy_scale = gr.Slider(label="Classifier guidance strength (x1e3)", value=2, minimum=0, maximum=10, step=0.1)
                        max_resolution = gr.Slider(label="Resolution", value=768, minimum=428, maximum=1024, step=1)
                        with gr.Accordion('Advanced options', open=False):
                            seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                            w_edit = gr.Slider(
                                        label="Weight of moving strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=3.5,
                                        interactive=True)
                            w_content = gr.Slider(
                                        label="Weight of content consistency strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=5,
                                        interactive=True)
                            SDE_strength = gr.Slider(
                                        label="Flexibility strength",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.4,
                                        interactive=True)
                            ip_scale = gr.Slider(
                                        label="Image prompt scale",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.1,
                                        interactive=True)

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    with gr.Row():
                        mask_base = gr.Image(source='upload', label="Mask of editing object", interactive=False, type="numpy")
                        mask_replace = gr.Image(tool="upload", label="Mask of reference object", interactive=False, type="numpy")
                    
                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery().style(grid=1, height='auto')

        img_base.select(
            segment_with_points, 
            inputs=[img_base, original_image_base, global_points_base, global_point_label_base], 
            outputs=[img_base, original_image_base, mask_base, global_points_base, global_point_label_base]
        )
        img_replace.select(
            segment_with_points, 
            inputs=[img_replace, original_image_replace, global_points_replace, global_point_label_replace], 
            outputs=[img_replace, original_image_replace, mask_replace, global_points_replace, global_point_label_replace]
        )
        with gr.Column():     
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(
                examples=examples_appearance,
                inputs=[img_base, img_replace, prompt, prompt_replace]
            )
        clear_button.click(fn=fun_clear, inputs=[original_image_base, original_image_replace, global_points_base, global_points_replace, global_point_label_base, global_point_label_replace, img_base, img_replace, mask_base, mask_replace], outputs=[original_image_base, original_image_replace, global_points_base, global_points_replace, global_point_label_base, global_point_label_replace, img_base, img_replace, mask_base, mask_replace])
        run_button.click(fn=runner, inputs=[original_image_base, mask_base, original_image_replace, mask_replace, prompt, prompt_replace, w_edit, w_content, seed, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale], outputs=[output])
    return demo

def create_demo_face_drag(runner):
    DESCRIPTION = """
    ## Face Modulation
    Usage:
    - Upload a source face and a reference face.
    - Click the `Edit` button to start editing."""

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(DESCRIPTION)
        with gr.Column(scale=1.9):
            with gr.Row():
                img_org = gr.Image(source='upload', label="Original Face", interactive=True, type="numpy")
                img_ref = gr.Image(source='upload', label="Reference Face", interactive=True, type="numpy")
            with gr.Row():
                run_button = gr.Button("Edit")
            with gr.Row():
                with gr.Column(scale=1.9):
                    with gr.Box():
                        seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                        guidance_scale = gr.Slider(label="Classifier-free guidance strength", value=4, minimum=1, maximum=10, step=0.1)
                        energy_scale = gr.Slider(label="Classifier guidance strength (x1e3)", value=3, minimum=0, maximum=10, step=0.1)
                        max_resolution = gr.Slider(label="Resolution", value=768, minimum=428, maximum=1024, step=1)
                        with gr.Accordion('Advanced options', open=False):
                            w_edit = gr.Slider(
                                        label="Weight of moving strength",
                                        minimum=0,
                                        maximum=20,
                                        step=0.1,
                                        value=12,
                                        interactive=True)
                            w_inpaint = gr.Slider(
                                        label="Weight of inpainting strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=0.4,
                                        interactive=True)
                            SDE_strength = gr.Slider(
                                        label="Flexibility strength",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.2,
                                        interactive=True)
                            ip_scale = gr.Slider(
                                        label="Image prompt scale",
                                        minimum=0,
                                        maximum=1,
                                        step=0.01,
                                        value=0.05,
                                        interactive=True)
                with gr.Column(scale=1.9):
                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery().style(grid=3, height='auto')
            with gr.Column():
                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples_face,
                    inputs=[img_org, img_ref]
                )
        run_button.click(fn=runner, inputs=[img_org, img_ref, w_edit, w_inpaint, seed, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale], outputs=[output])
    return demo

def create_demo_drag(runner):
    DESCRIPTION = """
    ## Content Dragging
    Usage:
    - Upload a source image.
    - Draw a mask on the source image.
    - Label the content's movement path on the masked image.
    - Add a text description to the image and click the `Edit` button to start editing."""

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(DESCRIPTION)
        original_image = gr.State(value=None) # store original image
        mask = gr.State(value=None) # store mask
        selected_points = gr.State([])
        with gr.Column(scale=1.9):
            with gr.Row():
                img_m = gr.Image(source='upload', tool="sketch", label="Original Image", interactive=True, type="numpy")
                img = gr.Image(source='upload', label="Original Image", interactive=True, type="numpy")
            img.select(
                get_point,
                [img, selected_points],
                [img],
            )
            img_m.edit(
                store_img,
                [img_m],
                [original_image, img, mask]
            )
            with gr.Row():
                run_button = gr.Button("Edit")
                clear_button = gr.Button("Clear points")
            with gr.Row():
                with gr.Column(scale=1.9):
                    prompt = gr.Textbox(label="Prompt")
                    with gr.Box():
                        seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                        guidance_scale = gr.Slider(label="Classifier-free guidance strength", value=4, minimum=1, maximum=10, step=0.1)
                        energy_scale = gr.Slider(label="Classifier guidance strength (x1e3)", value=2, minimum=0, maximum=10, step=0.1)
                        max_resolution = gr.Slider(label="Resolution", value=768, minimum=428, maximum=1024, step=1)
                        with gr.Accordion('Advanced options', open=False):
                            w_edit = gr.Slider(
                                        label="Weight of moving strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=4,
                                        interactive=True)
                            w_content = gr.Slider(
                                        label="Weight of content consistency strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=6,
                                        interactive=True)
                            w_inpaint = gr.Slider(
                                        label="Weight of inpainting strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=0.2,
                                        interactive=True)
                            SDE_strength = gr.Slider(
                                        label="Flexibility strength",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.4,
                                        interactive=True)
                            ip_scale = gr.Slider(
                                        label="Image prompt scale",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.1,
                                        interactive=True)
                with gr.Column(scale=1.9):
                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery().style(grid=1, height='auto')
            with gr.Column():
                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples_drag,
                    inputs=[img_m, prompt]
                )
        run_button.click(fn=runner, inputs=[original_image, mask, prompt, w_edit, w_content, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale], outputs=[output])
        clear_button.click(fn=clear_points, inputs=[img_m], outputs=[selected_points, img])
    return demo

def create_demo_paste(runner):
    DESCRIPTION = """
    ## Object Pasting
    Usage:
    - Upload a reference image, having the target object.
    - Label object masks on the reference image.
    - Upload a background image.
    - Modulate the size and position of the object after pasting.
    - Add a text description to the image and click the `Edit` button to start editing."""

    with gr.Blocks() as demo:
        global_points = gr.State([])
        global_point_label = gr.State([])
        original_image = gr.State(value=None) 
        mask_base = gr.State(value=None) 

        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    gr.Markdown("## 1. Upload image & Draw box to generate mask")
                    img_base = gr.Image(source='upload', label="Original image", interactive=True, type="numpy")
                    img_replace = gr.Image(source="upload", label="Reference image", interactive=True, type="numpy") 
                    gr.Markdown("## 2. Paste position & size")
                    dx = gr.Slider(
                        label="Horizontal movement",
                        minimum=-1000,
                        maximum=1000,
                        step=1,
                        value=0,
                        interactive=True
                    )
                    dy = gr.Slider(
                        label="Vertical movement",
                        minimum=-1000,
                        maximum=1000,
                        step=1,
                        value=0,
                        interactive=True
                    )
                    resize_scale = gr.Slider(
                        label="Resize object",
                        minimum=0,
                        maximum=1.5,
                        step=0.1,
                        value=1,
                        interactive=True
                    )
                    gr.Markdown("## 3. Prompt")
                    prompt = gr.Textbox(label="Prompt")
                    prompt_replace = gr.Textbox(label="Prompt of reference image")
                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

                    with gr.Box():
                        guidance_scale = gr.Slider(label="Classifier-free guidance strength", value=4, minimum=1, maximum=10, step=0.1)
                        energy_scale = gr.Slider(label="Classifier guidance strength (x1e3)", value=1.5, minimum=0, maximum=10, step=0.1)
                        max_resolution = gr.Slider(label="Resolution", value=768, minimum=428, maximum=1024, step=1)
                        with gr.Accordion('Advanced options', open=False):
                            seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                            w_edit = gr.Slider(
                                        label="Weight of pasting strength",
                                        minimum=0,
                                        maximum=20,
                                        step=0.1,
                                        value=4,
                                        interactive=True)
                            w_content = gr.Slider(
                                        label="Weight of content consistency strength",
                                        minimum=0,
                                        maximum=20,
                                        step=0.1,
                                        value=6,
                                        interactive=True)
                            SDE_strength = gr.Slider(
                                        label="Flexibility strength",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.4,
                                        interactive=True)
                            ip_scale = gr.Slider(
                                        label="Image prompt scale",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.1,
                                        interactive=True)

            with gr.Column():
                    with gr.Box():
                        gr.Markdown("# OUTPUT")
                        mask_base_show = gr.Image(source='upload', label="Mask of editing object", interactive=True, type="numpy")
                        gr.Markdown("<h5><center>Results</center></h5>")
                        output = gr.Gallery().style(grid=1, height='auto')
        img_replace.select(
            segment_with_points_paste, 
            inputs=[img_replace, original_image, global_points, global_point_label, img_base, dx, dy, resize_scale], 
            outputs=[img_replace, original_image, mask_base_show, global_points, global_point_label, mask_base]
        )
        img_replace.edit(
            upload_image_move,
            inputs = [img_replace, original_image],
            outputs = [original_image]
        )
        dx.change(
            paste_with_mask_and_offset,
            inputs = [img_replace, img_base, mask_base, dx, dy, resize_scale],
            outputs =  mask_base_show
        )
        dy.change(
            paste_with_mask_and_offset,
            inputs = [img_replace, img_base, mask_base, dx, dy, resize_scale],
            outputs =  mask_base_show
        )
        resize_scale.change(
            paste_with_mask_and_offset,
            inputs = [img_replace, img_base, mask_base, dx, dy, resize_scale],
            outputs =  mask_base_show
        )
        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(
                examples=examples_paste,
                inputs=[img_base, img_replace, prompt, prompt_replace, dx, dy, resize_scale]
            )
            
        clear_button.click(fn=fun_clear, inputs=[original_image, global_points, global_point_label, img_replace, mask_base, img_base], outputs=[original_image, global_points, global_point_label, img_replace, mask_base, img_base])
        run_button.click(fn=runner, inputs=[img_base, mask_base, original_image, prompt, prompt_replace, w_edit, w_content, seed, guidance_scale, energy_scale, dx, dy, resize_scale, max_resolution, SDE_strength, ip_scale], outputs=[output])
    return demo


def create_my_demo(runner):
    DESCRIPTION = """
    ## Baseline Demo with simple copy-paste and inpainting
    Usage:
    - Upload a source image, and then draw a box to generate the mask corresponding to the editing object.
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
                    gr.Markdown("## 1. Draw box to mask target object")
                    img_draw_box = gr.Image(source='upload', label="Draw box", interactive=True, type="numpy")

                    gr.Markdown("## 2. Draw arrow to describe the movement")
                    img = gr.Image(source='upload', label="Original image", interactive=True, type="numpy")

                    gr.Markdown("## 3. Inpaint reference Area (Optional)")
                    img_ref = gr.Image(sourc='upload',tool="sketch", label="Original image", interactive=True, type="numpy")

                    gr.Markdown("## 4. Prompt")
                    prompt = gr.Textbox(label="Prompt")

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
                        eta= gr.Slider(
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

                        mode = gr.Slider(
                            label="mode select for inpainting version 0:cv2.inpaint 1:laMa 2:sd-inpaint",
                            minimum=0,
                            maximum=2,
                            step=1,
                            value=1,
                            interactive=True)
                        move_with_mask = gr.Slider(
                            label="move with DIY mask 0:False 1:True",
                            minimum=0,
                            maximum=1,
                            step=1,
                            value=0,
                            interactive=True)
                        max_resolution = gr.Slider(label="Resolution", value=768, minimum=428, maximum=1024, step=1)
                        with gr.Accordion('Advanced options', open=False):
                            seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                            resize_scale = gr.Slider(
                                label="Object resizing scale",
                                minimum=0,
                                maximum=10,
                                step=0.1,
                                value=1,
                                interactive=True)
                        dilate_kernel_size = gr.Slider(
                            label="dilate_kernel_size for inpainting mask dilation",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=15,
                            interactive=True)


            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    mask = gr.Image(source='upload', label="Mask of object", interactive=True, type="numpy")
                    im_w_mask_ref = gr.Image(label="Mask of reference region", interactive=True, type="numpy")

                    gr.Markdown("<h5><center>GenResults</center></h5>")
                    output = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>EditResults</center></h5>")
                    output_edit = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>NoisedImage</center></h5>")
                    noised_img = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>InpaintMask</center></h5>")
                    INP_Mask = gr.Gallery().style(grid=1, height='auto')

                    # im_w_mask_ref = gr.Image(label="Mask of inpaint region", interactive=True, type="numpy")

            img.select(
                get_point_move,
                [original_image, img, selected_points],
                [img, original_image, selected_points,],
            )
            img_draw_box.select(
                segment_with_points,
                inputs=[img_draw_box, original_image, global_points, global_point_label, img],
                outputs=[img_draw_box, original_image, mask, global_points, global_point_label, img]
            )
            img_ref.edit(
                draw_inpaint_area,
                [img_ref],
                [original_image,im_w_mask_ref,mask_ref]
            )
        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(
                examples=examples_CPIG,
                inputs=[img_draw_box,img_ref, prompt]
            )

        # run_button.click(fn=runner,
        #                  inputs=[original_image, mask, mask_ref, prompt, resize_scale, w_edit, w_content, w_contrast,
        #                          w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution,
        #                          SDE_strength, ip_scale], outputs=[output])

        run_button.click(fn=runner,
                         inputs=[original_image, mask, prompt,  seed, selected_points, guidance_scale, num_step, max_resolution,mode,dilate_kernel_size,
                                 start_step,mask_ref,eta,move_with_mask], outputs=[output_edit ,output , noised_img,INP_Mask])
        clear_button.click(fn=fun_clear,
                           inputs=[original_image, global_points, global_point_label, selected_points, mask,
                                   img_draw_box, img, output, output_edit, noised_img,INP_Mask,img_ref],
                           outputs=[original_image, global_points, global_point_label, selected_points, mask,
                                    img_draw_box, img, output, output_edit , noised_img, INP_Mask,img_ref ])
    return demo

def create_my_demo_full_3D(runner):
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
                    gr.Markdown("## 1. Draw box to mask target object(optional)")
                    img_draw_box = gr.Image(source='upload', label="Draw box", interactive=True, type="numpy")

                    gr.Markdown("## 2. Draw arrow to describe the movement")
                    img = gr.Image(source='upload', label="Original image", interactive=True, type="numpy")

                    gr.Markdown("## 3. casual draw to mask target object(optional)")
                    img_ref = gr.Image(sourc='upload',tool="sketch", label="Original image", interactive=True, type="numpy")

                    gr.Markdown("## 4. Prompt")
                    prompt = gr.Textbox(label="Prompt")

                    gr.Markdown("## 5.Inpaint Prompt")
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
                            value=5,
                            interactive=True)
                        eta= gr.Slider(
                            label="eta setting in DDIM denoising process 0:DDIM 1:DDPM",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.5,
                            interactive=True)
                        num_step = gr.Slider(
                            label="number of diffusion steps",
                            minimum=0,
                            maximum=1000,
                            step=1,
                            value=10,
                            interactive=True)
                        start_step = gr.Slider(
                            label="number of start step of num_step",
                            minimum=0,
                            maximum=1000,
                            step=1,
                            value=5,
                            interactive=True)
                        mask_threshold = gr.Slider(
                            label=" mask_threshold",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.1,
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
                        exp_mask_type = gr.Slider(
                            label="exp_mask_type-0:INV 1:FOR 2:BOTH",
                            minimum=0,
                            maximum=2,
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
                            interactive=True)
                        standard_drawing = gr.Slider(
                            label="select the box draw or casual draw to upload mask 0:casual draw 1:standard box draw",
                            minimum=0,
                            maximum=1,
                            step=1,
                            value=1,
                            interactive=True)
                        blending_alpha= gr.Slider(
                            label="alpha_blending value for blending edited regions and original images",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.3,
                            interactive=True)
                        max_resolution = gr.Slider(label="Resolution", value=768, minimum=428, maximum=1024, step=1)
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
                            mask_threshold_target = gr.Slider(
                                label=" mask_threshold_target",
                                minimum=0,
                                maximum=1,
                                step=0.1,
                                value=0.1,
                                interactive=True)



            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    mask = gr.Image(source='upload', label="Mask of object", interactive=True, type="numpy")
                    im_w_mask_ref = gr.Image(label="Mask of reference region", interactive=True, type="numpy")

                    gr.Markdown("<h5><center>GenResults</center></h5>")
                    output = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>EditResults</center></h5>")
                    output_edit = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>NoisedImage</center></h5>")
                    noised_img = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>InpaintingMask</center></h5>")
                    INP_Mask = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>ExpansionMask</center></h5>")
                    EXP_Mask = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>ExpansionMask_2</center></h5>")
                    EXP_Mask_2 = gr.Gallery().style(grid=1, height='auto')

                    gr.Markdown("<h5><center>Retain_Region</center></h5>")
                    retain_region = gr.Gallery().style(grid=1, height='auto')

                    # im_w_mask_ref = gr.Image(label="Mask of inpaint region", interactive=True, type="numpy")

            img.select(
                get_point_move,
                [original_image, img, selected_points],
                [img, original_image, selected_points,],
            )
            img_draw_box.select(
                segment_with_points,
                inputs=[img_draw_box, original_image, global_points, global_point_label, img],
                outputs=[img_draw_box, original_image, mask, global_points, global_point_label, img]
            )
            img_ref.edit(
                draw_inpaint_area,
                [img_ref],
                [original_image,im_w_mask_ref,mask_ref]
            )
        with gr.Column():
            gr.Markdown("Try some of the examples below ⬇️")
            gr.Examples(
                examples=examples_CPIG_FULL,
                inputs=[img_draw_box,img_ref,img,prompt,INP_prompt]
            )

        # run_button.click(fn=runner,
        #                  inputs=[original_image, mask, mask_ref, prompt, resize_scale, w_edit, w_content, w_contrast,
        #                          w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution,
        #                          SDE_strength, ip_scale], outputs=[output])

        run_button.click(fn=runner,
                         inputs=[original_image, mask, prompt, INP_prompt, seed, selected_points, guidance_scale, num_step, max_resolution,mode,dilate_kernel_size,
                                 start_step,mask_ref,eta,use_mask_expansion,standard_drawing,contrast_beta,exp_mask_type,resize_scale,rotation_angle,strong_inpaint,flip_horizontal,flip_vertical,cross_enhance,
                                 mask_threshold,mask_threshold_target,blending_alpha], outputs=[output_edit ,output ,noised_img ,INP_Mask,EXP_Mask,EXP_Mask_2,retain_region])
        clear_button.click(fn=fun_clear,
                           inputs=[original_image, global_points, global_point_label, selected_points, mask,mask_ref,
                                   img_draw_box, img, output, output_edit, noised_img,INP_Mask,EXP_Mask,img_ref,EXP_Mask_2,retain_region],
                           outputs=[original_image, global_points, global_point_label, selected_points, mask,mask_ref,
                                    img_draw_box, img, output, output_edit , noised_img,INP_Mask, EXP_Mask,img_ref,EXP_Mask_2,retain_region ])
    return demo