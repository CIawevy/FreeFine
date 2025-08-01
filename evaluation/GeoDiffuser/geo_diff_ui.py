import gradio as gr
import os
from GeoDiffuser.utils.ui_utils import get_mask, get_depth, get_transformed_mask, get_edited_image, correct_depth, inpaint_mask, get_stitched_image, save_exp, read_exp_ui, clear_transforms, get_points, undo_point
from PIL import Image
import numpy as np
from copy import deepcopy


# def run_ui():
    
LENGTH=512 # length of the square area displaying/editing images
SAM_PATH = "./checkpoints/sam_vit_h_4b8939.pth"
MIDAS_DEPTH_PATH = "./weights/dpt_large-midas-2f21e586.pt"

def show_options():
    return gr.Row(visible=True)

def hide_options():
    return gr.Row(visible=False)

def copy_image(img):

    return deepcopy(img)


def resize_image_to_size(img, h, w):

    # print(type(img), type(h), type(w))

    h = int(h)
    w = int(w)
    # print(h, w)
    if img is None:
        return None
    input_img = np.array(Image.fromarray(img).resize((w, h), Image.NEAREST))
    return input_img


def resize_image(img):
    # print("Resizing image")
    input_img = np.array(Image.fromarray(img).resize((LENGTH, LENGTH)))
    return input_img, deepcopy(input_img)

def resize_image_bg(img):
    # print("Resizing image")
    original_h, original_w = img.shape[0], img.shape[1]
    input_img = np.array(Image.fromarray(img).resize((LENGTH, LENGTH)))
    return input_img, original_h, original_w

def resize_image_and_get_constant_depth(img):
    # print("Resizing image")

    original_h, original_w = img.shape[0], img.shape[1]
    input_img = np.array(Image.fromarray(img).resize((LENGTH, LENGTH)))

    depth = np.ones_like(input_img)
    depth_image = np.ones_like(input_img)
    depth, depth_im_vis = get_depth(input_img, "", depth, depth_image, depth_model = "constant_depth")

    return input_img, depth, depth_im_vis, int(original_h), int(original_w)

with gr.Blocks() as demo:
    # layout definition
    with gr.Row():
        gr.Markdown("""
        # Official Implementation of Geometry Diffusion Editor: GeoDiffuser
        """)

    # UI components for editing real images
    with gr.Tab(label="Editing Real Image"):
        # mask = gr.State(value=None) # store mask
        sam_path = gr.State(SAM_PATH)
        depth_pt_path = gr.State(MIDAS_DEPTH_PATH)

        transform_in = gr.State(value=np.eye(4).astype("float")) # store depth

        depth_image = gr.State(value=None) # store depth
        selected_points = gr.State([]) # store points
        # input_image = gr.State(value=None) # stores unedited input image

        # original_image = gr.State(value=None) # store original input image
        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Input <br> Image </p>""")
                input_image = gr.Image(type="numpy", label="Click Points \n",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=True) # for points clicking
                

    
                # gr.Markdown("""<p style="text-align: center; font-size: 10px">Foreground Image <br> Please set SAM checkpoint below before clicking on the image </p>""")
                with gr.Row():
                    sam_path = gr.Textbox(label = "SAM checkpoint path. Please change accordingly.", value = SAM_PATH, interactive=True)

                with gr.Row():
                    load_location = gr.Textbox(label = "Load exp directory", value = "/oscar/scratch/rsajnani/rsajnani/research/2023/GeometryDiffuser/dataset/large_scale_study_optimizer_sd_test/Translation_3D/34", interactive=True)
                with gr.Row():
                    load_exp_button = gr.Button("Load Experiment")

        # with gr.Row()

            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Foreground Image <br> Click Points to Select Object</p>""")
                click_points_image = gr.Image(type="numpy", label="Mask Image",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False)
                
                with gr.Row():

                    point_label = gr.Radio(choices=["Positive",  "Negative"], value="Positive", label="Point prompt", interactive=True)
                    undo_button = gr.Button("Undo point")
                    
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Image <br> Mask</p>""")
                mask_image = gr.Image(type="numpy", label="Mask Image",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False)
                
                with gr.Row():
                    H_txt = gr.Number(label="Height", value = LENGTH, interactive=False)
                    W_txt = gr.Number(label="Width", value= LENGTH, interactive=False)
                

                
                with gr.Row(visible=False):
                    background_upload = gr.Button("Show Background Image Tab")
                

        with gr.Row(visible=False) as back_tab:
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Foreground <br> Image</p>""")
                input_show_image = gr.Image(type="numpy", label="Foreground Image",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False)
                
                input_image.upload(fn = resize_image, inputs=[input_image], outputs=[input_show_image, click_points_image])
                input_image.change(fn = copy_image, inputs=[input_image], outputs=[click_points_image])              

            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Background Image for Stitching. <br> Leave Empty for Other Tasks</p>""")
                background_image = gr.Image(type="numpy", label="Background Image For Stitching",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=True)
                background_image.upload(fn = resize_image_bg, inputs=[background_image], outputs=[background_image, H_txt, W_txt])

                # undo_button = gr.Button("Undo point")

        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Transformed Image</p>""")
                transformed_image = gr.Image(type="numpy", label="Transformed Mask",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False) # for points clicking

            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Depth Image</p>""")
                depth_image_vis = gr.Image(type="numpy", label="Depth Image",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False)

            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Depth Options </p>""")

                with gr.Row():
                    translate_factor = gr.Slider(label='Push object depth farther away from camera [0-1]',
                    info='Push object depth farther away from camera [0-1]',
                    minimum=0,
                    maximum=5,
                    step=0.01,
                    value=0.7)
                
                with gr.Row():
                    depth_model = gr.Dropdown(label = "Depth Estimator", choices = ["midas_depth", "depth_anything", "constant_depth", "zoe_depth"], value = "depth_anything")
                    depth_button = gr.Button("Get Depth")
                # with gr.Row():

                with gr.Row():
                    depth_pt_path = gr.Textbox(label = "MIDAS checkpoint path checkpoint path", value = MIDAS_DEPTH_PATH, interactive=True)  

            # with gr.Column():
            #     gr.Markdown("""<p style="text-align: center; font-size: 20px">Place holder Image</p>""")
            #     placeholder = gr.Image(type="numpy", label="placeholder",
            #         show_label=True, height=LENGTH, width=LENGTH, interactive=False)


                # undo_button = gr.Button("Undo point")
        with gr.Row():
            
            with gr.Column():

                with gr.Row():

                    translation_x = gr.Slider(label='tx',
                                info='Translation slider along x axis',
                                minimum=-0.50,
                                maximum=0.50,
                                step=0.001,
                                value=0.0)
                    translation_y = gr.Slider(label='ty',
                                info='Translation slider along y axis',
                                minimum=-0.50,
                                maximum=0.50,
                                step=0.001,
                                value=0.0)
                    translation_z = gr.Slider(label='tz',
                                info='Translation slider along z axis',
                                minimum=-0.50,
                                maximum=0.50,
                                step=0.001,
                                value=0.0)

                with gr.Row():
                    clear_transforms_button = gr.Button("Clear Transforms")


                

                
            with gr.Column():
                
                with gr.Row():
                    rotation_x = gr.Slider(label='rx',
                                info='Rotation slider along x axis',
                                minimum=-90,
                                maximum=90,
                                step=1,
                                value=0.0)
                    rotation_y = gr.Slider(label='ry',
                                info='Rotation slider along y axis',
                                minimum=-90,
                                maximum=90,
                                step=1,
                                value=0.0)
                    rotation_z = gr.Slider(label='rz',
                                info='Rotation slider along z axis',
                                minimum=-90,
                                maximum=90,
                                step=1,
                                value=0.0)

                with gr.Row():
                    transform_button = gr.Button("Check Transformed Image")





            with gr.Column():
                with gr.Row():
                    scale_x = gr.Slider(label='sx',
                                info='Scale x',
                                minimum=0.5,
                                maximum=1.5,
                                step=0.01,
                                value=1.0)
                    scale_y = gr.Slider(label='sy',
                                info='Scale y',
                                minimum=0.5,
                                maximum=1.5,
                                step=0.01,
                                value=1.0)
                    scale_z = gr.Slider(label='sz',
                                info='Scale z',
                                minimum=0.5,
                                maximum=1.5,
                                step=0.01,
                                value=1.0)
                
                with gr.Row():
                    exp_transform_type = gr.Dropdown(label = "Experiment Type", choices = ["Mix", "Rotation_3D", "Translation_3D", "Removal", "Rotation_2D", "Translation_2D", "Scaling"], value = "Mix")
                    save_location = gr.Textbox(label = "Save Directory Parent Path", value = "./ui_outputs/", interactive=True)
                
                with gr.Row():
                    save_button = gr.Button("Save Experiment")

                
                with gr.Row(visible=False):
                    advanced_options_button = gr.Button("View Advanced Options")

                # with gr.Row(visible=False):
                #     movement_panel_show = gr.Button("Show Geometry Editing Loss Weights for Movement")

                



                

        # with gr.Row(visible=False) as advanced_options:
        # with gr.Row():
        with gr.Row():
            with gr.Accordion("Advanced Editing Options. Open for More!", open=False):
                # Not happy with this
                # gr.Markdown("""<p style="text-align: center; font-size: 20px">Advanced Options</p>""")
                gr.Markdown("""
                ### Advanced Options
                """)
                with gr.Row():
                    guidance_scale = gr.Slider(label='g_scale',
                                info='Guidance Scale',
                                minimum=0.0,
                                maximum=10.0,
                                step=0.1,
                                value=3.0)

                    cross_replace_steps = gr.Slider(label='Cross replace',
                                info='Cross replace',
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.97)
                    self_replace_steps = gr.Slider(label='Self replace',
                                info='Self replace',
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.97)
                    
                    



                    


                    # sigma_color = gr.Slider(label='sigma_color bilateral smoothing',
                    #             info='sigma_color bilateral smoothing',
                    #             minimum=0,
                    #             maximum=100,
                    #             step=0.01,
                    #             value=0.1)

                    # sigma_space = gr.Slider(label='sigma_space bilateral smoothing',
                    #             info='sigma_space bilateral smoothing',
                    #             minimum=0,
                    #             maximum=100,
                    #             step=0.01,
                    #             value=16)

                    # d_bf_radius = gr.Slider(label='d radius bilateral',
                    #             info='d radius bilateral',
                    #             minimum=0,
                    #             maximum=100,
                    #             step=1,
                    #             value=5)




                with gr.Row():
                    skip_steps = gr.Slider(label='skip_steps',
                                info='Skip Steps',
                                minimum=0,
                                maximum=10,
                                step=1,
                                value=2)

                    latent_replace_steps = gr.Slider(label='Latent replace',
                                info='Latent replace',
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.1)

                    optimize_steps = gr.Slider(label='Optimize steps',
                                info='optimize steps',
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.65)

                    fast_optim_steps = gr.Slider(label='Fast Optim Steps',
                                info='Fast Optim Steps',
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.0)

                    cam_focal_length = gr.Slider(label='cam_focal_length',
                                info='cam_focal_length',
                                minimum=0,
                                maximum=3000,
                                step=0.1,
                                value=550)
                    




                

                with gr.Row():
                    num_ddim_steps = gr.Slider(label='DDIM steps',
                                    info='ddim steps',
                                    minimum=25,
                                    maximum=50,
                                    step=1,
                                    value=50)    

                    num_first_optim_steps = gr.Slider(label='Num first optim steps',
                                    info='Num first optim steps',
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    value=1)  

                    optim_lr = gr.Slider(label='learning rate',
                                info='learning rate',
                                minimum=0.001,
                                maximum=10.0,
                                step=0.001,
                                value=0.03)  

                    splatting_radius = gr.Slider(label='splatting radius',
                                info='splatting radius',
                                minimum=0.0,
                                maximum=5.0,
                                step=0.01,
                                value=1.3)
                    
                    splatting_tau = gr.Slider(label='splatting tau',
                                info='splatting tau',
                                minimum=1e-3,
                                maximum=2.0,
                                step=1e-3,
                                value=1.0)

                    splatting_points_per_pixel = gr.Slider(label='splatting points per pixel',
                                info='splatting points per pixel',
                                minimum=1,
                                maximum=30,
                                step=1,
                                value=15)

                    # depth_correct_button = gr.Button("Median Filter on Depth")

        # with gr.Row(visible=True) as movement_options:
        with gr.Row():
            with gr.Accordion("Movement Loss Control", open=False):
            

                gr.Markdown("""<p style="text-align: center; font-size: 20px">Movement loss</p>""")
                
                with gr.Row():
                    movement_sim_loss_w_self = gr.Slider(label='Background loss (self)',
                                info='background loss (self)',
                                minimum=0.0,
                                maximum=1000,
                                step=0.001,
                                value=55.0)

                    movement_sim_loss_w_cross = gr.Slider(label='Background loss (cross)',
                                info='background loss (cross)',
                                minimum=0.0,
                                maximum=1000.0,
                                step=0.001,
                                value=45.0)

                    movement_removal_loss_w_self = gr.Slider(label='loss removal_scale (self)',
                                info='loss removal_scale (self)',
                                minimum=0.0,
                                maximum=1000.0,
                                step=0.01,
                                value=2.6)

                    movement_removal_loss_w_cross = gr.Slider(label='loss removal_scale (cross)',
                                info='loss removal_scale (cross)',
                                minimum=0.0,
                                maximum=1000.0,
                                step=0.01,
                                value=2.6)

                    removal_loss_adaptive_value = gr.Slider(label='Removal Loss Adaptive Value',
                                info='Value after which stop adaptive optimization',
                                minimum=-30.0,
                                maximum=0.0,
                                step=0.01,
                                value=-2.5)
                                
                    

                    
                    
                with gr.Row():
                    movement_loss_w_self = gr.Slider(label='foreground preservation loss (self)',
                                info='foreground preservation loss (self)',
                                minimum=0.0,
                                maximum=1000.0,
                                step=0.01,
                                value=30.5)
                    
                    movement_loss_w_cross = gr.Slider(label='foreground preservation loss (cross)',
                                info='foreground preservation loss (cross)',
                                minimum=0.0,
                                maximum=1000.0,
                                step=0.01,
                                value=30.34)

                    amodal_loss_w_self = gr.Slider(label='amodal loss (self)',
                                info='amodal loss (self)',
                                minimum=0.0,
                                maximum=1000.0,
                                step=0.01,
                                value=80.5)

                    amodal_loss_w_cross = gr.Slider(label='amodal loss (cross)',
                                info='amodal loss (cross)',
                                minimum=0.0,
                                maximum=1000.0,
                                step=0.01,
                                value=3.5)
                            

                with gr.Row():
                    movement_smoothness_loss_w_self = gr.Slider(label='loss movement_smoothness (self)',
                                info='loss movement_smoothness (self)',
                                minimum=0.0,
                                maximum=1000.0,
                                step=0.01,
                                value=30)
                    
                    movement_smoothness_loss_w_cross = gr.Slider(label='loss movement_smoothness (cross)',
                                info='loss movement_smoothness (cross)',
                                minimum=0.0,
                                maximum=1000.0,
                                step=0.01,
                                value=15)

                    diffusion_correction = gr.Slider(label='Diffusion Correction',
                                info='Diffusion Correction vs Edit Adherance. Setting high diffusion correction (~0.6-1.0) reduces edit adherance and does not perform the edit',
                                minimum=0.0,
                                maximum=0.4,
                                step=0.01,
                                value=0.1)


        with gr.Row():
            gr.Markdown("""<p style="text-align: center; font-size: 20px">Scroll Below for Input and Edited Image in Original Aspect Ratio.</p>""")
            
        with gr.Row():

            
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Edited Image</p>""")
                edited_image = gr.Image(type="numpy", label="Edited Image",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False)  



            
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Editing Options</p>""")
                
                with gr.Row():
                    prompt = gr.Textbox(info="Prompt for Editing (Optional)", value="")
                                
                # Add feature later!
                with gr.Row():
                    diffusion_model = gr.Dropdown(label = "Diffusion Model", choices = [
                        "CompVis/stable-diffusion-v1-4", 
                        "runwayml/stable-diffusion-v1-5", 
                        "stabilityai/stable-diffusion-2-base", 
                        "stabilityai/stable-diffusion-2-1-base"], 
                        value = "CompVis/stable-diffusion-v1-4")
        
                edit_button = gr.Button("Move Object")


        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Input image in original aspect ratio</p>""")
                download_image_input = gr.Image(type="numpy", label="Download Input Image",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False)   
            
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Edited image in original aspect ratio</p>""")
                download_image = gr.Image(type="numpy", label="Download Image",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False)  


        with gr.Row(visible=False) as stitching_options:
            # movement_panel_hide = gr.Button("Hide Movement Panel")
            gr.Markdown("""<p style="text-align: center; font-size: 20px">Stitching loss</p>""")
            
            with gr.Column():    
                stitching_sim_loss_w_self = gr.Slider(label='Background preservation loss (self)',
                            info='background preservation loss (self)',
                            minimum=0.0,
                            maximum=1000.0,
                            step=0.001,
                            value=20.0)

                stitching_sim_loss_w_cross = gr.Slider(label='Background preservation loss (cross)',
                            info='background preservation loss (cross)',
                            minimum=0.0,
                            maximum=1000.0,
                            step=0.001,
                            value=2.5)


                stitching_sim_loss_out_w_self = gr.Slider(label='Background loss Out (self)',
                            info='background loss Out (self)',
                            minimum=0.0,
                            maximum=1000.0,
                            step=0.001,
                            value=120)

                stitching_sim_loss_out_w_cross = gr.Slider(label='Background loss Out (cross)',
                            info='background loss Out (cross)',
                            minimum=0.0,
                            maximum=1000.0,
                            step=0.001,
                            value=30)

            with gr.Column():    
                stitching_movement_loss_w_self = gr.Slider(label='movement loss (self)',
                    info='movement loss (self)',
                    minimum=0.0,
                    maximum=1000.0,
                    step=0.01,
                    value=40)
                    
                stitching_movement_loss_w_cross = gr.Slider(label='movement loss (cross)',
                    info='movement loss (cross)',
                    minimum=0.0,
                    maximum=1000.0,
                    step=0.01,
                    value=5)

                stitching_movement_loss_out_w_self = gr.Slider(label='movement_out loss (self)',
                            info='movement_out loss (self)',
                            minimum=0.0,
                            maximum=1000.0,
                            step=0.01,
                            value=40.0)
                
                stitching_movement_loss_out_w_cross = gr.Slider(label='movement_out loss (cross)',
                            info='movement_out loss (cross)',
                            minimum=0.0,
                            maximum=1000.0,
                            step=0.01,
                            value=30.0) 

            with gr.Column():
                stitching_smoothness_loss_w_self = gr.Slider(label='loss stitching_smoothness (self)',
                            info='loss stitching_smoothness (self)',
                            minimum=0.0,
                            maximum=1000.0,
                            step=0.01,
                            value=2.0)
                
                stitching_smoothness_loss_w_cross = gr.Slider(label='loss stitching_smoothness (cross)',
                            info='loss stitching_smoothness (cross)',
                            minimum=0.0,
                            maximum=1000.0,
                            step=0.01,
                            value=1.0)
                
                # stitch_panel_hide = gr.Button("Hide Stitching Panel")
                stitch_button = gr.Button("Stitch Image")


  


        # with gr.Row():
        #     with gr.Column():
        #         gr.Markdown("""<p style="text-align: center; font-size: 20px">Input image in original aspect ratio</p>""")
        #         download_image_input = gr.Image(type="numpy", label="Download Input Image",
        #             show_label=True, height=LENGTH, width=LENGTH, interactive=False)   

        #     with gr.Column():
        #         gr.Markdown("""<p style="text-align: center; font-size: 20px">Edited image in original aspect ratio</p>""")
        #         download_image = gr.Image(type="numpy", label="Download Image",
        #             show_label=True, height=LENGTH, width=LENGTH, interactive=False)   


    with gr.Tab(label="Inpainting"):

        # mask = gr.State(value=None) # store mask


        # transform_in = gr.State(value=np.eye(4).astype("float")) # store depth

        # depth_image = gr.State(value=None) # store depth
        selected_points_inpainting = gr.State([]) # store points

        # original_image = gr.State(value=None) # store original input image
        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Foreground <br> Image</p>""")
                input_image_inpainting = gr.Image(type="numpy", label="Input Image \n",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=True) 

                

                with gr.Row():
                    sam_path_inpainting = gr.Textbox(label = "SAM checkpoint path", value = SAM_PATH, interactive=True)
                    # depth_pt_path = gr.State(MIDAS_DEPTH_PATH)


            with gr.Column():

                gr.Markdown("""<p style="text-align: center; font-size: 20px">Foreground Image <br> Click Points to Select Object</p>""")
                click_points_image_inpainting = gr.Image(type="numpy", label="Mask Image",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False)
                input_image_inpainting.change(fn = copy_image, inputs=[input_image_inpainting], outputs=[click_points_image_inpainting])
                
                with gr.Row():
                    point_label_inpainting = gr.Radio(choices=["Positive",  "Negative"], value="Positive", label="Point prompt", interactive=True)
                    undo_button_inpainting = gr.Button("Undo point")

            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Image <br> Mask</p>""")
                mask_image_inpainting = gr.Image(type="numpy", label="Mask Image",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False)

                with gr.Row():
                    H_txt_inpainting = gr.Number(label="Height", value = LENGTH, interactive=False)
                    W_txt_inpainting = gr.Number(label="Width", value=LENGTH, interactive=False)
                




        with gr.Row(visible=False):
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Transformed Image</p>""")
                transformed_image_inpainting = gr.Image(type="numpy", label="Transformed Mask",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False) # for points clicking
        


        with gr.Row():
            

                # with gr.Row(visible=False):
                #     inpainting_loss_button = gr.Button("Show inpainting loss weights")
                # with gr.Row(visible=False):
                #     stitching_loss_button = gr.Button("Show stitching loss weights")
            with gr.Column():
                with gr.Row():
                    load_location_inpainting = gr.Textbox(label = "Load exp directory", value = "./ui_outputs/rotation_2/3/", interactive=True)
                with gr.Row():
                    load_exp_button_inpainting = gr.Button("Load Experiment")
                
            with gr.Column():

                with gr.Row():
                    exp_transform_type_inpainting = gr.Dropdown(label = "Experiment Type", choices = ["Mix", "Rotation_3D", "Translation_3D", "Removal", "Rotation_2D", "Translation_2D", "Scaling", "Inpainting"], value = "Inpainting")
                    save_location_inpainting = gr.Textbox(label = "Save Directory Parent Path", value = "./ui_outputs/", interactive=True)
                
                with gr.Row():
                    save_button_inpainting = gr.Button("Save Experiment")






                # with gr.Row():
                #     advanced_options_button_inpainting = gr.Button("View Advanced Options")
                


                



                

        with gr.Accordion("Advanced Inpainting Options. Open for More!", open=False):
        
            # Not happy with this
            # gr.Markdown("""<p style="text-align: center; font-size: 20px">Advanced Options</p>""")
            gr.Markdown("""
            ### Advanced Options
            """)
            with gr.Row():
                guidance_scale_inpainting = gr.Slider(label='g_scale',
                            info='Guidance Scale',
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                            value=5.0)

                cross_replace_steps_inpainting = gr.Slider(label='Cross replace',
                            info='Cross replace',
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.95)
                self_replace_steps_inpainting = gr.Slider(label='Self replace',
                            info='Self replace',
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.95)
                
                

                translate_factor_inpainting = gr.Slider(label='Push object depth farther away from camera [0-1]',
                            info='Push object depth farther away from camera [0-1]',
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.1)

                
                # obj_edit_step_inpainting = gr.Slider(label='obj_edit_step',
                #             info='obj_edit_step',
                #             minimum=0,
                #             maximum=1,
                #             step=0.01,
                #             value=0.7)

                # sigma_color_inpainting = gr.Slider(label='sigma_color bilateral smoothing',
                #             info='sigma_color bilateral smoothing',
                #             minimum=0,
                #             maximum=100,
                #             step=0.01,
                #             value=0.1)

                # sigma_space_inpainting = gr.Slider(label='sigma_space bilateral smoothing',
                #             info='sigma_space bilateral smoothing',
                #             minimum=0,
                #             maximum=100,
                #             step=0.01,
                #             value=16)

                # d_bf_radius_inpainting = gr.Slider(label='d radius bilateral',
                #             info='d radius bilateral',
                #             minimum=0,
                #             maximum=100,
                #             step=1,
                #             value=5)




            with gr.Row():
                skip_steps_inpainting = gr.Slider(label='skip_steps',
                            info='Skip Steps',
                            minimum=0,
                            maximum=10,
                            step=1,
                            value=2)

                latent_replace_steps_inpainting = gr.Slider(label='Latent replace',
                            info='Latent replace',
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.1)

                optimize_steps_inpainting = gr.Slider(label='Optimize steps',
                            info='optimize steps',
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.65)

                fast_optim_steps_inpainting = gr.Slider(label='Fast Optim Steps',
                            info='Fast Optim Steps',
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.0)

                cam_focal_length_inpainting = gr.Slider(label='cam_focal_length',
                            info='cam_focal_length',
                            minimum=0,
                            maximum=3000,
                            step=0.1,
                            value=550)
                




            

            with gr.Row():
                num_ddim_steps_inpainting = gr.Slider(label='DDIM steps',
                                info='ddim steps',
                                minimum=25,
                                maximum=50,
                                step=1,
                                value=50)    

                num_first_optim_steps_inpainting = gr.Slider(label='Num first optim steps',
                                info='Num first optim steps',
                                minimum=1,
                                maximum=50,
                                step=1,
                                value=1)  

                optim_lr_inpainting = gr.Slider(label='learning rate',
                            info='learning rate',
                            minimum=0.001,
                            maximum=10.0,
                            step=0.001,
                            value=0.03)  

                splatting_radius_inpainting = gr.Slider(label='splatting radius',
                            info='splatting radius',
                            minimum=0.0,
                            maximum=5.0,
                            step=0.01,
                            value=1.3)
                
                splatting_tau_inpainting = gr.Slider(label='splatting tau',
                            info='splatting tau',
                            minimum=1e-3,
                            maximum=2.0,
                            step=1e-3,
                            value=1.0)

                splatting_points_per_pixel_inpainting = gr.Slider(label='splatting points per pixel',
                            info='splatting points per pixel',
                            minimum=1,
                            maximum=30,
                            step=1,
                            value=15)

                # depth_correct_button = gr.Button("Median Filter on Depth")


        
        # with gr.Row(visible=True) as inpainting_options:
        with gr.Accordion("Inpainting Loss Control", open=False):
        

            gr.Markdown("""<p style="text-align: center; font-size: 20px">Inpainting loss</p>""")
            
            with gr.Row():
                inpainting_sim_loss_w_self = gr.Slider(label='Background preservation loss (self)',
                            info='background preservation loss (self)',
                            minimum=0.0,
                            maximum=1000.0,
                            step=0.001,
                            value=55.0)
                

                inpainting_sim_loss_w_cross = gr.Slider(label='Background preservation loss (cross)',
                            info='background preservation loss (cross)',
                            minimum=0.0,
                            maximum=1000.0,
                            step=0.001,
                            value=45.0)

                

                
                
            with gr.Row():
                inpainting_removal_loss_w_self = gr.Slider(label='Removal Loss (self)',
                            info='Removal Loss (self)',
                            minimum=0.0,
                            maximum=1000.0,
                            step=0.01,
                            value=4.6)
                
                inpainting_removal_loss_w_cross = gr.Slider(label='Removal Loss (cross)',
                            info='Removal Loss (cross)',
                            minimum=0.0,
                            maximum=1000.0,
                            step=0.01,
                            value=4.6)

                removal_loss_adaptive_value_inpainting = gr.Slider(label='Removal Loss Adaptive Value',
                            info='Value after which stop adaptive optimization',
                            minimum=-30.0,
                            maximum=0,
                            step=0.01,
                            value=-2.5)       


            with gr.Row():
                inpainting_smoothness_loss_w_self = gr.Slider(label='loss inpainting_smoothness (self)',
                            info='loss inpainting_smoothness (self)',
                            minimum=0.0,
                            maximum=1000.0,
                            step=0.01,
                            value=30)
                
                inpainting_smoothness_loss_w_cross = gr.Slider(label='loss inpainting_smoothness (cross)',
                            info='loss inpainting_smoothness (cross)',
                            minimum=0.0,
                            maximum=1000.0,
                            step=0.01,
                            value=15)

            
            # with gr.Row(visible=False):
            #     inpaint_panel_hide = gr.Button("Hide Inpainting Panel")


        with gr.Row():

            gr.Markdown("""<p style="text-align: center; font-size: 20px">Scroll Below for Input and Edited Image in Original Aspect Ratio.</p>""")

        with gr.Row():

            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Edited Image</p>""")
                edited_image_inpainting = gr.Image(type="numpy", label="Edited Image",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False)    




            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Editing Options</p>""")
                
                with gr.Row():
                    prompt_inpainting = gr.Textbox(info="Prompt for Editing (Optional)", value="")
                                
                # Add feature later!
                with gr.Row():
                    diffusion_model_inpainting = gr.Dropdown(label = "Diffusion Model", choices = [
                        "CompVis/stable-diffusion-v1-4", 
                        "runwayml/stable-diffusion-v1-5", 
                        "stabilityai/stable-diffusion-2-base", 
                        "stabilityai/stable-diffusion-2-1-base"], 
                        value = "CompVis/stable-diffusion-v1-4")

                with gr.Row():
                    inpaint_mask_button = gr.Button("Inpaint Mask")

                        


        with gr.Row():
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Input image in original aspect ratio</p>""")
                download_image_input_inpainting = gr.Image(type="numpy", label="Download Input Image",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False)   
            
            with gr.Column():
                gr.Markdown("""<p style="text-align: center; font-size: 20px">Edited image in original aspect ratio</p>""")
                download_image_inpainting = gr.Image(type="numpy", label="Download Image",
                    show_label=True, height=LENGTH, width=LENGTH, interactive=False)   


        background_image_inpainting = gr.State(value=None)
        depth_image_inpainting = gr.State(value=None)
        depth_image_vis_inpainting = gr.State(value=None)
        transform_in_inpainting = gr.State(value=None)


                    
    


    
    input_image.upload(fn = resize_image_and_get_constant_depth, inputs=[input_image], outputs=[input_image, depth_image, depth_image_vis, H_txt, W_txt])

    edited_image.change(fn = resize_image_to_size, inputs=[edited_image, H_txt, W_txt], outputs=[download_image])
    edited_image.change(fn = resize_image_to_size, inputs = [input_image, H_txt, W_txt], outputs = [download_image_input])



    save_button.click(
        save_exp,
        [save_location, 
        input_image,
        depth_image,
        depth_image_vis,
        mask_image,
        transform_in,
        transformed_image,
        edited_image,
        background_image,
        H_txt,
        W_txt, 
        exp_transform_type,
        download_image_input,
        download_image],
        [],
    )


    input_image.select(
    get_mask,
    [input_image, mask_image, selected_points, sam_path],
    [input_image, mask_image],
    )



    depth_button.click(
        get_depth, 
        [input_image, depth_pt_path, depth_image, depth_image_vis, depth_model, translate_factor],
        [depth_image, depth_image_vis]
    )

    # depth_correct_button.click(
    #     correct_depth,
    #     [depth_image, mask_image, sigma_color, sigma_space, d_bf_radius],
    #     [depth_image]
    # )

    transform_button.click(
        get_transformed_mask,
        [input_image, mask_image, depth_image, transformed_image, translation_x, translation_y, translation_z, rotation_x, rotation_y, rotation_z, transform_in, splatting_radius, background_image, scale_x, scale_y, scale_z, splatting_tau, splatting_points_per_pixel, cam_focal_length],
        [transformed_image, transform_in]

    )

    edit_button.click(
        get_edited_image,
        [input_image, depth_image, mask_image, transform_in, edited_image, guidance_scale, skip_steps, num_ddim_steps, optim_lr, cross_replace_steps, self_replace_steps, latent_replace_steps, optimize_steps, splatting_radius, movement_sim_loss_w_self, movement_sim_loss_w_cross, movement_loss_w_self, movement_loss_w_cross, movement_removal_loss_w_self, movement_removal_loss_w_cross, movement_smoothness_loss_w_self, movement_smoothness_loss_w_cross, amodal_loss_w_cross, amodal_loss_w_self, splatting_tau, splatting_points_per_pixel, prompt, diffusion_correction, diffusion_model, removal_loss_adaptive_value],
        [edited_image]
    )

    load_exp_button.click(
        read_exp_ui,
        [load_location],
        [input_image, mask_image, background_image, depth_image, depth_image_vis, transform_in, transformed_image, edited_image, H_txt, W_txt]
        # ed["input_image_png"], ed["input_mask_png"], ed["background_image_png"], ed["depth_npy"], ed["depth_png"], ed["transform_npy"], ed["transformed_image_png"], ed["result_png"], ed["image_shape_npy"][0], ed["image_shape_npy"][1]
    )
    

    stitch_button.click(
        get_stitched_image,
        [input_image, background_image, depth_image, mask_image, transform_in, edited_image, guidance_scale, skip_steps, num_ddim_steps, optim_lr, cross_replace_steps, self_replace_steps, latent_replace_steps, optimize_steps, splatting_radius, stitching_movement_loss_w_self, stitching_movement_loss_w_cross, stitching_sim_loss_w_self, stitching_sim_loss_w_cross, fast_optim_steps, num_first_optim_steps, stitching_smoothness_loss_w_self, stitching_smoothness_loss_w_cross, stitching_sim_loss_out_w_self, stitching_sim_loss_out_w_cross, stitching_movement_loss_out_w_self, stitching_movement_loss_out_w_cross, splatting_tau, splatting_points_per_pixel],
        [edited_image]
    )


    click_points_image.select(
        get_points,
        [input_image, selected_points, point_label, sam_path],
        [click_points_image, mask_image],
    )

    undo_button.click(
        undo_point,
        [input_image, selected_points, sam_path],
        [click_points_image, mask_image, selected_points]
    )

        
    # advanced_options_button.click(
    #     show_options,
    #     [],
    #     [advanced_options]
    # )


    clear_transforms_button.click(
        clear_transforms,
        [],
        [translation_x, translation_y, translation_z, rotation_x, rotation_y, rotation_z, scale_x, scale_y, scale_z]
    )


    # movement_panel_show.click(
    #     show_options,
    #     [],
    #     [movement_options]
    # )

    # movement_panel_hide.click(
    #     hide_options,
    #     [],
    #     [movement_options]
    # )



    # stitching_loss_button.click(
    #     show_options,
    #     [],
    #     [stitching_options]
    # )

    # inpaint_panel_hide.click(
    #     hide_options,
    #     [],
    #     [inpainting_options]
    # )

    # stitch_panel_hide.click(
    #     hide_options,
    #     [],
    #     [stitching_options]
    # )
    background_upload.click(
        show_options,
        [],
        [back_tab]
    )



    ################################ Inpainting


    click_points_image_inpainting.select(
        get_points,
        [input_image_inpainting, selected_points_inpainting, point_label_inpainting, sam_path_inpainting],
        [click_points_image_inpainting, mask_image_inpainting],
    )

    undo_button.click(
        undo_point,
        [input_image_inpainting, selected_points_inpainting, sam_path_inpainting],
        [click_points_image_inpainting, mask_image, selected_points]
    )

    input_image_inpainting.upload(fn = resize_image_and_get_constant_depth, inputs=[input_image_inpainting], outputs=[input_image_inpainting, depth_image, depth_image_vis, H_txt_inpainting, W_txt_inpainting])
    
    edited_image_inpainting.change(fn = resize_image_to_size, inputs=[edited_image_inpainting, H_txt_inpainting, W_txt_inpainting], outputs=[download_image_inpainting])
    
    edited_image_inpainting.change(fn = resize_image_to_size, inputs = [input_image_inpainting, H_txt_inpainting, W_txt_inpainting], outputs = download_image_input_inpainting)


    save_button_inpainting.click(
        save_exp,
        [save_location_inpainting, 
        input_image_inpainting,
        depth_image,
        depth_image_vis,
        mask_image_inpainting,
        transform_in,
        transformed_image_inpainting,
        edited_image_inpainting,
        background_image,
        H_txt_inpainting,
        W_txt_inpainting, 
        exp_transform_type_inpainting,
        download_image_input_inpainting,
        download_image_inpainting],
        [],
    )


    input_image_inpainting.select(
    get_mask,
    [input_image_inpainting, mask_image_inpainting, selected_points_inpainting, sam_path_inpainting],
    [input_image_inpainting, mask_image_inpainting],
)   

    inpaint_mask_button.click(
        inpaint_mask,
        [input_image_inpainting, mask_image_inpainting, edited_image_inpainting, guidance_scale_inpainting, skip_steps_inpainting, num_ddim_steps_inpainting, optim_lr_inpainting, cross_replace_steps_inpainting, self_replace_steps_inpainting, latent_replace_steps_inpainting, optimize_steps_inpainting, splatting_radius_inpainting, inpainting_sim_loss_w_self, inpainting_sim_loss_w_cross, inpainting_removal_loss_w_self, inpainting_removal_loss_w_cross, inpainting_smoothness_loss_w_self, inpainting_smoothness_loss_w_cross, diffusion_model_inpainting, prompt_inpainting, removal_loss_adaptive_value_inpainting],
        [edited_image_inpainting]

    )

    load_exp_button_inpainting.click(
        read_exp_ui,
        [load_location_inpainting],
        [input_image_inpainting, mask_image_inpainting, background_image_inpainting, depth_image_inpainting, depth_image_vis_inpainting, transform_in_inpainting, transformed_image_inpainting, edited_image_inpainting, H_txt_inpainting, W_txt_inpainting]

        # input_image, mask_image, background_image, depth_image, depth_image_vis, transform_in, transformed_image, edited_image, H_txt, W_txt
        # ed["input_image_png"], ed["input_mask_png"], ed["background_image_png"], ed["depth_npy"], ed["depth_png"], ed["transform_npy"], ed["transformed_image_png"], ed["result_png"], ed["image_shape_npy"][0], ed["image_shape_npy"][1]
    )

    # advanced_options_button_inpainting.click(
    #     show_options,
    #     [],
    #     [advanced_options_inpainting]
    # )
    # inpainting_loss_button.click(
    #     show_options,
    #     [],
    #     [inpainting_options]
    # )
    # transform_button.click(
    
    # )
    
# if __name__=="__main__":
            
demo.queue()
demo.launch(share=True, debug=True)
# if __name__=="__main__":

#     run_ui()
    

# todo
# improve color histogram post processing - checked no luck
# increase speed by not performing bilinear interpolation repeatedly - done
# Better amodal mask for editing - in progress
# Check gradio demo and fix the issue of hanging - not happening :(
# Write a editing readme that uses depth
# set high depth as default value (more far away from camera essentially) - done
# Clean code - in progress
# Test some wild edits - to add
# Add feature to change model - done
# Make the ui easier to navigate - in progress

    
    
