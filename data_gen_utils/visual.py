import cv2, os
import numpy as np
from PIL import  Image
import  matplotlib.pyplot as plt
def temp_view_img(image: Image.Image, title: str = None) -> None:
    # PIL -> ndarray OR ndarray->PIL->ndarray
    if not isinstance(image, Image.Image):  # ndarray
        # image_array = Image.fromarray(image).convert('RGB')
        image_array = image
    else:  # PIL
        if image.mode != 'RGB':
            image.convert('RGB')
        image_array = np.array(image)

    plt.imshow(image_array)
    if title is not None:
        plt.title(title)
    plt.axis('off')  # Hide the axis
    plt.show()

darken_factor = 0.5
image_id = 14
dash_length=10
gap_length=5
def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask
# o_image_path = "/data/Hszhu/dataset/Geo-Bench-SC/source_img/33.png"
# o_image_path = "/data/Hszhu/dataset/Geo-Bench-SC/3433.png"
o_image_path = "/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/img_5.png"
# t_image_path = f"compare/{image_id}/our.png"
# m_image_path = "/data/Hszhu/dataset/Geo-Bench-SC/target_mask/33/0/2.png"
m_image_path = "/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/tar_mask.png"
# om_image_path = "/data/Hszhu/dataset/Geo-Bench-SC/source_mask/33/0.png"
om_image_path = "/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/mask_5.png"
# draw_mask_path = "/data/Hszhu/dataset/Geo-Bench-SC/draw_mask/33/0/draw_2.png"


image = cv2.imread(o_image_path)
mask = cv2.resize(cv2.imread(m_image_path, cv2.IMREAD_GRAYSCALE), (image.shape[1],image.shape[0], ))
omask = cv2.resize(cv2.imread(om_image_path, cv2.IMREAD_GRAYSCALE), (image.shape[1],image.shape[0], ))

_, binary_mask = cv2.threshold(omask, 127, 255, cv2.THRESH_BINARY)
_, omask = cv2.threshold(omask, 127, 255, cv2.THRESH_BINARY)
image[omask == 255] = (image[omask == 255] * darken_factor).astype(np.uint8)

# if os.path.exists(draw_mask_path):
#     draw_mask = cv2.resize(cv2.imread(draw_mask_path, cv2.IMREAD_GRAYSCALE), (image.shape[0], image.shape[1]))
#     _, draw_mask = cv2.threshold(draw_mask, 127, 255, cv2.THRESH_BINARY)
#     color_layer = image.copy()
#     # (102, 217, 255) (190, 251, 183)
#     color_layer[draw_mask == 255] = (141, 208, 168)
#     a = 0.5
#     image = cv2.addWeighted(image, 1-a, color_layer, a, 0)

ocontours, _ = cv2.findContours(omask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, ocontours, -1, (255, 255, 255), thickness=5)
# cv2.drawContours(image, contours, -1, (102, 217, 255), thickness=5)

# cv2.imshow("Contour on Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite(f"/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/sun_1.png",image)

# #TODO:HERE
# da_n = 45
# ins_n = 0
# edit_n = 6
# coar_img_path = f"/data/Hszhu/dataset/Geo-Bench-SC/coarse_img/{da_n}/{ins_n}/{edit_n}.png"
# # coar_img_path = "/data/Hszhu/dataset/PIE-Bench_v1/Subset_0/coarse_input/6/0/8.png"
# draw_mask_path = f"/data/Hszhu/dataset/Geo-Bench-SC/draw_mask/{da_n}/{ins_n}/draw_{edit_n}.png"
# # draw_mask_path =  "/data/Hszhu/Reggio/draw_mask2.png"
# #
# image = cv2.cvtColor(cv2.imread(coar_img_path),cv2.COLOR_BGR2RGB)
# mask = cv2.resize(cv2.imread(draw_mask_path, cv2.IMREAD_GRAYSCALE), (image.shape[0], image.shape[1]))
# if os.path.exists(draw_mask_path):
#     draw_mask = cv2.resize(cv2.imread(draw_mask_path, cv2.IMREAD_GRAYSCALE), (image.shape[0], image.shape[1]))
#     _, draw_mask = cv2.threshold(draw_mask, 127, 255, cv2.THRESH_BINARY)
#     color_layer = image.copy()
#     # (102, 217, 255) (190, 251, 183)
#     color_layer[draw_mask == 255] = (255, 255, 0)
#     a = 0.5
#     image = cv2.addWeighted(image, 1-a, color_layer, a, 0)
#
# temp_view_img(image)
# gen_img_dir = "/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir_2/"
# subfolder_path = os.path.join(gen_img_dir, str(da_n))
# os.makedirs(subfolder_path, exist_ok=True)
#
# ins_subfolder_path = os.path.join(subfolder_path, str(ins_n))
# os.makedirs(ins_subfolder_path, exist_ok=True)
# final_path = os.path.join(ins_subfolder_path, f"mask_{edit_n}.png")
#
# Image.fromarray(image).save(final_path)  # 保存为PNG格式（单通道
#
#
#
#
o_image_path = "/data/Hszhu/dataset/Geo-Bench-SC/source_img/33.png"
# t_image_path = f"compare/{image_id}/our.png"
# m_image_path = "/data/Hszhu/dataset/Geo-Bench-SC/source_mask/33/0.png"
draw_mask_path = "/data/Hszhu/dataset/Geo-Bench-SC/source_mask/33/0.png"
coar_img_path = "/data/Hszhu/dataset/Geo-Bench-SC/temp_viss.png"
# coar_img_path = "/data/Hszhu/dataset/PIE-Bench_v1/Subset_0/coarse_input/6/0/8.png"
# draw_mask_path = "/data/Hszhu/dataset/Geo-Bench-SC/source_img/33.png"
# draw_mask_path =  "/data/Hszhu/Reggio/draw_mask2.png"
#
o_image_path = "/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/img_5.png"

# t_image_path = f"compare/{image_id}/our.png"
# m_image_path = "/data/Hszhu/dataset/Geo-Bench-SC/target_mask/33/0/2.png"
m_image_path = "/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/tar_mask.png"
# om_image_path = "/data/Hszhu/dataset/Geo-Bench-SC/source_mask/33/0.png"
om_image_path = "/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/mask_5.png"
coar_img_path = "/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/coarse_2.png"
draw_mask_path = om_image_path
image = cv2.cvtColor(cv2.imread(coar_img_path),cv2.COLOR_BGR2RGB)
mask = cv2.resize(cv2.imread(draw_mask_path, cv2.IMREAD_GRAYSCALE), (image.shape[1],image.shape[0]))
if os.path.exists(draw_mask_path):
    draw_mask = cv2.resize(cv2.imread(draw_mask_path, cv2.IMREAD_GRAYSCALE), (image.shape[1],image.shape[0], ))
    draw_mask = dilate_mask(draw_mask,30)
    _, draw_mask = cv2.threshold(draw_mask, 127, 255, cv2.THRESH_BINARY)
    color_layer = image.copy()
    # (102, 217, 255) (190, 251, 183)
    color_layer[draw_mask == 255] = (255, 255, 0)
    a = 0.5
    image = cv2.addWeighted(image, 1-a, color_layer, a, 0)

temp_view_img(image)

Image.fromarray(image).save("/data/Hszhu/dataset/Geo-Bench-SC/vis_Dir/sun_2.png")  # 保存为PNG格式（单通道

