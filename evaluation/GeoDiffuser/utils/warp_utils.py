import torch
import torch.nn.functional as F
from torch import nn
from GeoDiffuser.utils import softsplat
from pytorch3d.structures import Pointclouds
from pytorch3d.structures import Meshes
import numbers, math



import os.path as osp
import json
from tqdm import  tqdm
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
def temp_view_img(image: Image.Image, title: str = None) -> None:
    # Convert to ndarray if the input is not already in that format
    if not isinstance(image, Image.Image):  # ndarray
        image_array = image
    else:  # PIL
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)

    # Function to crop white borders
    def crop_white_borders(img_array):
        gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
        mask = gray < 255  # Mask of non-white pixels
        coords = np.argwhere(mask)  # Find the coordinates of the non-white pixels
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)
        return img_array[x0:x1+1, y0:y1+1]

    # Crop the white borders
    cropped_image_array = crop_white_borders(image_array)

    # Display the cropped image
    fig, ax = plt.subplots()
    ax.imshow(cropped_image_array)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')  # Hide the axis

    # Remove the white border around the figure
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)

    # Set the position of the axes to fill the entire figure
    ax.set_position([0, 0, 1, 1])

    # Show the image
    plt.show()
def visualize_rgb_image(image: Image.Image, title: str = None) -> None:
    """
    Visualize an RGB image from a PIL Image format with an optional title.

    Parameters:
    image (PIL.Image.Image): The RGB image represented as a PIL Image.
    title (str, optional): The title to display above the image.

    Raises:
    ValueError: If the input is not a PIL Image or is not in RGB mode.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")
    if image.mode != 'RGB':
        raise ValueError("Input image must be in RGB mode.")

    image_array = np.array(image)

    plt.imshow(image_array)
    if title is not None:
        plt.title(title)
    plt.axis('off')  # Hide the axis
    plt.show()
def temp_view( mask, title='Mask', name=None):
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
EPS = 1e-2


from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing, TexturesVertex, TexturesUV, MeshRenderer, MeshRasterizer
from pytorch3d.renderer.points import rasterize_points
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments


import numpy as np

# rast_opt = torch.compile(rasterize_points)

pixel_coords = None


class RasterizePointsXYsBlending(nn.Module):
    """
    Borrowed from: https://github.com/facebookresearch/synsin/blob/501ec49b11030a41207e7b923b949fab8fd6e1b5/models/layers/z_buffer_layers.py#L12
    Rasterizes a set of points using a differentiable renderer. Points are
    accumulated in a z-buffer using an accumulation function
    defined in opts.accumulation and are normalised with a value M=opts.M.
    Inputs:
    - pts3D: the 3D points to be projected
    - src: the corresponding features
    - C: size of feature
    - learn_feature: whether to learn the default feature filled in when
                     none project
    - radius: where pixels project to (in pixels)
    - size: size of the image being created
    - points_per_pixel: number of values stored in z-buffer per pixel
    - opts: additional options

    Outputs:
    - transformed_src_alphas: features projected and accumulated
        in the new view
    """

    def __init__(
        self,
        radius=1.3,
        # Change
        points_per_pixel=15,
        accumulation = 'alphacomposite',
        # Change
        tau = 1.0,
        rad_pow = 2,

    ):
        super().__init__()
        self.radius = radius
        self.points_per_pixel = points_per_pixel
        self.accumulation = accumulation
        self.tau = tau
        self.rad_pow = rad_pow
        self.rasterization_dict = {}
    
    def clear_cache(self):
        self.rasterization_dict = {}
    
    @torch.autocast("cuda")
    @torch.no_grad()
    def forward(self, pts3D, src):
#         pts3D - b, N, 3
#         src - b, F, N

        # print(self.tau, self.points_per_pixel, self.radius)
        
        pts3D = pts3D.to(torch.float32)
        src = src.to(torch.float32)
        bs = src.size(0)
        image_size = np.sqrt(src.shape[-1])

        # Make sure these have been arranged in the same way
        assert pts3D.size(2) == 3
        assert pts3D.size(1) == src.size(2)  
        
#         Convert to pytorch 3d coordinate frame where x - left, y - up, z - front
        pts3D[:,:,1] = - pts3D[:,:,1]
        pts3D[:,:,0] = - pts3D[:,:,0]
        
#         self.radius = image_size * 2
        radius = float(self.radius) / float(image_size) * 2.0

        # print(pts3D.dtype, src.dtype)

        im_size = int((image_size))

        # with torch.autocast("cuda"):

        # pts3D_c = pts3D.detach().clone()
        # Using cached points
        # if im_size in self.rasterization_dict:
        #     points_idx_cache = self.rasterization_dict[im_size]["points_idx"][:1].repeat(bs, 1, 1, 1)
        #     dist_cache = self.rasterization_dict[im_size]["dist"][:1].repeat(bs, 1, 1, 1)
        #     pts3D_cache = self.rasterization_dict[im_size]["pts3d"][:1].repeat(bs, 1, 1, 1)

        pts3D = Pointclouds(points=pts3D, features=src.permute(0,2,1))
        # else:
        points_idx, _, dist = rasterize_points(
                pts3D,  im_size, radius, self.points_per_pixel
            )
            # print(points_idx.shape, dist.shape)
            # exit()
        # if im_size in self.rasterization_dict:
        
        #     print((dist - dist_cache).abs().mean(), f" dist {im_size} {src.shape}")
        #     print(torch.abs(points_idx - points_idx_cache).to(torch.half).mean(), f" point_idx {im_size} {src.shape}")
        #     print(points_idx.shape, points_idx_cache.shape)
        #     print((pts3D_c - pts3D_cache).abs().mean())

        # if not im_size in self.rasterization_dict:
        # self.rasterization_dict[im_size] = {}
        # self.rasterization_dict[im_size]["points_idx"] = points_idx.detach().clone()
        # self.rasterization_dict[im_size]["dist"] = dist.detach().clone()
        # self.rasterization_dict[im_size]["pts3d"] = pts3D_c.detach().clone()
#         if os.environ["DEBUG"]:
#             print("Max dist: ", dist.max(), pow(radius, self.opts.rad_pow))

        dist = dist / pow(radius, self.rad_pow)

#         if os.environ["DEBUG"]:
#             print("Max dist: ", dist.max())

        alphas = (
            (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
            .pow(self.tau)
            .permute(0, 3, 1, 2)
        )

        # print(alphas.shape)
        # transmittance = torch.clamp(1 - torch.cumsum(alphas, 1), min=0, max=1)
        # # appending 1 to the first transmittance value
        # ones_transmittance = torch.ones_like(transmittance[:, :1])
        # transmittance = torch.cat([ones_transmittance, transmittance[:, :-1]], 1)
        # alphas_new = transmittance * alphas
        # alphas_new = alphas_new / (torch.sum(alphas_new, 1, keepdims=True) + 1e-8)
        # alphas = alphas_new
        # print(alphas_new.max(), alphas_new.min(), alphas_new.sum(1))
        # exit()

        # print(alphas.min(), alphas.max())

        if self.accumulation == 'alphacomposite':
            transformed_src_alphas = compositing.alpha_composite(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.accumulation == 'wsum':
            transformed_src_alphas = compositing.weighted_sum(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
        elif self.accumulation == 'wsumnorm':
            transformed_src_alphas = compositing.norm_weighted_sum(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )
            
        

        return transformed_src_alphas.to(torch.half)


SPLATTER = RasterizePointsXYsBlending()


def rotateAxis(degrees, axis):
    '''
    Function to rotate around given axis

    Input:
        degrees - scalar - Angle in degrees
        
        axis - scalar - options:
            0 - around x axis
            1 - around y axis
            2 - around z axis  
    
    Returns:
        Homogeneous rotation matrix
    '''

    radians = np.radians(degrees)

    if axis == 2: # z - axis

        rotation_mat = torch.tensor([[np.cos(radians), -np.sin(radians),           0,          0],
                                 [np.sin(radians),  np.cos(radians),           0,          0],
                                 [              0,                0,           1,          0],
                                 [              0,                0,           0,          1]])

    elif axis == 1: # y - axis

        rotation_mat = torch.tensor([[np.cos(radians),                0,  np.sin(radians),          0],
                                 [              0,                1,                0,          0],
                                 [-np.sin(radians),               0, np.cos(radians),          0],
                                 [              0,                0,                0,          1]])

    elif axis == 0: # x - axis


        rotation_mat = torch.tensor([[             1,                0,                0,          0],
                                [              0,  np.cos(radians), -np.sin(radians),          0],
                                [              0,  np.sin(radians),  np.cos(radians),          0], 
                                [              0,                0,                0,          1]])
    
    return rotation_mat

def translateMatrixFromVector(v):

    translation_matrix = torch.eye(4).type_as(v)
    # translation_matrix[0,3] += x
    # translation_matrix[1,3] += y
    # translation_matrix[2,3] += z
    translation_matrix[:3, 3] += v

    return translation_matrix#.type_as(torch.double)


def splatter_mesh(mesh, splatter, image_size):


    radius = float(1e-6) / float(2 * image_size)

    # print(pts3D.dtype, src.dtype)
    # pts3D = Pointclouds(points=pts3D, features=src.permute(0,2,1))

    im_size = int((image_size))



    p_to_face, zbuf, bary_coords, dists = rasterize_meshes(mesh, im_size, radius, splatter.points_per_pixel, perspective_correct=True)
    

    # print(p_to_face.shape, zbuf.shape, bary_coords.shape, dists.shape)

    frags = Fragments(pix_to_face=p_to_face, zbuf = zbuf, bary_coords=bary_coords, dists=dists)
    # print(frags.pix_to_face.shape, frags.zbuf.shape, frags.bary_coords.shape, frags.dists)
    # print(frags.pix_to_face.shape, frags.zbuf.shape, frags.bary_coords.shape, frags.dists.shape)

    # points_idx, _, dist = rasterize_points(
    #         pts3D,  im_size, radius, self.points_per_pixel
    #     )

    # print(mesh.num_faces_per_mesh())
    transformed_src_alphas = mesh.sample_textures(frags) # 1, 1, 1, F


    # print(transformed_src_alphas.shape)

    transformed_src_alphas = transformed_src_alphas[:, :, :, 0].permute(0, -1, 1, 2)
    # exit()

    # dist = dist / pow(radius, splatter.rad_pow)

    # alphas = (
    #     (1 - dist.clamp(max=1, min=1e-3).pow(0.5))
    #     .pow(splatter.tau)
    #     .permute(0, 3, 1, 2)
    # )

    # if splatter.accumulation == 'alphacomposite':
    #     transformed_src_alphas = compositing.alpha_composite(
    #         face_idx.permute(0, 3, 1, 2).long(),
    #         alphas,
    #         mesh.textures.sample_textures(face_idx).permute(1,0),
    #     )
    # elif splatter.accumulation == 'wsum':
    #     transformed_src_alphas = compositing.weighted_sum(
    #         face_idx.permute(0, 3, 1, 2).long(),
    #         alphas,
    #         pts3D.features_packed().permute(1,0),
    #     )
    # elif splatter.accumulation == 'wsumnorm':
    #     transformed_src_alphas = compositing.norm_weighted_sum(
    #         face_idx.permute(0, 3, 1, 2).long(),
    #         alphas,
    #         pts3D.features_packed().permute(1,0),
    #     )
        
    

    return transformed_src_alphas#.to(torch.half)





def get_coordinate_array(mask):
    # Get mapping from image pixels to points on mesh for faces

    # Filling with -1 for null in the beginning
    mapping = torch.ones_like(mask[0, 0]) * -1

    num_points = mapping[mask[0, 0] >= 0.5].shape[0]

    mapping[mask[0, 0] >= 0.5] = torch.linspace(0, num_points - 1, steps = num_points).type_as(mapping)#.to(torch.int64)
    
    # print(mapping.dtype, mapping[0, 0])
    mapping = mapping.long()

    return mapping

def get_indexing_grid(im_shape):


    h, w = im_shape

    x = torch.linspace(0, h-1, steps=h)
    y = torch.linspace(0, w-1, steps=w)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

    return grid_x.long(), grid_y.long() # shape h,w


def create_triangles(im2pcd_mapping, grid_x, grid_y):

    # print(grid_x.max(), grid_x.min(), im2pcd_mapping.shape, grid_y.min(), grid_y.max())

    # tl - tr
    # |
    # bl
    tl = im2pcd_mapping[grid_x, grid_y] # h, w
    tr = im2pcd_mapping[grid_x, grid_y + 1] # h, w
    bl = im2pcd_mapping[grid_x + 1, grid_y] # h, w
    # print(tl.shape, tr.shape, bl.shape, "tl")
    # exit()
    
    face_idx = torch.stack([tl, tr, bl], 0) # 3, h, w

    #       tr
    #       |
    # bl - br
    br = im2pcd_mapping[grid_x + 1, grid_y + 1] # h, w
    face_idx_2 = torch.stack([bl, tr, br], 0) # 3, h, w

    face_idx = face_idx.reshape(3, -1)
    face_idx_2 = face_idx_2.reshape(3, -1)

    face_ids = torch.cat([face_idx, face_idx_2], -1)
    # print(face_ids.shape, face_ids.min(), face_ids.max(), "before mask")

    min_face_ids = torch.min(face_ids, 0).values

    # Remove faces outside the mask
    face_ids = face_ids[:, min_face_ids > -1]
    return face_ids.transpose(0, 1)

def get_mesh(src_camera_coords, obj_mask):
    '''
    scc - b, 3, h, w
    obj_mask - 1, 1, h, w
    '''
    # print("running mesh operation")

    im2pcd_mapping = get_coordinate_array(obj_mask) # h, w shape

    src_camera_coords_pytorch3d = src_camera_coords.clone()
    # Converting from x-right, y-down, z-front to x-left, y-up, z-front for pytorch3d
    src_camera_coords_pytorch3d[:, :2] = -src_camera_coords[:, :2]



    h, w = obj_mask.shape[-2:]
    grid_x, grid_y = get_indexing_grid((h-1, w-1))



    pts = src_camera_coords_pytorch3d[0, :, obj_mask[0, 0] >= 0.5].reshape(3, -1).transpose(0, 1).to(src_camera_coords_pytorch3d.device).to(torch.float32) # V, 3
    verts_rgb = torch.ones_like(pts[None]).to(src_camera_coords_pytorch3d.device)[..., :1] # 1, V, 3
    face_ids = create_triangles(im2pcd_mapping, grid_x, grid_y).to(src_camera_coords_pytorch3d.device) # F, 3
    
    textures = TexturesVertex(verts_features=verts_rgb)

    # Create a Meshes object
    mesh = Meshes(
        verts=[pts],   
        faces=[face_ids],
        textures=textures
    )

    # print(face_ids.shape, face_ids.min(), face_ids.max())
    
    return mesh







def forward_splatting_pytorch3d_warp(tgt_image, depth, intrinsics, pose_transform, return_coordinates = False, obj_mask = None, return_mesh = False, return_projected_depth = False):


    depth = depth.float()
    intrinsics = intrinsics.type_as(depth)
    if obj_mask is not None:
        obj_mask = obj_mask.type_as(depth)
    src_camera_coords = pixel2cam(depth, intrinsics.inverse())
    pose_transform = pose_vec2mat(pose_transform).type_as(depth)


    # print(src_camera_coords.shape, pose_transform.shape, " forward splatting warp") # b, 3, h, w x b, 4, 4


    # Find the center of the object to bring it to the origin for performing the transformation
    b, c, h, w = src_camera_coords.shape
    cam_coords_pcd = src_camera_coords.reshape(b, c, -1)


    mask_pcd = obj_mask.reshape(1, 1, -1)
    obj_center = torch.mean(cam_coords_pcd[:, :, mask_pcd[0, 0] >= 0.5], -1)

    # print(obj_center.shape, obj_center)
    # translateMatrix()
    translate_transform = translateMatrixFromVector(-obj_center[0])
    translate_transform = translate_transform.repeat(b, 1, 1)
    

    pose_transform = translate_transform.inverse() @ pose_transform.type_as(depth) @ translate_transform

    src_cam_to_tgt_cam = pose_transform.float()


    # exit()
    tgt_cam_2_proj = src_cam_to_tgt_cam[:, :3, :] # Bx3x3 Bx3x4
    rot, tr = tgt_cam_2_proj[:,:,:3], tgt_cam_2_proj[:,:,-1:]
    # print(src_camera_coords.dtype, rot.dtype, tr.dtype, intrinsics.dtype)
    tgt_pix_coords = cam2pixel_vanilla(src_camera_coords, rot, tr, intrinsics, norm_scale = True, return_z=True) # b, h, w, 3

    
        
#     print(tgt_pix_coords.shape)
    

    tgt_image = tgt_image.type_as(tgt_pix_coords[0])
    
    b, h, w, _2 = tgt_pix_coords.shape
    _, f, _, _ = tgt_image.shape
    
    if return_mesh:
        mesh = get_mesh(tgt_pix_coords.permute(0, -1, 1, 2), obj_mask)
        projected_mask = splatter_mesh(mesh, SPLATTER, image_size=h)
        # print(projected_mask.shape, "projected mask")
#     print(depth.max(), depth.min())
    
#     print(tgt_pix_coords.shape, tgt_image.shape)
    
    tgt_pix_coords_in = tgt_pix_coords.reshape(b, h*w, -1)
    tgt_image_in = tgt_image.reshape(b, f, h*w)
    depth_in = depth.reshape(b, -1, h*w)
    
    
#     print(tgt_pix_coords_in.shape, tgt_image_in.shape)
    projected_img = SPLATTER(tgt_pix_coords_in, tgt_image_in)
    depth_projected = SPLATTER(tgt_pix_coords_in, depth_in)
#     valid_points = tgt_pix_coords_in
    valid_points = tgt_pix_coords[..., :2].abs().max(dim=-1)[0] <= 1
    
#     print(projected_img.shape, "splatted point cloud")
#     projected_img = softsplat.softsplat(tenIn=tgt_image, tenFlow=tgt_pix_coords.permute(0, -1, 1, 2), tenMetric=torch.exp(-depth / (depth.max() + 1e-8)), strMode="soft")
#     depth_projected = softsplat.softsplat(tenIn=depth, tenFlow=tgt_pix_coords.permute(0, -1, 1, 2), tenMetric=torch.exp(-depth / (depth.max() + 1e-8)), strMode="soft")
#     valid_points = tgt_pix_coords.abs().max(dim=-1)[0] <= 1
#     return projected_img, None, None

    # tgt_pix_coords_out = 
    # print(tgt_pix_coords.shape)
    # exit()
    if return_coordinates:
        if return_mesh:
            return projected_img, valid_points, depth_projected, tgt_pix_coords, projected_mask
        return projected_img, valid_points, depth_projected, tgt_pix_coords
    
    if return_mesh:
        
        return projected_img, valid_points, depth_projected, projected_mask
    return projected_img, valid_points, depth_projected


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, intrinsics):
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]

 



    # print(cam_coords_flat.shape, torch.mean(cam_coords_flat[0], -1))
    # input_coords = intrinsics @ cam_coords_flat # B, 3, H*W

    
    # X_in = input_coords[:, 0]
    # Y_in = input_coords[:, 1]
    # Z_in = input_coords[:, 2].clamp(min=1e-8)

    # X_in_norm = X_in / Z_in
    # Y_in_norm = Y_in / Z_in
    # Get mask for high depth coordinates
    # print(cam_coords_flat.shape, cam_coords_flat[:, -1])

    if proj_c2p_rot is not None:
        # print(proj_c2p_rot)
        pcoords = proj_c2p_rot.float() @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr.type_as(proj_c2p_rot)  # [B, 3, H*W]

    mask = (pcoords[:, -1] > 100.0)*1.0 # B, H * W
    # mask = (cam_coords_flat[:, -1] > 10.0)*1.0 # B, H * W
    

    # print(pcoords.shape, torch.mean(pcoords[0], -1))
    pcoords = intrinsics @ pcoords
    # print(pcoords.shape, torch.mean(pcoords[0], -1))

    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-8)



    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]


    global pixel_coords
    B, C, H, W = pixel_coords.shape
    pixel_coords_input = pixel_coords.reshape(-1, 3, H * W)[:, :2] # 1, 2, H * W
    X_in_norm = (pixel_coords_input[:, 0] * 2.0) / (w - 1) - 1.0
    Y_in_norm = (pixel_coords_input[:, 1] * 2.0) / (h - 1) - 1.0

    # print(X_in_norm.max(), X_in_norm.min(), Y_in_norm.max(), Y_in_norm.min())
    

    pixel_coords_f = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    pixel_in_coords = torch.stack([X_in_norm, Y_in_norm], dim=2)  # [B, H*W, 2]
    mask = mask[..., None]
    pixel_coords_f = mask * pixel_in_coords + (1.0 - mask) * pixel_coords_f

    # # print()
    delta_X = pixel_coords_f - pixel_in_coords

    delta_X = delta_X.permute(0, 2, 1 ) # B, 2, H*W
    delta_X = delta_X.reshape(1, 2, h, w)
    
    
    pixel_grid_y = ((pixel_coords_f[..., 1] + 1.0) / 2 * (h - 1)).to(torch.int64).clamp(min=0, max = h-1)
    pixel_grid_x = ((pixel_coords_f[..., 0] + 1.0) / 2 * (w - 1)).to(torch.int64).clamp(min=0, max=w-1)
    b_id = (torch.ones_like(pixel_grid_y) * 0).to(torch.int64)

    
    p_i_y = ((pixel_in_coords[..., 1] + 1.0) / 2 * (h - 1)).to(torch.int64).clamp(min=0, max = h-1)
    p_i_x = ((pixel_in_coords[..., 0] + 1.0) / 2 * (w - 1)).to(torch.int64).clamp(min=0, max = w-1)
    

    idx_sorted = torch.argsort(Z[0], descending=True)
    delta_new = delta_X.index_put_((b_id[0], torch.zeros_like(b_id[0]), pixel_grid_y[0][idx_sorted], pixel_grid_x[0][idx_sorted]), delta_X[b_id[0][idx_sorted], 0, p_i_y[0][idx_sorted], p_i_x[0][idx_sorted]])

    delta_new = delta_new.index_put_((b_id[0][idx_sorted], torch.ones_like(b_id[0]), pixel_grid_y[0][idx_sorted], pixel_grid_x[0][idx_sorted]), delta_X[b_id[0][idx_sorted], 1, p_i_y[0][idx_sorted], p_i_x[0][idx_sorted]])
    
    pixel_coords_f = pixel_coords_f.reshape(b, h, w, 2)

    delta_new = delta_new.permute(0, 2, 3, 1).reshape(1, h*w, 2) # 1, h*w, 2

    pixel_coords_f = pixel_in_coords - delta_new
    # print(Z[0].shape, "Z")

    
    # print(pixel_coords_f.shape)

    # (pixel_coords_f[..., 0] - 1) * ()

    # pixel_grid_y = ((pixel_coords_f[..., 1] + 1.0) / 2 * (h - 1)).to(torch.int64).clamp(min=0, max = h-1)
    # pixel_grid_x = ((pixel_coords_f[..., 0] + 1.0) / 2 * (w - 1)).to(torch.int64).clamp(min=0, max=w-1)
    # b_id = (torch.ones_like(pixel_grid_y) * 0).to(torch.int64)


    return pixel_coords_f.reshape(b,h,w,2)



def cam2pixel_vanilla(cam_coords, proj_c2p_rot, proj_c2p_tr, intrinsics, norm_scale = True, return_z = False):
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]



    if proj_c2p_rot is not None:
        # print(proj_c2p_rot)
        pcoords = proj_c2p_rot.float() @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr.type_as(proj_c2p_rot)  # [B, 3, H*W]

    mask = (pcoords[:, -1] > 100.0)*1.0 # B, H * W
    # mask = (cam_coords_flat[:, -1] > 10.0)*1.0 # B, H * W
    

    # print(pcoords.shape, torch.mean(pcoords[0], -1))
    pcoords = intrinsics @ pcoords
    # print(pcoords.shape, torch.mean(pcoords[0], -1))

    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)



    if norm_scale:
        
        X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    else:
        print("no norm scale")
        X_norm = (X / Z)  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        Y_norm = (Y / Z)  # Idem [B, H*W]


    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]


    if return_z:
        pixel_coords = torch.stack([X_norm, Y_norm, Z], dim=2)
        return pixel_coords.reshape(b,h,w,3)
    # print(pixel_coords.reshape(b,h,w,2).shape)
    return pixel_coords.reshape(b,h,w,2)

    


def forward_splatting_warp(tgt_image, depth, intrinsics, pose_transform, return_coordinates = False):

    src_camera_coords = pixel2cam(depth, intrinsics.inverse())
    pose_transform = pose_vec2mat(pose_transform).type_as(depth)
    intrinsics = intrinsics.type_as(depth)

    src_cam_to_tgt_cam = pose_transform

    tgt_cam_2_proj = src_cam_to_tgt_cam[:, :3, :] # Bx3x3 Bx3x4
    rot, tr = tgt_cam_2_proj[:,:,:3], tgt_cam_2_proj[:,:,-1:]
    tgt_pix_coords = cam2pixel_vanilla(src_camera_coords, rot, tr, intrinsics, norm_scale = True) # b, h, w, 2

    tgt_image = tgt_image.type_as(tgt_pix_coords[0])
    print(tgt_image.max(), tgt_image.min())
    print(depth.min(), depth.max())


    projected_img = softsplat.softsplat(tenIn=tgt_image.to(torch.float32), tenFlow=tgt_pix_coords.permute(0, -1, 1, 2).to(torch.float32), tenMetric=None, strMode="avg")
    depth_projected = softsplat.softsplat(tenIn=depth.to(torch.float32), tenFlow=tgt_pix_coords.permute(0, -1, 1, 2).to(torch.float32), tenMetric=None, strMode="avg")


    projected_img = projected_img.type_as(tgt_image)
    depth_projected = depth_projected.type_as(tgt_image)
    # projected_img = softsplat.softsplat(tenIn=tgt_image, tenFlow=tgt_pix_coords.permute(0, -1, 1, 2), tenMetric=torch.exp(-depth), strMode="soft")
    # depth_projected = softsplat.softsplat(tenIn=depth, tenFlow=tgt_pix_coords.permute(0, -1, 1, 2), tenMetric=torch.exp(-depth), strMode="soft")
    valid_points = tgt_pix_coords.abs().max(dim=-1)[0] <= 1
    if return_coordinates:
        return projected_img, valid_points, depth_projected, tgt_pix_coords
    
    return projected_img, valid_points, depth_projected



def quat2mat(quat):

    x, y, z, w = quat[:,0], quat[:,1], quat[:,2], quat[:,3]

    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    n = w2 + x2 + y2 + z2
    x = x / n
    y = y / n
    z = z / n
    w = w / n
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([1 - 2*y2 - 2*z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, 1 - 2*x2 - 2*z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, 1 - 2*x2 - 2*y2], dim=1).reshape(B, 3, 3)
    return rotMat

def pose_vec2mat(vec):

    size_list = list(vec.size())

    if len(size_list) == 3:
        # if dimension is [B 4 4] for multiview blender dataset

        return vec
    else:
        # If dimension is [B 7] for multiview nocs dataset
        b = vec.size(0)
        translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
        rot = vec[:,3:]
        rot_mat = quat2mat(rot)  # [B, 3, 3]

        invert_mat = torch.eye(4)
        invert_mat[0, 0] *= -1
        invert_mat[1, 1] *= -1

        # Adding 0.5 offset for dataset
        transform_mat = torch.cat([rot_mat,   (translation) + 0.5], dim=2)  # [B, 3, 4]
        transform_mat = torch.cat([transform_mat, torch.tensor([[0,0,0,1]]).unsqueeze(0).expand(1,1,4).type_as(transform_mat).repeat(b, 1, 1)], dim=1) # [B, 4, 4]
        return transform_mat @ invert_mat.type_as(transform_mat)


def set_id_grid(depth):
    global pixel_coords
    b, _, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1).type_as(depth)  # [1, 3, H, W]
    pixel_coords.to(depth.device)

def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    b, _, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    pixel_coords = pixel_coords.to(depth.device)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv.type_as(depth) @ current_pixel_coords.type_as(depth))
    cam_coords = cam_coords.reshape(b, 3, h, w)
    return cam_coords * depth


def pose_2_matrix(transformation_mat):
    '''
    Function to convert pose to transformation matrix
    '''
    flip_x = torch.eye(4)
    flip_x[2, 2] *= -1
    flip_x[1, 1] *= -1

    # views = pose.size(0)

    

    flip_x = flip_x.type_as(transformation_mat)

    # 180 degree rotation around x axis due to blender's coordinate system
    return  transformation_mat @ flip_x


def forward_warp(tgt_image, depth, intrinsics, pose_transform, return_coordinates = False):

    src_camera_coords = pixel2cam(depth, intrinsics.inverse())
    pose_transform = pose_vec2mat(pose_transform).type_as(depth)
    intrinsics = intrinsics.type_as(depth)

    src_cam_to_tgt_cam = pose_transform
    # print(src_cam_to_tgt_cam)
    tgt_cam_2_proj = src_cam_to_tgt_cam[:, :3, :] # Bx3x3 Bx3x4
    rot, tr = tgt_cam_2_proj[:,:,:3], tgt_cam_2_proj[:,:,-1:]
    tgt_pix_coords = cam2pixel(src_camera_coords, rot, tr, intrinsics, )

    # print(tgt_pix_coords.shape, tgt_pix_coords, "pix coords")

    tgt_image = tgt_image.type_as(tgt_pix_coords[0])

    # b_i, h_i, w_i = tgt_pix_coords
    # ones_ = torch.ones_like(b_i)
    # projected_image = tgt_image.index_input((b_i[0], 0 * ones_[0], h_i[0], w_i[0]), tgt_image[b_i[0], 0 * ones_[0], h_i[0], w_i[0]])
    projected_img = F.grid_sample(tgt_image, tgt_pix_coords, padding_mode='zeros', align_corners=True, mode="bilinear")
    depth_projected = F.grid_sample(depth, tgt_pix_coords, padding_mode='zeros', align_corners=False, mode="bilinear")
    valid_points = tgt_pix_coords.abs().max(dim=-1)[0] <= 1

    if return_coordinates:
        return projected_img, valid_points, depth_projected, tgt_pix_coords


    return projected_img, valid_points, depth_projected

# @torch.no_grad()
def warp_grid_edit(src, t_coords, padding_mode = None, mode = None, align_corners = False, depth=None, use_softsplat = True, splatting_radius = None, splatting_tau = None, splatting_points_per_pixel = None):

    if use_softsplat:

        if splatting_radius is not None:
            SPLATTER.radius = splatting_radius
        if splatting_tau is not None:
            SPLATTER.tau = splatting_tau
        if splatting_points_per_pixel is not None:
            SPLATTER.points_per_pixel = splatting_points_per_pixel
            
        store_device = str(src.device)
        if not str(src.device).startswith("cuda"):
            src = src.to("cuda")
            t_coords = t_coords.to("cuda")


        b, f, h, w = src.shape



        t_coords_in = t_coords.reshape(b, h*w, -1)#.float32()

        src_in = src.reshape(b, f, h*w)#.float32()
        out = SPLATTER(t_coords_in, src_in)

        if not store_device.startswith("cuda"):
            out = out.to(store_device)
    else:
        padd_mode = "zeros"
        m_mode = MODE
        if padding_mode is not None:
            padd_mode = padding_mode
        
        if mode is not None:
            m_mode = mode

        out = F.grid_sample(src, t_coords, padding_mode=padd_mode, align_corners=align_corners, mode=m_mode)

    return out
