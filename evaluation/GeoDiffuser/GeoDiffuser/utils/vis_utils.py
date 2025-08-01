import numpy as np
import cv2
# import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from PIL import Image
import json
from mpl_toolkits.mplot3d import Axes3D as mpl_3D
# import pptk
import glob
import os, argparse
from GeoDiffuser.utils.warp_utils import forward_warp, pose_2_matrix, forward_splatting_warp, forward_splatting_pytorch3d_warp
from PIL import Image
from scipy.spatial import KDTree

FLOAT_EPS = np.finfo(np.float16).eps


def show_image(image):

    plt.imshow(image)
    plt.show()


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

def translateMatrix(x, y, z):

    translation_matrix = torch.eye(4)
    translation_matrix[0,3] += x
    translation_matrix[1,3] += y
    translation_matrix[2,3] += z

    return translation_matrix#.type_as(torch.double)



def camera_matrix(focal_x, focal_y, c_x, c_y):
    '''
    Constructs camera matrix
    '''

    K = np.array([[ focal_x,      0,     c_x],
                  [     0,  focal_y,     c_y],
                  [     0,      0,       1]])

    return K
# Points generator
def get_grid(x, y):
    '''
    Get index grid from image
    '''
    # coords = np.indices((x, y)).reshape(2, -1)
    # return np.vstack((coords, np.ones(coords.shape[1])))

    y_i, x_i = np.indices((x, y))
    coords = np.stack([x_i, y_i, np.ones_like(x_i)], axis = -1).reshape(x*y, 3)
    # coords = np.indices((x, y)).reshape(2, -1)
    # return np.vstack((coords, np.ones(coords.shape[1])))
    # print(coords)
    return coords.T

def depth_decode(depth_image):

    # depth_image = np.array(depth_image)
    # # first 16 bits (first 2 channels) are 16-bit depth
    # R is the 8 LSB and G are the others
    depth_image_16 = depth_image[:,:,[1, 0]]
    # B are 8-bit version
    depth_image_8 = depth_image[:,:,2]
    # plt.imshow(depth_image_8)
    # plt.show()

    # last 8 are empty
    depth_single_channel = np.zeros((depth_image_16.shape[0], depth_image_16.shape[1]))
    # convert 16 bit to actual depth values
    for i in range(depth_single_channel.shape[0]):
        for j in range(depth_single_channel.shape[1]):
            bit_str = '{0:08b}'.format(depth_image_16[i, j, 0]) + '{0:08b}'.format(depth_image_16[i, j, 1])
            depth_single_channel[i, j] = int(bit_str, 2)

    depth_single_channel /= 1000
    # print(np.min(depth_single_channel))

    depth_single_channel
    depth_vector = depth_single_channel.reshape(1, -1)
    
    return depth_single_channel, depth_vector 

def extract_depth_tiff(depth_tiff):

    '''
    Extract depth from tiff image
    '''

    depth = np.array(depth_tiff)

    depth = 10 * (1 - depth)
    depth = depth.reshape(1, -1)
    return depth

def depth_2_point_cloud(invK, image, depth_image, depth_tiff=None):
    '''
    Convert depth map to point cloud

    '''
    points_hom = get_grid(image.shape[0], image.shape[1])
    


    if depth_tiff != None:
        print('tiff\n')
        depth = extract_depth_tiff(depth_tiff)
    else:
        depth = extract_depth(depth_image)
    # depth_map, depth = depth_decode(depth_image)
    
    # print(np.min(depth), np.max(depth[depth<30]))
    point_3D = invK @ points_hom
    point_3D = point_3D / point_3D[2, :]
    point_3D = point_3D * depth
    
    return point_3D
    
def save_point_cloud(K, image, mask, depth, num, output_directory, depth_tiff = None, max_depth = np.inf):
    '''
    Save point cloud given depth map
    '''

    directory = '%s/' % output_directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    image_colors = image.reshape(-1, 3)
    # print(image_colors.shape)
    invK = np.linalg.inv(K)
    # invK[0,0] *= 1
    # print(invK)
    point_cloud = depth_2_point_cloud(invK, image, depth, depth_tiff)   
    # point_cloud[0, :] *= -1
    mask = mask.reshape(-1, 1)
    mask = mask > 0.5
    # print(mask.shape)
    image_colors = image_colors[mask[:, 0], :]
    point_cloud = point_cloud[:, mask[:, 0]]
    # image_colors = image_colors[point_cloud[2,:] < 30, :]
    # point_cloud = point_cloud[:, point_cloud[2,:] < 30]
    
    image_colors = image_colors[point_cloud[2,:] < max_depth, :]
    point_cloud = point_cloud[:, point_cloud[2,:] < max_depth]
    

    # image_colors = image_colors[point_cloud[2,:] < 300, :]
    # point_cloud = point_cloud[:, point_cloud[2,:] < 300]
    
    # point_cloud[, :] = -point_cloud[2,:]
    # point_cloud[2, :] *= 10
    # print(np.min(point_cloud[2, :]), np.max(point_cloud[2,:]))
    # print(point_cloud.shape)
    
    
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point_cloud.T)
    # pcd.colors = o3d.utility.Vector3dVector(image_colors)
    # print(pcd.colors)
    # ply_name = "%s/frame_%06d_point_cloud.ply" % (output_directory, num)
    # o3d.io.write_point_cloud(ply_name, pcd)
    return point_cloud#, pcd

def extract_depth(depth_map):

    '''
    Get a depth vector from image
    '''

    depth = depth_map.reshape(1, -1)
    # depth = depth.max() - depth
    # depth = 30 - 30 * depth
    # depth = 1 / (depth + 0.001)
    
    return depth

def quat2mat(quat):
    
    x, y, z, w = quat
    Nq = w*w + x*x + y*y + z*z
    if Nq < FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])


def json_to_numpy(pose):
    pose = np.array([pose['position']['x'], 
                     pose['position']['y'], 
                     pose['position']['z'],
                     pose['rotation']['x'], 
                     pose['rotation']['y'], 
                     pose['rotation']['z'], 
                     pose['rotation']['w'] 
                     ])
    return pose
    
def pose_to_transformation(pose):

    '''
    Convert poses to transformation matrix
    '''


    temp_0 = pose[0]
    temp_1 = pose[1]
    temp_2 = pose[2]

    # quaternions
    temp_x = pose[3]
    temp_y = pose[4]
    temp_z = pose[5]
    temp_w = pose[6]

    pose[4:6] *= -1
    pose[0] *= -1

    # print(pose)
    rot_mat = quat2mat(pose[3:])
    translation_vector = np.array([[pose[0]], [pose[1]], [pose[2]]]) / 1000  
    print(translation_vector)
    # translation_offset = np.array([[2.25], [-1.25], [0.5]])
    # translation_offset = np.array([[0.0], [-0.5], [0.0]])
    
    rot_mat_2 = np.array([[ 0,  1, 0, 0],
                          [-1,  0, 0, 0],
                          [ 0,  0, 1, 0],
                          [ 0,  0, 0, 1]])
    
    flip_x = np.eye(4)
    flip_x[0, 0] *= -1

    trans = flip_x @ rot_mat_2
    translation_offset = np.ones((3,1)) * 1


    # different transformation matrix
    transformation_mat = np.vstack((np.hstack((rot_mat, translation_vector + 0.5 ) ), np.array([0, 0, 0, 1]))) 
    
    # transformation_mat = np.vstack((np.hstack((rot_mat.T,  rot_mat.T @ translation_vector)), np.array([0, 0, 0, 1])))
    
    # transformation_mat = np.vstack((np.hstack((rot_mat.T,   (translation_vector) )), np.array([0, 0, 0, 1])))
    
    # translation_offset = -np.array([[1.0], [1.0], [2.5]])
    
    # transformation_mat = np.vstack((np.hstack((rot_mat.T,  rot_mat.T @ (translation_offset)  + translation_vector)), np.array([0, 0, 0, 1])))
    
    print(transformation_mat.shape)
    return transformation_mat @ trans



# def custom_draw_geometry(pcd):

#     vis = o3d.visualization.Visualizer()
    
#     vis.create_window()
#     pcd_origin = o3d.geometry.PointCloud()
#     pcd_origin.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0],
#                                                              [0, 0, 1],
#                                                              [1, 0, 0]]))
#     pcd_origin.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
#     vis.add_geometry(pcd)
#     vis.add_geometry(pcd_origin)
    
#     ctr = vis.get_view_control()
#     print(ctr)
#     vis.run()

def read_json_file(path):
    '''
    Read json file
    '''
    json_data = []

    with open(path) as fp:
        for json_object in fp:
            json_data.append(json.loads(json_object))
    
    return json_data


def convert_tensor_to_numpy(t):

    return t.detach().cpu().numpy()

def read_depth(path, size = None):

    d = cv2.imread(path,  -1)

    if size is not None:
        d = np.array(Image.fromarray(d).resize((size[0], size[1])))

    return d

def preprocess_depth(depth):

    depth = depth.max() - depth
    depth = depth / depth.max()
    depth[depth > 0.8] = 1000000.0

    return depth



class CoordinateDistances:

    def __init__(self):
        self.coord_distance_dict = {}
        self.theta = torch.eye(3)[:2][None]
    
    @torch.no_grad()
    def get_coord_distance(self, size, device="cuda"):
        if size not in self.coord_distance_dict:
            grid = torch.nn.functional.affine_grid(self.theta, (1, 1, size, size), align_corners=None)# 1, H, W, 2
            # print(grid.shape)
            d = grid.reshape(1, -1, 2)
            # print(d.shape)
            dist = torch.sqrt(torch.sum(torch.square(d[:, :, None] - d[:, None]), -1) + 1e-12).to(device) # 1, hw, hw

            self.coord_distance_dict[size] = dist

        return self.coord_distance_dict[size]

coord_distance_depth = CoordinateDistances()

def fill_background_depth(depth, mask_in, device = "cuda"):


    # Borrowed from - https://stackoverflow.com/questions/3662361/fill-in-missing-values-with-nearest-neighbour-in-python-numpy-masked-arrays
    x,y=np.mgrid[0:depth.shape[0],0:depth.shape[1]]
    mask = (mask_in[0, 0] < 0.5)

    xygood = np.array((x[~mask],y[~mask])).T
    xybad = np.array((x[mask],y[mask])).T

    # print(depth)
    depth[mask] = depth[~mask][KDTree(xygood).query(xybad)[1]]
    # print(depth)



    

    return depth
    # exit()

def get_transform_coordinates(image, depth, obj_mask = None, transform_in = torch.eye(4), use_softsplat = True, focal_length = 550, return_mesh=False):

    K = camera_matrix(focal_length, focal_length, image.shape[1] / 2.0, image.shape[0] / 2.0)
    # print(depth.min(), depth.max())
    # depth = depth.max() - depth
    # Constant depth case
    if np.sum(depth) == 0.5 * (depth.shape[0] * depth.shape[1]):
        depth = np.ones_like(depth) * 0.5

    else:
        # Normalize depth
        depth = depth / (depth.max() + 1e-8)
        # making sure that the closest point is far enough from the camera to rotate without causing smearing
        # depth = depth
        depth[depth > 0.95] = 1.0
        # depth = 1.0 / depth
        # print(depth.min(), depth.max())
        # depth = depth * 10

    mask = (depth < 0.95) * 1.0

    if obj_mask is not None:
        mask = obj_mask * mask


    mean_depth = np.mean(depth[mask >= 0.5])
    mask_torch = (torch.tensor(mask)[None, None] >= 0.5) * 1.0

    # depth = fill_background_depth(depth, mask_torch)

    # print(depth[mask >= 0.5].min(), depth[mask >= 0.5].max(), " min max")
    # print(mean_depth, depth.shape, " mean")

    # degrees = args.degrees
    # axis = args.axis

    # pose_transform = rotateAxis(degrees, 1)[None] @ rotateAxis(degrees, 2)[None]

    # pose_transform = rotateAxis(degrees, axis)[None]

    # pose_transform_2 = rotateAxis(0, 0)[None]

    # translate_transform = translateMatrix(0.0, 0.0, -mean_depth)
    
    # translate_transform_2 = translateMatrix(0, 0.0, 0).type_as(translate_transform)
    
    # # .type_as(translate_transform)

    # # pose_transform = translate_transform.inverse() @ pose_transform @ translate_transform
    # pose_transform = translate_transform.inverse() @ transform_in.type_as(translate_transform) @ translate_transform
    # print(transform_in, pose_transform)
    # print("Pose transform: \n ", pose_transform)

    pose_transform = transform_in
    image = torch.from_numpy(image)[None].permute(0, 3, 1, 2)

    with torch.no_grad():
        if use_softsplat:

            if return_mesh:
                projected_image, valid_points, depth_projected, transform_coordinates, mask_projected_mesh = forward_splatting_pytorch3d_warp(image.to("cuda"), torch.from_numpy(depth)[None][None].to("cuda"), torch.from_numpy(K)[None].to("cuda"), pose_transform[None].to("cuda"), return_coordinates=True, obj_mask = mask_torch, return_mesh=True)
            else:
                projected_image, valid_points, depth_projected, transform_coordinates = forward_splatting_pytorch3d_warp(image.to("cuda"), torch.from_numpy(depth)[None][None].to("cuda"), torch.from_numpy(K)[None].to("cuda"), pose_transform[None].to("cuda"), return_coordinates=True, obj_mask = mask_torch, return_mesh=False)

        else:
            projected_image, valid_points, depth_projected, transform_coordinates = forward_warp(image, torch.from_numpy(depth)[None][None], torch.from_numpy(K)[None], pose_transform[None], return_coordinates=True)


    # # print(projected_image.shape, valid_points.shape)
    projected_image = projected_image * valid_points[:, None]
    np_image = np.clip(convert_tensor_to_numpy(projected_image[0].permute(1, 2, 0)), 0, 1)
    t_coords = convert_tensor_to_numpy(transform_coordinates[0])

    if return_mesh:
        return t_coords, np_image, convert_tensor_to_numpy(mask_projected_mesh)
    return t_coords, np_image



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--input_depth", type=str)
    parser.add_argument("-im", "--input_image", type=str)
    parser.add_argument("-o", "--output_directory", type=str, default="./outputs/debug_outputs/")
    parser.add_argument("-d", "--degrees", type=float, default=0.0)
    parser.add_argument("-a", "--axis", type=int, default=0)
    parser.add_argument("-m", "--mask", type=str, default=None)


    args = parser.parse_args()

    


    image = cv2.imread(args.input_image, -1) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0


        
    if args.input_depth.split(".")[-1] == "npz":
        depth = np.load(args.input_depth)["arr_0"]
        # [734.9394386120183, 734.9394386120183, 639.5, 359.5]
        K = camera_matrix(734.9394386120183, 734.9394386120183, 639.5, 359.5)
        if args.mask is not None:
            mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE) / 255.0
            print(mask.min(), mask.max())
            image = image * mask[..., None]
            depth = depth * mask
        
        depth = depth / (depth.max() + 1e-8)
        depth[depth==0] = 1000000.0
        depth = depth.astype(np.float32)   
    else:
        depth = cv2.imread(args.input_depth,  -1)
        K = camera_matrix(550, 550, image.shape[1] / 2.0, image.shape[0] / 2.0)



        print(depth.min(), depth.max())
        depth = depth.max() - depth
        depth = depth / depth.max()
        depth[depth > 0.8] = 1000000.0
        # depth = 1.0 / depth
        print(depth.min(), depth.max())
    # depth = depth / depth.max() + 1e-8
    # show_image(depth / depth.max())
    # K = camera_matrix(550, 550, image.shape[1] / 2.0, image.shape[0] / 2.0)
    mask = (depth < 100.0) * 1.0
    
    output_directory_name = os.path.join(args.output_directory, "")
    # transform = torch.eye(4)


    # SAVE POINT CLOUD
    point_cloud = save_point_cloud(K,image, mask, depth, 1, output_directory_name, max_depth=np.inf)
    # o3d.visualization.draw_geometries([pcd])


    # print(np.mean(depth))


    # print(depth[0])
    # plt.imshow(depth)
    # plt.show()
    if args.input_depth.split(".")[-1] == "npz" and args.mask is not None:
        mean_depth = np.mean(depth[(depth > 0.001) * (depth < 1.0)])
    else:
        mean_depth = np.mean(depth[depth < 1.0])

    print(mean_depth)
    degrees = args.degrees
    axis = args.axis

    # pose_transform = rotateAxis(degrees, 1)[None] @ rotateAxis(degrees, 2)[None]

    pose_transform = rotateAxis(degrees, axis)[None]

    pose_transform_2 = rotateAxis(0, 0)[None]

    translate_transform = translateMatrix(0.0, 0.0, -mean_depth).type_as(pose_transform)
    
    translate_transform_2 = translateMatrix(0, 0.0, 0).type_as(pose_transform)
    
    # .type_as(translate_transform)

    pose_transform =  translate_transform_2 @ translate_transform.inverse() @ pose_transform_2 @ pose_transform @ translate_transform

    image = torch.from_numpy(image)[None].permute(0, 3, 1, 2)

    with torch.no_grad():
        projected_image, valid_points, depth_projected, transform_coordinates = forward_warp(image, torch.from_numpy(depth)[None][None], torch.from_numpy(K)[None], pose_transform, return_coordinates=True)


    # # print(projected_image.shape, valid_points.shape)
    projected_image = projected_image * valid_points[:, None]
    np_image = np.clip(convert_tensor_to_numpy(projected_image[0].permute(1, 2, 0)), 0, 1)

    t_coords = convert_tensor_to_numpy(transform_coordinates[0])
    # convert_tensor_to_numpy(projected_image[0].permute()
    np_depth = convert_tensor_to_numpy(depth_projected[0, 0])


    args.output_directory = os.path.join(args.output_directory,"")
    os.makedirs(args.output_directory, exist_ok=True)

    od = args.output_directory
    print(t_coords.shape)

    plt.imsave(od + "projected_image.png", np_image)
    np.save(od + "transform_coordinates.npy", t_coords)
    np.save(od + "depth.npy", np_depth)
    plt.imsave(od + "input.png", convert_tensor_to_numpy(image[0].permute(1, 2, 0)))
    np.save(od + "transform.npy", convert_tensor_to_numpy(pose_transform)[0])
    # plt.imsave()
    # print(np_image.shape)
    # plt.imshow(np_image)
    # plt.show()



