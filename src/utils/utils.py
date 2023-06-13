import os
from typing import List
import matplotlib
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt, animation
import math
import cv2
import matplotlib.patches as mpatches
import tqdm
from matplotlib.widgets import Slider

def apply_cmap(img: np.ndarray, cmap_name: str = 'bone') -> np.ndarray:
    """ Apply a colormap to a 2D image. """
    cmap_function = matplotlib.colormaps[cmap_name]
    return cmap_function(img)

def translation(
        point: tuple[float, float, float],
        translation_vector: tuple[float, float, float]
        ) -> tuple[float, float, float]:
    """ Perform translation of `point` by `translation_vector`. """
    x, y, z = point
    v1, v2, v3 = translation_vector
    # Your code here
    # ...
    return (x+v1, y+v2, z+v3)

def multiply_quaternions(
        q1: tuple[float, float, float, float],
        q2: tuple[float, float, float, float]
        ) -> tuple[float, float, float, float]:
    """ Multiply two quaternions, expressed as (1, i, j, k). """
    # Your code here:
    #   ...
    return (
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    )


def conjugate_quaternion(
        q: tuple[float, float, float, float]
        ) -> tuple[float, float, float, float]:
    """ Multiply two quaternions, expressed as (1, i, j, k). """
    # Your code here:
    #   ...
    return (
        q[0], -q[1], -q[2], -q[3]
    )

def axial_rotation(
        point: tuple[float, float, float],
        angle_in_rads: float,
        axis_of_rotation: tuple[float, float, float]) -> tuple[float, float, float]:
    """ Perform axial rotation of `point` around `axis_of_rotation` by `angle_in_rads`. """
    x, y, z = point
    v1, v2, v3 = axis_of_rotation
    # Normalize axis of rotation to avoid restrictions on optimizer
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm
    # Your code here:
    #   ...
    #   Quaternion associated to point.
    p = (0, x, y, z)
    #   Quaternion associated to axial rotation.
    cos, sin = math.cos(angle_in_rads / 2), math.sin(angle_in_rads / 2)
    q = (cos, sin * v1, sin * v2, sin * v3)
    #   Quaternion associated to image point
    q_star = conjugate_quaternion(q)
    p_prime = multiply_quaternions(q, multiply_quaternions(p, q_star))
    #   Interpret as 3D point (i.e. drop first coordinate)
    return p_prime[1], p_prime[2], p_prime[3]

def center_crop(vol, dim):
    """Returns center cropped volume.
    Args:
    vol: volume to be center cropped
    dim: dimensions to be cropped
    """
    X, Y, Z = vol.shape

    # process crop width and height for max available dimension
    crop_x = dim[0] #if dim[0]<img.shape[1] else img.shape[1]
    crop_y = dim[1] #if dim[1]<img.shape[0] else img.shape[0]
    crop_z = dim[2] 
    mid_x, mid_y, mid_z = int(X/2), int(Y/2), int(Z/2)
    cx2, cy2, cz2 = int(crop_x/2), int(crop_y/2), int(crop_z/2)
    crop_vol = vol[mid_x-cx2:mid_x+cx2+1, mid_y-cy2:mid_y+cy2+1, mid_z-cz2:mid_z+cz2+1]
    return crop_vol

def center_pad(vol, dim):
    """Returns center aligned padded volume.
    Args:
        vol: volume to be center padded
        dim: dimensions to be padded
    """
    X, Y, Z = vol.shape
    pad_x = dim[0] if dim[0] >= X else X
    pad_y = dim[1] if dim[1] >= Y else Y
    pad_z = dim[2] if dim[2] >= Z else Z 

    if [pad_x, pad_y, pad_z] != dim:
        new_x = dim[0] if dim[0] < X else X
        new_y = dim[1] if dim[1] < Y else Y
        new_z = dim[2] if dim[2] < Z else Z
        vol = center_crop(vol, [new_x, new_y, new_z])
        X, Y, Z = vol.shape

    # Calculate the padding required for each dimension
    mid_x, mid_y, mid_z = int(pad_x/2), int(pad_y/2), int(pad_z/2)
    px, py, pz = int(X/2), int(Y/2), int(Z/2)
    
    

    print(f'DEBUG: {mid_x}-{mid_y}-{mid_z} AND {px}-{py}-{pz}')
    # Create a new black canvas with the larger dimensions (grayscale mode)
    larger_vol = np.zeros((pad_x, pad_y, pad_z), dtype=vol.dtype)

    # Paste the smaller image onto the larger image
    larger_vol[mid_x-px:mid_x+px+1, mid_y-py:mid_y+py+1, mid_z-pz:mid_z+pz] = vol
    return larger_vol

def pad_center(input_img, larger_height, larger_width):
    # Calculate the padding required for each dimension
    padding_width = (larger_width - input_img.shape[1]) // 2
    padding_height = (larger_height - input_img.shape[0]) // 2

    # Create a new black canvas with the larger dimensions (grayscale mode)
    larger_image = np.zeros((larger_height, larger_width), dtype=np.uint8)

    # Paste the smaller image onto the larger image
    larger_image[padding_height : padding_height + input_img.shape[0], padding_width : padding_width + input_img.shape[1]] = input_img
    return larger_image

def cv2_clipped_zoom(img, zoom_factor=0):

    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    ------
    Args:
        img : ndarray
            Image array
        zoom_factor : float
            amount of zoom as a ratio [0 to Inf). Default 0.
    ------
    Returns:
        result: ndarray
           numpy ndarray of the same shape of the input img zoomed by the specified factor.          
    """
    if zoom_factor == 0:
        return img


    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    
    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    
    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)
    
    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result

def alpha_fusion(img: np.ndarray, mask: np.ndarray, n_objects: int, object_colors: List, alpha: float=0.5)->np.ndarray:
    """ Visualize both image and mask in the same plot. """
    # TECHNIQUE: USING PLT COLMAPS
    # cmap = matplotlib.colormaps['bone']
    # cmap2 = matplotlib.colormaps['Set1']
    # norm = matplotlib.colors.Normalize(vmin=np.amin(img), vmax=np.amax(img))
    # fused_slice = \
    #     (1-alpha)*cmap(norm(img)) + \
    #     alpha*cmap2((mask/4))*mask[..., np.newaxis].astype('bool')

    # TECHNIQUE: USING RGB COLORING
    cmap = matplotlib.colormaps['bone']
    norm = matplotlib.colors.Normalize(vmin=np.amin(img), vmax=np.amax(img))
    col_mask = np.zeros(list(mask.shape)+[3])

    for k in range(n_objects):
        col_mask[mask==(k+1)] = object_colors[k]

    fused_slice = \
        (1-alpha)*cmap(norm(img))[..., :3] + \
        alpha*col_mask
    return (fused_slice * 255).astype('uint8')

def MIP_per_plane(img_dcm: np.ndarray, axis: int = 2) -> np.ndarray:
    """ Compute the maximum intensity projection on the defined orientation. """
    return np.max(img_dcm, axis=axis)

def visualize_MIP_per_plane(img_dcm: np.ndarray, pixel_len_mm: List):
    """ Creates an MIP visualize for each of the axis planes. """
    labels = ['Axial Plane', 'Coronal Plane', 'Sagittal Plane']
    ar = [(1,2),(0,2),(0,1)]
    for i in range(3):
        plt.imshow(MIP_per_plane(img_dcm, i), aspect=pixel_len_mm[ar[i][0]]/pixel_len_mm[ar[i][1]])
        plt.title(f'MIP for {labels[i]}')
        plt.show()

def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """
    return ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)

def create_3d_demo(projections: List, n: int, pixel_len_mm: List):
    """ creates an interactive demo to scroll through the projections. """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    current_index = 0
    current_image = projections[current_index]
    img_plot = ax.imshow(current_image, aspect=pixel_len_mm[0]/pixel_len_mm[1])

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])  # [left, bottom, width, height]
    slider = Slider(ax_slider, 'Rotation index', 0, n - 1, valinit=current_index, valstep=1)

    def update_image(val):
        current_index = int(val)
        current_image = projections[current_index]
        img_plot.set_data(current_image)
        fig.canvas.draw_idle()

    slider.on_changed(update_image) 
    plt.show()

def create_animation(img_dcm: np.ndarray, pixel_len_mm: List, labels: List, object_colors: List,
                     n=6, save_dir='results', show=True):
    """creates an animation by rotating the image on its sagittal plane"""
    # Create projections varying the angle of rotation
    #   Configure visualization colormap
    img_min = np.amin(img_dcm)
    img_max = np.amax(img_dcm)
    fig, _ = plt.subplots(figsize=(8, 10))
    #   Configure directory to save results
    os.makedirs(save_dir, exist_ok=True)
    #   Create projections
    projections = []
    #   Creating legend for figures
    patches = []
    for k in range(len(labels)):
        patches.append(mpatches.Patch(color=object_colors[k], label=labels[k]))

    for idx, alpha in tqdm.tqdm(enumerate(np.linspace(0, 360*(n-1)/n, num=n)), total=n):
        rotated_img = rotate_on_axial_plane(img_dcm, alpha)
        projection = MIP_per_plane(rotated_img)
        plt.imshow(projection, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0]/pixel_len_mm[1])
        legend = plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5))
        legend.set_title('Legend')
        plt.subplots_adjust(right=0.75)
        plt.savefig(os.path.join(save_dir, f'Projection_{idx}.png'), bbox_inches='tight')      # Save animation
        projections.append(projection)  # Save for later animation

    # Save and visualize animation
    animation_data = [
        [plt.imshow(img, animated=True, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0]/pixel_len_mm[1])]
        for img in projections
    ]
    legend = plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5))
    legend.set_title('Legend')
    anim = animation.ArtistAnimation(fig, animation_data,
                              interval=200, blit=True)
    anim.save(os.path.join(save_dir, 'Animation.gif'))  # Save animation
    if show:
        plt.show()                              # Show animation
    plt.close()
    create_3d_demo(projections, n, pixel_len_mm)
