import os
from typing import List
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage.morphology import binary_dilation, binary_erosion
from utils.utils import center_crop, center_pad, create_3d_demo
from scipy.optimize import least_squares
import cv2
from scipy.ndimage import zoom, rotate, shift



def get_thalamus_mask(img_atlas: np.ndarray) -> np.ndarray:
    # Your code here:
    #   ...
    thalamus_mask = np.zeros_like(img_atlas)
    thalamus_mask[img_atlas>=121] = 1
    thalamus_mask[img_atlas>150] = 0
    thalamus_mask = binary_erosion(thalamus_mask, np.ones((3,3,3)))
    thalamus_mask = binary_dilation(thalamus_mask, np.ones((3,3,3)))

    assert np.sum(thalamus_mask>0)
    return thalamus_mask


def find_centroid(mask: np.ndarray) -> np.ndarray:
    # Your code here:
    #   Consider using `np.where` to find the indices of the voxels in the mask
    #   ...
    idcs = np.where(mask == 1)
    centroid = np.stack([
        np.mean(idcs[0]),
        np.mean(idcs[1]),
        np.mean(idcs[2]),
    ])
    print(centroid)
    return centroid


def visualize_axial_slice(
        img: np.ndarray,
        mask: np.ndarray,
        mask_centroid: np.ndarray,
        pixel_len_mm: List
        ):
    """ Visualize the axial slice (firs dim.) of a single region with alpha fusion. """
    # Your code here
    #   Remember `matplotlib.colormaps['cmap_name'](...)`
    #   See also `matplotlib.colors.Normalize(vmin=..., vmax=...)`
    #   ...
    img_slice = img[mask_centroid[0].astype(np.int32), :, :]
    mask_slice = mask[mask_centroid[0].astype(np.int32), :, :]

    cmap = matplotlib.colormaps['bone']
    norm = matplotlib.colors.Normalize(vmin=np.amin(img_slice), vmax=np.amax(img_slice))
    fused_slice = \
        0.75*cmap(norm(img_slice))[..., :3] + \
        0.25*np.stack([mask_slice, np.zeros_like(mask_slice), np.zeros_like(mask_slice)], axis=-1)
    plt.imshow(fused_slice, aspect=pixel_len_mm[1]/pixel_len_mm[2])
    plt.title('Axial visualization of Volume Centroid')
    plt.show()

def visualize_coronal_slice(
        img: np.ndarray,
        mask: np.ndarray,
        mask_centroid: np.ndarray,
        pixel_len_mm: List
        ):
    """ Visualize the axial slice (firs dim.) of a single region with alpha fusion. """
    # Your code here
    #   Remember `matplotlib.colormaps['cmap_name'](...)`
    #   See also `matplotlib.colors.Normalize(vmin=..., vmax=...)`
    #   ...
    img_slice = img[:, mask_centroid[1].astype(np.int32), :]
    mask_slice = mask[:, mask_centroid[1].astype(np.int32), :]

    cmap = matplotlib.colormaps['bone']
    norm = matplotlib.colors.Normalize(vmin=np.amin(img_slice), vmax=np.amax(img_slice))
    fused_slice = \
        0.75*cmap(norm(img_slice))[..., :3] + \
        0.25*np.stack([mask_slice, np.zeros_like(mask_slice), np.zeros_like(mask_slice)], axis=-1)
    fused_slice = np.flip(fused_slice, axis=0)
    plt.imshow(fused_slice, aspect=pixel_len_mm[0]/pixel_len_mm[2])
    plt.title('Coronal visualization of Volume Centroid')
    plt.show()

def visualize_sagittal_slice(
        img: np.ndarray,
        mask: np.ndarray,
        mask_centroid: np.ndarray,
        pixel_len_mm: List
        ):
    """ Visualize the axial slice (last dim.) of a single region with alpha fusion. """
    img_slice = img[:,:,mask_centroid[2].astype(np.int32)]
    mask_slice = mask[:,:,mask_centroid[2].astype(np.int32)]

    cmap = matplotlib.colormaps['bone']
    norm = matplotlib.colors.Normalize(vmin=np.amin(img_slice), vmax=np.amax(img_slice))
    fused_slice = \
        0.75*cmap(norm(img_slice))[..., :3] + \
        0.25*np.stack([mask_slice, np.zeros_like(mask_slice), np.zeros_like(mask_slice)], axis=-1)
    fused_slice = np.flip(fused_slice, axis=0)
    plt.imshow(fused_slice, aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.title('Sagittal visualization of Volume Centroid')
    plt.show()

def rigid_transformation(volume: np.ndarray, parameters: tuple[float, ...], ref_volume: np.ndarray, angle_in_rads: int = 360):
    """ Apply to `volume` a translation followed by an axial rotation and scaling, defined by `parameters`. """
    t1, t2, t3, v1, v2, v3, s1, s2, s3 = parameters
    # Scale to original values
    t1, t2, t3 = np.array([t1, t2, t3])*40 - 20
    s1, s2, s3 = np.array([s1, s2, s3])*0.1 + 0.9

    # apply transformation
    t_volume = shift(volume,(t1, t2, t3))
    t_volume = rotate(t_volume, angle_in_rads * v3, (0, 1))
    t_volume = rotate(t_volume, angle_in_rads * v1, (1, 2))
    t_volume = rotate(t_volume, angle_in_rads * v2, (2, 0))
    t_volume = zoom(t_volume, (s1, s2, s3))

    # crop from the center to have the same volume as reference
    t_volume = center_crop(t_volume, ref_volume.shape)
    return t_volume

def inverse_rigid_transformation(volume: np.ndarray, parameters: tuple[float, ...], ref_volume: np.ndarray , angle_in_rads: int = 360):
    """ Apply the inverse transformation to input volume by inferring from defined `parameters`. """
    t1, t2, t3, v1, v2, v3, s1, s2, s3 = parameters
    t1, t2, t3 = np.array([t1, t2, t3])*40 - 20
    s1, s2, s3 = np.array([s1, s2, s3])*0.1 + 0.9

    t_volume = rotate(volume, -(angle_in_rads * v3), (0, 1))
    t_volume = rotate(t_volume, -(angle_in_rads * v1), (1, 2))
    t_volume = rotate(t_volume, -(angle_in_rads * v2), (2, 0))
    t_volume = shift(t_volume, (-t1, -t2, -t3))
    t_volume = zoom(t_volume, (1/s1, 1/s2, 1/s3))

    # pad to the center to have the same volume as reference
    t_volume = center_pad(t_volume, ref_volume.shape)
    return t_volume

def mutual_information(img_input: np.ndarray, img_reference) -> np.ndarray:
    """ Compute the Shannon Mutual Information between two images. """
    nbins = [10, 10]

    # Compute entropy of each image
    hist = np.histogram(img_input.ravel(), bins=nbins[0])[0]
    prob_distr = hist / np.sum(hist)
    entropy_input = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))
    hist = np.histogram(img_reference.ravel(), bins=nbins[0])[0]
    prob_distr = hist / np.sum(hist)
    entropy_reference = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))

    # Compute joint entropy
    joint_hist = np.histogram2d(img_input.ravel(), img_reference.ravel(), bins=nbins)[0]
    prob_distr = joint_hist / np.sum(joint_hist)
    joint_entropy = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))

    # Compute mutual information
    return entropy_input + entropy_reference - joint_entropy

def vector_of_residuals(ref_points: np.ndarray, inp_points: np.ndarray) -> np.ndarray:
    """ Given 3d input and reference volume, compute vector of residuals using MI. """
    error = []
    for b, a in zip(ref_points, inp_points):
        error.append(mutual_information(a, b))
    return np.mean(error)

def coregister_landmarks(ref_landmarks: np.ndarray, inp_landmarks: np.ndarray):
    """ Coregister two sets of landmarks using a rigid transformation. """
    initial_parameters = [
        0.25, 0.5, 0.4285,  # Translation vector, without scaling range [-20, 20]
        0.4875, 0, 0.005,   # Axis of rotation, without scaling range   [0, 360]
        0.6, 1, .9          # Scaling parameters, without scaling range [0.9, 1]
    ]

    def function_to_minimize(parameters):
        """ Transform input landmarks, then compare with reference landmarks."""
        inp_landmarks_transf = rigid_transformation(inp_landmarks, parameters, ref_landmarks)
        return vector_of_residuals(ref_landmarks, inp_landmarks_transf)

    # Apply least squares optimization
    result = least_squares(
        function_to_minimize,
        x0=initial_parameters,
        bounds=([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        verbose=2)
    return result
    # minimize(
    #     function_to_minimize,
    #     x0=initial_parameters,
    #     bounds=([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    #     method='CG',
    #     options={'disp': True})
    # return result

def find_region_volume(region_mask):
    """ Returns the volume of the region in mm^3. """
    # Your code here:
    #   ...
    return np.sum(region_mask)

def min_max_normalize(vol: np.ndarray):
    amin = np.amin(vol)
    amax = np.amax(vol)
    norm_vol = (vol - amin) / (amax - amin)
    return norm_vol

def find_region_surface(mask):
    """ Returns the surface of the region in mm^2. """
    # Your code here:
    #   See `skimage.morphology.binary_erosion()` and `skimage.morphology.binary_dilation()`
    #   ...
    inner_surface = mask - binary_erosion(mask, np.ones((3, 3, 3)))
    outer_surface = binary_dilation(mask, np.ones((3, 3, 3))) - mask
    return (np.sum(inner_surface) + np.sum(outer_surface) ) / 2     # Average of inner and outer surface

def main():
    # Load data
    phantom_dir = 'data/Coregistration/icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm'
    atlas_dir = 'data/Coregistration/AAL3_1mm.dcm'
    tgt_dir = 'data/Coregistration/RM_Brain_3D-SPGR/'
    dcm_phantom = pydicom.dcmread(phantom_dir)
    img_phantom = dcm_phantom.pixel_array
    print('P', img_phantom.shape)
    img_phantom = img_phantom[6:-6, 6:-6, 6:-6]     # Crop phantom to atlas size
    
    # dcm_atlas = pydicom.dcmread(os.path.join(data_dir, os.listdir(data_dir)[0]))
    dcm_atlas = pydicom.dcmread(atlas_dir)
    img_atlas = dcm_atlas.pixel_array
    print("A", img_atlas.shape)

    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(img_phantom[len(img_phantom) // 2, :, :], cmap='bone')
    # axs[1].imshow(img_atlas[len(img_phantom) // 2, :, :], cmap='tab20')
    # plt.show()
    # return
    assert img_phantom.shape == img_atlas.shape
    count = 0
    order_array = []
    dcm_slices = []
    for f in os.listdir(tgt_dir):
        img = os.path.join(tgt_dir, f)
        ct_dcm = pydicom.dcmread(img)
        if hasattr(ct_dcm, 'ImagePositionPatient'):
            dcm_slices.append(ct_dcm)
        else:
            count+=1
    print(f"skip count: {count}")
    dcm_slices = sorted(dcm_slices, key=lambda x: x.ImagePositionPatient[-1])
    zoom_factor = dcm_slices[0].PixelSpacing[0]
    pixel_len_mm = [dcm_slices[0].SpacingBetweenSlices]
    pixel_len_mm.extend(dcm_slices[0].PixelSpacing)

    img_phantom = min_max_normalize(img_phantom)

    dim1 = int(dcm_slices[0].pixel_array.shape[0] * zoom_factor)
    dim2 = int(dcm_slices[0].pixel_array.shape[1] * zoom_factor)
    scan_3d_pspace = np.array([x.pixel_array for x in dcm_slices])
    scan_3d = np.array([cv2.resize(x.pixel_array, [dim1, dim2]) for x in dcm_slices])
    scan_3d = min_max_normalize(scan_3d)
    parameters = [2.51809783e-01, 5.00640507e-01, 4.27904346e-01, 4.83209821e-01, 8.26430720e-05, 1.14487524e-02, 6.00000000e-01, 1.00000000e+00,  9.00000000e-01]
    # parameters =    [
    #     0.25, 0.5, 0.4285,  # Translation vector, without scaling range [-20, 20]
    #     0.4875, 0, 0.005,   # Axis of rotation, without scaling range   [0, 360]
    #     0.6, 1, .9          # Scaling parameters, without scaling range [0.9, 1]
    # ]
    #
    # [-5.10347310e+00,  4.90641158e+00, -2.01109179e+00,  1.80210520e+02,9.99988819e-01,  9.69120554e-11,  6.81191108e-03,  9.00000000e-01,  9.50000000e-01, 9.75000000e-01]
    # # [-5, 5, -2, 180, 1, 0, 0, 0.9, 0.95, 0.975]
    # # [-5.14178794, 4.54777266, -1.66039817, 180.8684099, 0.92519374, 0.23853952, 0.31502524, 0.9, 0.95, 0.975]
    # # 
    # scan_3d = rigid_transformation(scan_3d, parameters, img_phantom)
    # scan_3d_pspace = img_phantom
    # img_atlas_pspace = img_atlas
    # plt.subplot(131)
    # plt.imshow(scan_3d_pspace[100,:,:], cmap='bone')
    # plt.title("axial plane")

    # plt.subplot(132)
    # plt.imshow(scan_3d_pspace[:,100,:], cmap='bone')
    # plt.title("coronal plane")

    # plt.subplot(133)
    # plt.imshow(scan_3d_pspace[:,:,100], cmap='bone')
    # plt.title("sagittal plane")
    # plt.show()
    # return

    # plt.subplot(121)
    # plt.imshow(scan_3d[len(img_phantom) // 2,:,:], cmap='bone')
    # plt.subplot(122)
    # plt.imshow(img_atlas[len(img_phantom) // 2,:,:], cmap='tab20')
    # plt.title("Axial plane")
    # plt.show()
    # return

    dimp1 = int(img_atlas.shape[2] * (1 / zoom_factor))
    dimp2 = int(img_atlas.shape[1] * (1 / zoom_factor))
    img_atlas_pspace = np.array([zoom(x, (1 / zoom_factor)) for x in img_atlas])
    img_atlas_pspace = np.array([cv2.resize(x, [dimp1, dimp2], cv2.INTER_NEAREST_EXACT) for x in img_atlas])
    img_atlas_pspace = img_atlas_pspace[4:-4,:,:]
    print('PSPACE:',img_atlas_pspace.shape)
    print('PSPACE:',scan_3d_pspace.shape)
    img_atlas_pspace = inverse_rigid_transformation(img_atlas_pspace, parameters, scan_3d_pspace)
    # print(vector_of_residuals(scan_3d, img_phantom))


    # print(scan_3d.shape)
    # print(img_phantom.shape)
    # Coregister landmarks
    # result = coregister_landmarks(img_phantom, scan_3d)
    # solution_found = result.x
    # print(solution_found)
    # with open('optimzation_results.txt', 'w') as f:
    #     f.write(f'Param: {solution_found}')
    #     f.write(f'\n\nReport:\n\
    #             Fun: {result.fun}\n\
    #                 Optimality: {result.optimality}\n\
    #                     NIT: {result.nit}\n\
    #                         nfev: {result.nfev}\n\
    #                             njev: {result.njev}\n\
    #                                 Message: {result.message}')


    # print(scan_3d_pspace.shape)
    # print(img_atlas_pspace.shape)
    # plt.subplot(121)
    # plt.imshow(scan_3d_pspace[100,:,:])
    # plt.subplot(122)
    # plt.imshow(img_atlas_pspace[100,:,:])
    # plt.title("axial plane")
    # plt.show()

    # plt.subplot(121)
    # plt.imshow(scan_3d_pspace[:,100,:], aspect=1/zoom_factor)
    # plt.subplot(122)
    # plt.imshow(img_atlas_pspace[:,100,:], aspect=1/zoom_factor)
    # plt.title("coronal plane")
    # plt.show()

    # plt.subplot(121)
    # plt.imshow(scan_3d_pspace[:,:,100], aspect=1/zoom_factor)
    # plt.subplot(122)
    # plt.imshow(img_atlas_pspace[:,:,100], aspect=1/zoom_factor)
    # plt.title("sagittal plane")
    # plt.show()

    # for i in range(10):
    #     j=i*5
    #     fig, axs = plt.subplots(1, 2)
    #     axs[0].imshow(img_phantom[:, -j, :], cmap='bone')
    #     axs[1].imshow(scan_3d[:, -j, :], cmap='bone')
    #     fig.show()
    #     plt.show()

    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(img_phantom[50, :, :], cmap='bone')
    # axs[1].imshow(img_atlas[50, :, :], cmap='tab20')
    # fig.show()
    # plt.show()

    thalamus_mask = get_thalamus_mask(img_atlas_pspace)
    # mask_centroid = find_centroid(thalamus_mask) #np.array([40.0, 150.0, 100.0]) 
    # visualize_axial_slice(scan_3d_pspace, thalamus_mask, mask_centroid, pixel_len_mm)
    # visualize_coronal_slice(scan_3d_pspace, thalamus_mask, mask_centroid, pixel_len_mm)
    # visualize_sagittal_slice(scan_3d_pspace, thalamus_mask, mask_centroid, pixel_len_mm)

    # scan_3d = rigid_transformation(scan_3d, parameters, img_atlas)
    # scan_3d = img_phantom
    # thalamus_mask = get_thalamus_mask(img_atlas)
    # mask_centroid = find_centroid(thalamus_mask)
    # visualize_axial_slice(scan_3d, thalamus_mask, mask_centroid, [1,1,1])
    # visualize_coronal_slice(scan_3d, thalamus_mask, mask_centroid, [1,1,1])
    # visualize_sagittal_slice(scan_3d, thalamus_mask, mask_centroid, [1,1,1])
    # visualize_axial_slice(img_phantom, img_atlas, mask_centroid)

    vol = find_region_volume(thalamus_mask)
    surf = find_region_surface(thalamus_mask)

    print('thalamus volume:')
    print(f'  >> Result: {vol} mm^3')
    print(f'  >> Expected: 3744 mm^3')

    print('thalamus surface:')
    print(f'  >> Result: {surf} mm^2')
    print(f'  >> Expected: 1849-6920 mm^2 (depending on the approximation)')

if __name__ == '__main__':
    main()