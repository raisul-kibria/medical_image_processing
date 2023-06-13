import os
import pydicom
import numpy as np
from utils.utils import create_animation, alpha_fusion, visualize_MIP_per_plane
import matplotlib.pyplot as plt

VISUALIZE = False # flag to display intermediate results

def main():
    # Directory of data
    dataset_dir = r'data/HCC-TACE-Seg/HCC_006/08-06-1999-NA-ABDPEL LIVER-65146/103.000000-LIVER 3 PHASE AP-45033'
    segment_dir = r'data/HCC-TACE-Seg/HCC_006/08-06-1999-NA-ABDPEL LIVER-65146/300.000000-Segmentation-58134/1-1.dcm'
    seg_dcm = pydicom.dcmread(segment_dir)

    # Object names and color configuration
    objects = ['Liver', 'Mass', 'Portal vein', 'Abdominal aorta']
    object_colors = [(1,0,0), (0,1,0), (0,0,1), (0,0.75,0.5)]

    # Creating an index of the scans referring to corresponding segmentation masks
    valid_masks = {} # maintains a dict {scan_id: mask_id: {idx}}
    series_dt = seg_dcm.PerFrameFunctionalGroupsSequence
    for i in range(len(series_dt)):
        ref_ct = series_dt[i].DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
        if ref_ct not in valid_masks:
            valid_masks[str(ref_ct)] = {
                str(series_dt[i].SegmentIdentificationSequence[0].ReferencedSegmentNumber):
                {
                    "idx": i,
                    "image_position": series_dt[i].PlanePositionSequence[0].ImagePositionPatient
                }
            }
        else:
            valid_masks[str(ref_ct)][str(series_dt[i].SegmentIdentificationSequence[0].ReferencedSegmentNumber)] = {
                "idx": i,
                "image_position": series_dt[i].PlanePositionSequence[0].ImagePositionPatient
            }      

    # Ordering the scanned images & checking for multi-acqusitions
    ct_slices = []
    count = 0
    acquisition = -1
    for f in os.listdir(dataset_dir):
        img = os.path.join(dataset_dir, f)
        ct_dcm = pydicom.dcmread(img)
        if acquisition == -1:
            acquisition = ct_dcm.AcquisitionNumber
            # in case of multi-acquisitions check, the Acquisition Number of only 
            # the first slice is considered as the primary one.
        slice_acquisition = ct_dcm.AcquisitionNumber
        if hasattr(ct_dcm, 'SliceLocation') and slice_acquisition == acquisition:
            ct_slices.append(ct_dcm)
        elif slice_acquisition != acquisition:
            # different acqusition slices are skipped
            print(f'Multi acquisition found with Acquisition Number: {slice_acquisition}')
            count+=1
        else:
            # the sorting of slices is based on the SliceLocation attribute
            # the slices without the attribute are also skipped.
            count+=1

    ct_slices = sorted(ct_slices, key=lambda x: -x.SliceLocation) 
    # the SliceLocation sorts the acquisition from top-to-bottom of the patient

    pixel_len_mm = [ct_dcm.SliceThickness]
    pixel_len_mm.extend(ct_dcm.PixelSpacing)

    img_3d = [f.pixel_array for f in ct_slices]
    if VISUALIZE:
        for img in img_3d:
            plt.imshow(img, aspect=pixel_len_mm[1]/pixel_len_mm[2], cmap='bone')
            plt.title('CT slices on axial plane')
            plt.show()

    seg_3d = np.array(seg_dcm.pixel_array) # the unordered segmentation volume
    reordered_seg = []

    # Creating the volume for the mask with id's for each object in correct order
    no_mask_count = 0
    for ct_dcm in ct_slices:
        mask = np.zeros((seg_3d[0].shape))
        if str(ct_dcm.SOPInstanceUID) in valid_masks:
            for k in valid_masks[str(ct_dcm.SOPInstanceUID)].keys():
                ref_mask = seg_3d[valid_masks[str(ct_dcm.SOPInstanceUID)][str(k)]['idx']]
                mask[ref_mask == 1] = int(k) # we create a multilabel mask with different object identifiers
        else:
            print(f'No ROI in the slice {str(ct_dcm.SOPInstanceUID)}')
            no_mask_count+=1
        reordered_seg.append(mask)
    print(f"Skip count: {count}")
    print(f"No masks associated count: {no_mask_count}")

    img_segmented = []
    print(f'Number of scan images: {len(img_3d)} and Number of masks: {len(reordered_seg) - no_mask_count}')

    # Alpha fusing all the scanned images on their axial plane (can be done on other planes as well)
    for i in range(len(img_3d)):
        img_segmented.append(alpha_fusion(img_3d[i], reordered_seg[i], len(objects), object_colors))
        if VISUALIZE:
            if i%2 == 0:
                plt.imshow(img_segmented[-1], aspect=pixel_len_mm[1]/pixel_len_mm[2])
                plt.title('Alpha-fused Slices on Axial Plane')
                plt.show()
    img_3d = np.array(img_segmented)
    if VISUALIZE:
        visualize_MIP_per_plane(img_3d, pixel_len_mm)
    create_animation(img_3d, pixel_len_mm, objects, object_colors, n=20, save_dir='results/results_20', show=VISUALIZE)

if __name__ == '__main__':
    main()
