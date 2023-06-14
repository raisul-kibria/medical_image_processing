# medical_image_processing
## Structure
### Dataset:
- src/data/HCC-TACE-Seg: Contains the assigne HCC_06 dataset folder
- src/Coregistration: Contains the Phantom, AAL and RM_Brain_3D-SPGR data
### Script:
- src/utils/utils.py
The script provides utility functions for visualizations like MIP, and animation creation. Also, common functions like create_3d_demo used in both tasks are included.
- src/MultiObject_DICOM_loading_and_visualization.py
The script loads the patient CT data and segmentation series. It also aligns the CT images to corresponding masks and produces different visualizations. To visualize the intermediate outputs, the VISUALIZE constant has to be set to true.
- src/3D_Rigid_Coregistration_final.py
The script uses 3d image registration methods to create 3-dimensional parameters for translation, rotation and scaling. Each task in the code are commented in and out as required. They would be transformed to parameterized code in future commits.
