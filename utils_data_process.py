# **Utils**

import re
import SimpleITK as sitk
import six
import nibabel as nib
import radiomics
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd
import os

def check_dir(path_dir):
  
    if os.path.isdir(path_dir):
        pass
    else:
       os.makedirs(path_dir)
       
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def path_load(syn_scar_path, file_name):
      ED_img_scar = [f for f in os.listdir(syn_scar_path) if f.endswith(file_name)] # these are info files
      #ED_img_scar.sort()
      ED_img_scar = sorted_alphanumeric(ED_img_scar)
      return ED_img_scar
def to_cuda_variable(tensor):
    """
    Converts tensor to cuda variable
    :param tensor: torch tensor, of any size
    :return: torch Variable, of same size as tensor
    """
    if torch.cuda.is_available():
        return Variable(tensor).cuda()
    else:
        return Variable(tensor)
    
def to_numpy(variable: Variable):
    """
    Converts torch Variable to numpy nd array
    :param variable: torch Variable, of any size
    :return: numpy nd array, of same size as variable
    """
    if torch.cuda.is_available():
        return variable.data.cpu().numpy()
    else:
        return variable.data.numpy()

def nii_reader(path):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("")
    reader.SetFileName(path)
    img = reader.Execute();
    return img

def mask_operations(mask, patient_id, path_binmask, low_thresh = 0, up_thresh = 0, Train = True):
    ##### Step 1 : Binary threshold 
    mask = sitk.Mask(mask, sitk.Not(mask>2),2)
    ##### Step 2 : Morphological closing --> fill the pixels with 0 inside ROI with 1 (fill the holes inside the ROI)
    closing = sitk.BinaryMorphologicalClosingImageFilter()
    mask =closing.Execute(mask)
    ### save image to check later
    binmask_path = path_binmask + patient_id + '_binMask.nii.gz'
    sitk.WriteImage(mask, binmask_path)
    return mask

def find_max_dim(shape_img, shape_gt):
    res1 = max(shape_img, key = lambda i : i[0])[0]
    res2 = max(shape_img, key = lambda i : i[1])[1]
    res3 = max(shape_img, key = lambda i : i[2])[2]
    print("Maximum image dimensions, dim1: %d, dim2: %d, dim3: %d"%(res1, res2, res3))
    res4 = max(shape_gt, key = lambda i : i[0])[0]
    res5 = max(shape_gt, key = lambda i : i[1])[1]
    res6 = max(shape_gt, key = lambda i : i[2])[2]
    print("Maximum GT dimensions, dim1: %d, dim2: %d, dim3: %d"%(res4, res5, res6)) # These dimensions must be the same as image patches (res1, res2 and res3)

    return

def create_patches(path, patient, cohort_name, Train):
    low_threshold = 2
    up_threshold = 3 # WHOLE IMAGE ##################
    # Define the image paths [images and masks]
    if cohort_name == "SYN":
        
        img_path = os.path.join(path,"%s.nii.gz" %(patient)) # Directory of image nii
        gt_path =os.path.join(path, "%s_gt.nii.gz" %(patient)) # Directory of Ground truth
        binMask_path = "/content/gdrive/MyDrive/Synthetic Data/syn_python/EMIDEC_cropped/synthetic_from_emidec_binMask/" # eger myo+lv yi syn icin kullanicaksan bu adresi degistir
    else: 
        img_path=os.path.join(path,"Images","%s.nii.gz" %(patient)) # Directory of image nii
        gt_path =os.path.join(path, "Contours","%s.nii.gz" %(patient)) # Directory of Ground truth
        binMask_path = "/content/gdrive/MyDrive/EMIDEC/Emidec_DATASET/processed_emidec_dataset/myo_lv/binMask/"

    print(f"Image path {img_path}")
    # Load the images with sitk
    img = nii_reader(img_path)
    gt_img = nii_reader(gt_path)

    # mask operations
    # Combine 2 labels together
   
    gt_img_l1 = mask_operations(gt_img, patient, binMask_path, low_threshold, up_threshold, Train)  # make the segmentation 1-layer only

    cropped_gt_1_label = sitk.Mask(gt_img_l1, sitk.Not(gt_img_l1>1),1)
    closing = sitk.BinaryMorphologicalClosingImageFilter()  
    cropped_gt_1_label =closing.Execute(cropped_gt_1_label)

    # ED
    (bbox, corrected_mask) = radiomics.imageoperations.checkMask(img, cropped_gt_1_label, correctMask=True )
    (cropped_img, cropped_gt) = radiomics.imageoperations.cropToTumorMask(img, gt_img_l1, boundingBox=bbox, label = 2)
    (cropped_gt_scar, a) = radiomics.imageoperations.cropToTumorMask(gt_img, gt_img_l1, boundingBox=bbox , label = 2)
    return cropped_img, cropped_gt, cropped_gt_scar

def preprocess_files(path_training, minf_cohort, cohort_name, Train = True):
    patient_ids=[] 
    Label = []
    images = []  # store the images and the segmentations for ED and ES separately - for the future experiments
    segs =[] 
    segs_scar =[]
    shape_img=[] 
    shape_gt = []

    if cohort_name =="SYN":
       save_path = "/content/gdrive/MyDrive/Synthetic Data/syn_python/EMIDEC_cropped/saved_syn_from_emidec_myo_only/"
    
    else:
      save_path = "/content/gdrive/MyDrive/EMIDEC/Emidec_DATASET/processed_emidec_dataset/whole_image/Training_cropped"

   
    for j in range(len(minf_cohort)):
          if cohort_name =="SYN":
            path_files=path_training
            file_p = path_training + minf_cohort[j] + ".nii.gz"
            gt_p =  path_training + minf_cohort[j] + "_gt.nii.gz"
           
          else:
            path_files = os.path.join(path_training, minf_cohort[j] )



          patient_id = minf_cohort[j]
          print('%s with %s'%(patient_id, cohort_name))  
          
         

          
          [cropped_img, cropped_gt, cropped_gt_scar] = create_patches(path_files, patient_id, cohort_name, Train)

          # 
          cropped_arr =  sitk.GetArrayFromImage(cropped_img)
          cropped_gt_arr = sitk.GetArrayFromImage(cropped_gt)
          
          cropped_gt_1_label = sitk.Mask(cropped_gt, sitk.Not(cropped_gt>1),1)
          closing = sitk.BinaryMorphologicalClosingImageFilter()  
          cropped_gt_1_label =closing.Execute(cropped_gt_1_label)

          #################################################################################################
          # multiply mask and ground truth to remove LV blood pool from the image = only for EMIDEC dataset
          multiply = sitk.MultiplyImageFilter()
          cropped_gt_casted = sitk.Cast(cropped_gt_1_label, sitk.sitkFloat64)
          
          new_img = multiply.Execute(cropped_img, cropped_gt_casted)
          
          new_img_arr = sitk.GetArrayFromImage(new_img)
          cropped_img = new_img
          #################################################################################################

          # Multiply scar segmentation with 1-label segmentation = Removes LV Blood pool from the segmentation 
          # This part is for visualization purposes
          

          filename_img = os.path.join(save_path,"%s.nii.gz" %(patient_id))
          print(f"saved in {filename_img}")
          filename_gt = os.path.join(save_path,"%s_gt.nii.gz" %(patient_id))
          #filename_img_org = os.path.join(save_path,"%s_org.nii.gz" %(patient_id))
          filename_gt_scar = os.path.join(save_path,"%s_gt_scar.nii.gz" %(patient_id))
          images.append(filename_img)
          segs.append(filename_gt)
          #segs_scar.append(filename_gt_scar)

          sitk.WriteImage(cropped_img, filename_img)
          #sitk.WriteImage(cropped_img, filename_img_org)
          sitk.WriteImage(cropped_gt, filename_gt)

          sitk.WriteImage(cropped_gt_scar, filename_gt_scar)
          #shape_img.append(cropped_arr.shape)
          shape_img.append(cropped_arr.shape)
          shape_gt.append(cropped_gt_arr.shape)

          if cohort_name =="MINF":
                    value = 1
          else: 
                value = 0
          
          Label.append(value)
          patient_ids.append(patient_id)


    return images, segs, Label, shape_img, shape_gt


# read the files
# load the variables
# If the files already cropped, you only need to load the images.

def load_preprocessed_files(path_training, minf_cohort, cohort_name):
    patient_ids=[] 
    Label = []
    images = [] 
    segs =[] 
    segs_scar =[]
    shape_img=[] 
    shape_gt = []

    if cohort_name =="SYN":
      save_path = path_training
    
    else:
      save_path = "/content/gdrive/MyDrive/EMIDEC/Emidec_DATASET/processed_emidec_dataset/whole_image/Training_cropped"
    #data = np.zeros((nb_patients,428,512,25)) # PREPROCESSING NEEDED : BOUNDARY BOX


    for j in range(len(minf_cohort)):
          path_files = os.path.join(path_training, minf_cohort[j] )
          patient_id = minf_cohort[j]
 
          print('%s with %s'%(patient_id, cohort_name))  

          if cohort_name =="SYN":
            filename_img_path = os.path.join(save_path,"%s" %(patient_id))
            filename_gt_path = os.path.join(save_path,"%s" %(patient_id))
            #filename_img_org = os.path.join(save_path,"%s_org.nii.gz" %(patient_id))
            filename_gt_scar_path = filename_gt_path

          else:

            filename_img_path = os.path.join(save_path,"%s.nii.gz" %(patient_id))
            filename_gt_path = os.path.join(save_path,"%s_gt.nii.gz" %(patient_id))
            #filename_img_org = os.path.join(save_path,"%s_org.nii.gz" %(patient_id))
            filename_gt_scar_path = os.path.join(save_path,"%s_gt_scar.nii.gz" %(patient_id))
            
          images.append(filename_img_path)
          segs.append(filename_gt_path)
          segs_scar.append(filename_gt_scar_path)

          filename_img = nii_reader(filename_img_path)
          filename_gt = nii_reader(filename_gt_path)
          filename_scar = nii_reader(filename_gt_scar_path)
      
          
          filename_img_arr = sitk.GetArrayFromImage(filename_img)
          gt_shape = sitk.GetArrayFromImage(filename_gt)
          scar_shape = sitk.GetArrayFromImage(filename_scar)

          shape_img.append(filename_img_arr.shape)
          shape_gt.append(gt_shape.shape)
          ####################################
          if cohort_name =="MINF":
                    value = 1
          else: 
                value = 0
          
          Label.append(value)
          patient_ids.append(patient_id)


    return images, segs, Label, shape_img, shape_gt

def radiomics_feature_extractor(img_dir, mask_dir, label):
  patient_ids = []
  radiomicsfeats_all = pd.DataFrame()

  for i in range(len(img_dir)):
        fn = img_dir[i][100:109]
        #print(fn)
        radiomics_extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
    
        radiomics_extractor.enableAllFeatures()
    
        img_ = sitk.ReadImage(img_dir[i],sitk.sitkUInt8)
        mask_ = sitk.ReadImage(mask_dir[i],sitk.sitkUInt8)
     
        
        radiomics_ = pd.DataFrame([radiomics_extractor.execute(img_, mask_, label)])
        if label== 1:
            radiomics_ = radiomics_.add_suffix("_LV")
        elif label == 2:
             radiomics_ = radiomics_.add_suffix("_MYO")
        if i == 0:
               radiomicsfeats_all=pd.DataFrame(data = np.zeros((len(img_dir),radiomics_.shape[1])),columns=radiomics_.columns)
                  
               radiomicsfeats_all=radiomics_
               patient_ids.append( fn )
        else:
               radiomicsfeats_all = radiomicsfeats_all.append(radiomics_)        
        
               patient_ids.append( fn )

        radiomicsfeats_all.index = patient_ids
        radiomicsfeats_all = radiomicsfeats_all.loc[:,~radiomicsfeats_all.columns.str.contains('diagnostics')]

  return radiomicsfeats_all

emidec_columns = ["GBS", "SEX", "AGE", "TOBACCO", "OVERWEIGHT", "ART_HYPERT", "DIABETES", "FHCAD", "ECG", "TROPONIN", "KILLIP", "FEVG", "NTProNBP"]


def load_clinical_info(path, info_files, emidec_columns, folders):
    clinical_info = pd.DataFrame(index=folders, columns=emidec_columns)
    for i in range(len(info_files)):

        Info_path=os.path.join(path,info_files[i])

        with open(Info_path, 'r', encoding='mac_roman') as Info : 
             line = Info.readlines()
             for j in range(2, len(emidec_columns)+2):   
                 if j == 2:
                    clinical_info.iloc[i,j-2] = line[j].rstrip().split(":")[1][:3]
                 else:
                    clinical_info.iloc[i,j-2] = line[j].rstrip().split(":")[1]   


    return clinical_info
                


# Overlay Original image and GradCAM heatmap
def superimposed_image_def(cam_img, original_img, alpha = 0.6):
    original_img = original_img[0, 0,:,:] # If you use different image check here and change
    cam_img = cam_img[0,0,:,:] # Same above
    # Make 1-channel image to 3-channel : To overlay this image with the heatmap (JET colormap = 3-channels)
    original_img = cv2.merge((original_img, original_img, original_img))
    cam_img = (cam_img - cam_img.min()) / (
               cam_img.max() - cam_img.min()
    ) *255
    
    # Convert to Heatmap ---- JET COLORMAP
    #cam_img = cam_img.astype(np.uint8)
    cam_img = cv2.applyColorMap(np.uint8(cam_img), cv2.COLORMAP_JET)

    original_img = np.uint8(
        (original_img - original_img.min())
        / (original_img.max() - original_img.min())
        * 255
    )

    ###### Superimpose Heatmap on Image Data #####################################
   
    superimposed_image = cv2.addWeighted(original_img, alpha, cam_img, (1-alpha), 0.0)

    return superimposed_image, cam_img, original_img



