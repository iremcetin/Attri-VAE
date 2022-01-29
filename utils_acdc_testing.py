# **1.UTILS**
import SimpleITK as sitk
import os
import torch
from utils_evaluation import from_nested_to_
from loss_functions import mean_accuracy
from utils_data_process import nii_reader, check_dir
import radiomics
def compute_representations_acdc(train_loader, model, batch_size ):
    model.eval()
    batch_1 = batch_size
    latent_z = []
    latent_mu = []
    targets = []
    
    filenames=[]
    z_nonregularized = []
    output_images = []
    input_images = []
    z_distribution = []
    z_gpu = []
    for batch_1 in train_loader:
        
        data = batch_1[0]
  
        data = data.cuda().float() 
        recon_batch, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist = model(data) 
        #recon_batch, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist = gcam.forward(data)
        

        z_gpu.append(z_tilde)

        z = z_tilde.detach().cpu().numpy()
        latent_z.append(z)
        z_distribution.append(z_dist)
        z_nonregularized.append(z_prior.detach().cpu().numpy())
        latent_mu.append(mu.detach().cpu().numpy())

        output_images.append(recon_batch.detach().cpu().numpy())
        input_images.append(batch_1[0].detach().cpu().numpy())
       
        targets.append(batch_1[1].detach().cpu().numpy())
        filenames.append(batch_1[2])

    z_gpu = torch.cat((z_gpu[0], z_gpu[1], z_gpu[2], z_gpu[3], z_gpu[4]), dim=0)
    return z_distribution, from_nested_to_(latent_z), from_nested_to_(z_nonregularized), from_nested_to_(latent_mu), from_nested_to_(output_images), from_nested_to_(input_images), from_nested_to_(targets),  from_nested_to_(filenames), z_gpu

def acdc_test(epoch, model, test_loader):
    model.eval()
    test_loss_ = 0
    iter = 0
    test_loss = 0
    acc_metric = 0
    acc_value = 0
    auc_value=0
    auc_metric = 0
    with torch.no_grad():
        for batch_idx, (data_test, label, filename) in enumerate(test_loader):
            
            iter = iter + 1
            
            data_test = data_test.to(device)
            label = label.to(device)
           
            recon_batch, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist = model(data_test)
            
            
            accuracy,  roc = mean_accuracy(label,  out_mlp)
            
            #acc_0 = torch.eq(out_mlp.argmax(dim=1), label)
            #acc_1 = acc_0.sum().item()/ len(acc_0)
            #acc_value += acc_1
            acc_value += accuracy
            auc_value += roc

           
    
    
    acc_metric = acc_value /iter
    auc_metric = auc_value /iter
    return acc_metric, auc_metric



def cohort_to_label(cohort_name):
    value = 0
    if cohort_name == 'NOR':
        value = 0
   
    elif cohort_name == 'MINF':
        value = 1
   

    return value

def mask_operations(mask, patient_id, path_binmask, low_thresh = 0, up_thresh = 0):
    ##### Note : Mask is not binarize --> You need to binarize it
    ##### Step 1 : Binary threshold 
#    mask_array = sitk.GetArrayFromImage(mask)
    btif=sitk.BinaryThresholdImageFilter()
    btif.SetLowerThreshold(low_thresh) # lower value to set as background = 0 : Inside value
    btif.SetUpperThreshold(up_thresh) # upper value, if you set 2: label 3 will be visible, if you set 1, label 2 and 3 visible 
    btif.SetInsideValue(0)
    btif.SetOutsideValue(1)
    mask = btif.Execute(mask)
    ##### Step 2 : Morphological closing --> fill the pixels with 0 inside ROI with 1 (fill the holes inside the ROI)
    closing = sitk.BinaryMorphologicalClosingImageFilter()
    mask =closing.Execute(mask)
#    binMask = sitk.GetArrayFromImage(mask) ### To check the mask's pixel values
    
    ### save image to check later
    binmask_path = path_binmask + patient_id + '_binMask.nii.gz'
    sitk.WriteImage(mask, binmask_path)
    
    
    return mask

def create_patches(path, patient, syn = False):# if it is for synthetic images, syn = True
    low_threshold = 0
    up_threshold = 1
    #radiomics_extractor = radiomics.featureextractor.RadiomicsFeaturesExtractor()
    Info_path=os.path.join(path, patient,'Info.cfg')
    Info=open( Info_path)
    Info_2=[]

    with open( Info_path) as Info : 
         for line in Info:
#    a.append(line)
             data=line.rstrip().split(":")
             Info_2.append(str(data[1]))    
    
    ED=int(Info_2[0])
    ES=int(Info_2[1])    

 
    # Define the image paths [images and masks]
    ED_path=os.path.join(path,patient,"%s_frame%02d.nii.gz" %(patient,ED)) # Directory of ED nii
    ED_gt_path =os.path.join(path, patient,"%s_frame%02d_gt.nii.gz" %(patient,ED)) # Directory of ED Ground truth

    ES_path=os.path.join(os.sep, path,patient,"%s_frame%02d.nii.gz" %(patient,ES)) # Directory of ES nii
    ES_gt_path =os.path.join(path, patient,"%s_frame%02d_gt.nii.gz" %(patient,ES)) # Directory of ED Ground truth


    #[nii_data_ED, aff, hdr] =load_nii(ED_path)
    #[nii_data_ES, nii_affine_ES, nii_hdr_ES] =load_nii(ES_path)
    #ED_img = sitk.ReadImage(ED_path,sitk.sitkUInt8)
    #ED_gt_image = sitk.ReadImage(ED_gt_path, sitk.sitkUInt8)

    if syn == False:
    # Load the images with sitk
      ED_img = nii_reader(ED_path)
      ED_gt_img = nii_reader(ED_gt_path)

      ES_img = nii_reader(ES_path)
      ES_gt_img = nii_reader(ES_gt_path)
    else :
      ED_img = sitk.ReadImage(ED_path)
      ED_gt_img = sitk.ReadImage(ED_gt_path)

      ES_img = sitk.ReadImage(ES_path)
      ES_gt_img = sitk.ReadImage(ES_gt_path)
    # mask operations
    
    ED_gt_img_l1 = mask_operations(ED_gt_img, patient, path, low_threshold, up_threshold)  # make the segmentation 1-layer only
    ES_gt_img_l1 = mask_operations(ES_gt_img, patient, path, low_threshold, up_threshold )

    # ED
    (bbox_ed, corrected_mask_ed) = radiomics.imageoperations.checkMask(ED_img, ED_gt_img_l1, correctMask=True )
    (cropped_ED, cropped_ED_gt) = radiomics.imageoperations.cropToTumorMask(ED_img, ED_gt_img_l1, boundingBox=bbox_ed)

    # ES
    (bbox_es, corrected_mask_es) = radiomics.imageoperations.checkMask(ES_img, ES_gt_img_l1, correctMask=True )
    (cropped_ES, cropped_ES_gt) = radiomics.imageoperations.cropToTumorMask(ES_img, ES_gt_img_l1, boundingBox=bbox_es)
    return cropped_ED, cropped_ED_gt, ED, cropped_ES, cropped_ES_gt, ES 

def acdc_dataload_process(save_path_testing, MINF_path, NOR_path)
    patient_ids=[] ; Diagnosis=[]
    labels_acdc = [] ; images_ED = [] ; images_ES = [] # store the images and the segmentations for ED and ES separately - for the future experiments
    segs_ED =[] ; segs_ES = []
    shape_ED=[] ; shape_ED_gt=[]
    shape_ES=[] ; shape_ES_gt=[]
    #save_path_testing = "/content/gdrive/MyDrive/ACDC/processed_acdc_dataset/Patient_Groups/Cropped/"
    check_dir(save_path_testing)
#data = np.zeros((nb_patients,428,512,25)) # PREPROCESSING NEEDED : BOUNDARY BOX
    path_testing = [MINF_path, NOR_path]
    cohorts =["MINF", "NOR"]

    for j in range(len(path_testing)):
        path_cvd = path_testing[j]
        print(path_cvd)
        cohort_name = cohorts[j]
        nb_patients=len(next(os.walk(path_cvd))[1])
        patient_ids_test = [f for f in os.listdir(path_testing[j]) ]
        
    for i in range(nb_patients):
        patient = '%s' %(patient_ids_test[i]) 
        print('%s with %s'%(patient, cohort_name))  
        
        [cropped_ED, cropped_ED_gt, ED, cropped_ES, cropped_ES_gt, ES] = create_patches(path_cvd, patient)
        
        # ED
        cropped_ED_arr =  sitk.GetArrayFromImage(cropped_ED)
        cropped_ED_gt_arr = sitk.GetArrayFromImage(cropped_ED_gt)
        cropped_gt_1_label_ED = sitk.Mask(cropped_ED_gt, sitk.Not(cropped_ED_gt>1),1)
        closing = sitk.BinaryMorphologicalClosingImageFilter()  
        cropped_gt_1_label_ED =closing.Execute(cropped_gt_1_label_ED)
        #################################################################################################
        # multiply mask and ground truth to remove LV blood pool from the image = only for EMIDEC dataset
        multiply = sitk.MultiplyImageFilter()
        cropped_gt_casted_ED = sitk.Cast(cropped_gt_1_label_ED,  sitk.sitkFloat64)
        cropped_ED_casted = sitk.Cast(cropped_ED,  sitk.sitkFloat64)
        
        new_img_ED = multiply.Execute(cropped_ED_casted, cropped_gt_casted_ED)
              
        new_img_arr_ED = sitk.GetArrayFromImage(new_img_ED)
        cropped_img_ED = new_img_ED
        #################################################################################################
        filename_ED = os.path.join(save_path_testing,"%s_%s_frame%02d.nii.gz" %(cohort_name, patient, ED))
        filename_ED_gt = os.path.join(save_path_testing,"%s_%s_frame%02d_gt.nii.gz" %(cohort_name, patient, ED))
        
        images_ED.append(filename_ED)
        segs_ED.append(filename_ED_gt)
        
        sitk.WriteImage(cropped_img_ED, filename_ED)
        sitk.WriteImage(cropped_ED_gt, filename_ED_gt)
        
        shape_ED.append(cropped_ED_arr.shape)
        shape_ED_gt.append(cropped_ED_gt_arr.shape)
        
        
        # ES
        cropped_ES_arr =  sitk.GetArrayFromImage(cropped_ES)
        cropped_ES_gt_arr = sitk.GetArrayFromImage(cropped_ES_gt)
        cropped_gt_1_label_ES = sitk.Mask(cropped_ES_gt, sitk.Not(cropped_ES_gt>1),1)
        closing = sitk.BinaryMorphologicalClosingImageFilter()  
        cropped_gt_1_label_ES =closing.Execute(cropped_gt_1_label_ES)
        #################################################################################################
        # multiply mask and ground truth to remove LV blood pool from the image = only for EMIDEC dataset
        multiply = sitk.MultiplyImageFilter()
        cropped_gt_casted_ES = sitk.Cast(cropped_gt_1_label_ES, sitk.sitkFloat64)
        cropped_ES_casted = sitk.Cast(cropped_ES, sitk.sitkFloat64)
        new_img_ES = multiply.Execute(cropped_ES_casted, cropped_gt_casted_ES)
              
        new_img_arr_ES = sitk.GetArrayFromImage(new_img_ES)
        cropped_img_ES = new_img_ES
        #################################################################################################
        filename_ES = os.path.join(save_path_testing,"%s_%s_frame%02d.nii.gz" %(cohort_name, patient, ES))
        filename_ES_gt = os.path.join(save_path_testing,"%s_%s_frame%02d_gt.nii.gz" %(cohort_name, patient, ES))
        
        images_ES.append(filename_ES)
        segs_ES.append(filename_ES_gt)
        
        sitk.WriteImage(cropped_img_ES, filename_ES)
        sitk.WriteImage(cropped_ES_gt, filename_ES_gt)
        
        Diagnosis.append(cohort_name)
        
        shape_ES.append(cropped_ES_arr.shape)
        shape_ES_gt.append(cropped_ES_gt_arr.shape)
        
        ####################################
        value = cohort_to_label(cohort_name)
        labels_acdc.append(value)
        patient_ids.append(patient)
        return images_ED, images_ES, segs_ED, segs_ES, labels_acdc
     