
import os
from utils_data_process import load_clinical_info, radiomics_feature_extractor
from utils_interpretable_features import myocardialmass, myocardial_thickness
# **Load the Emidec dataset & clinical info (from gdrive)**

def data_load_feature_extract(training_path ):
    #training_path = "/content/gdrive/MyDrive/EMIDEC/Emidec_DATASET/Training"  
    emidec_columns = ["GBS", "SEX", "AGE", "TOBACCO", "OVERWEIGHT", "ART_HYPERT", "DIABETES", "FHCAD", "ECG", "TROPONIN", "KILLIP", "FEVG", "NTProNBP"]

    path_files = os.listdir(training_path)
    info_files = [f for f in os.listdir(training_path) if f.endswith('.txt')] # these are info files
    info_files.sort()
    folders =  [f for f in os.listdir(training_path) if not f.endswith('.txt')]# the others are the patient folders
    folders.sort()

    ####  CLINICAL INFORMATION ##################
    clinical_info_training = load_clinical_info(training_path, info_files, emidec_columns, folders)


    # Preprocess the values
    cleanup_nums = {"SEX":     {" F": 1, " M": 0},
                "OVERWEIGHT": {" N": 0, " Y": 1},
                "ART_HYPERT": {" N": 0, "N":0, " Y": 1},
                "DIABETES": {" N": 0, " Y": 1},
                "FHCAD" : {" N": 0, " Y": 1},
                "ECG" : {" N": 0, " Y": 1}}


    clinical_info_training = clinical_info_training.replace(cleanup_nums)
    clinical_info_training_nor = clinical_info_training[:33]
    clinical_info_training_minf = clinical_info_training[33:]

    healthy_cohort =  [param for param in folders if param.split("_")[1].startswith('N')] # the ones start with N (after _) are healthy
    minf_cohort = [param for param in folders if param.split("_")[1].startswith('P')] # the ones start with P are MINF
    nb_healthy =  len(healthy_cohort)
    nb_minf = len(minf_cohort)
    print(f"There are {nb_healthy} healthy and {nb_minf} heart attack patients in the dataset.")

    if doyouhavecroppedfiles == True: 
        images_minf, segs_minf, Label_minf, shape_img_minf, shape_gt_minf  = load_preprocessed_files(training_path, minf_cohort, cohort_name='MINF')
        images_nor, segs_nor, Label_nor,  shape_img_nor, shape_gt_nor  = load_preprocessed_files(training_path, healthy_cohort, cohort_name='NOR')
    else:
        images_minf, segs_minf, Label_minf, shape_img_minf, shape_gt_minf = preprocess_files(training_path, minf_cohort, cohort_name='MINF', Train=True)
        images_nor, segs_nor, Label_nor, shape_img_nor, shape_gt_nor = preprocess_files(training_path, healthy_cohort, cohort_name='NOR', Train=True)

    ### RADIOMICS FEATURE EXTRACTIONS #######

    radiomics_all_minf_lv = radiomics_feature_extractor(images_minf, segs_minf, label= 1)
    radiomics_all_minf_myo = radiomics_feature_extractor(images_minf, segs_minf, label= 2)
    radiomics_all_nor_lv = radiomics_feature_extractor(images_nor, segs_nor, label = 1)
    radiomics_all_nor_myo = radiomics_feature_extractor(images_nor, segs_nor, label = 2)


    #print(f"Shape of the radiomics_minf: {radiomics_all_minf_myo.shape}")
    #print(f"Shape of the radiomics_nor: {radiomics_all_nor_myo.shape}")

    ### all cases from emidec training dataset
    images_tra_ = images_nor + images_minf 
    segs_tra_ = segs_nor +  segs_minf
    labels_tra_ = Label_nor + Label_minf 
    radiomics_tra_lv = pd.concat([radiomics_all_nor_lv , radiomics_all_minf_lv], axis=0)
    radiomics_tra_myo = pd.concat([radiomics_all_nor_myo , radiomics_all_minf_myo], axis=0)
    #MERGE LV AND MYO 
    radiomics_tra = pd.concat([radiomics_tra_lv , radiomics_tra_myo], axis=1)
    indices = np.arange(len(images_tra_))

    ### Extract interpretable features for the experiments ####
    
    interpretable_volume_lv  = radiomics_tra_lv.filter(regex = "original_shape_VoxelVolume", axis = 1)
    interpretable_volume_myo = radiomics_tra_myo.filter(regex = "original_shape_VoxelVolume", axis = 1)
    myo_mass = myocardialmass(interpretable_volume_myo)
    myo_thickness_max_avg = []
    myo_thickness_std_avg = []
    myo_thickness_mean_std = []
    myo_thickness_std_std = []

    for i in range(len(segs_tra_)):
        myo_thickness = myocardial_thickness(segs_tra_[i]) 
        myo_thickness_max_avg.append(np.amax(myo_thickness[0]))
        myo_thickness_std_avg.append(np.std(myo_thickness[0]))
        myo_thickness_mean_std.append(np.mean(myo_thickness[1]))
        myo_thickness_std_std.append(np.std(myo_thickness[1]))

    interpretable_feats_tra = pd.DataFrame(index = interpretable_volume_lv.index, columns=["LV_Volume", "MYO_Volume", "MYO_Mass", "MYO_thickness_std_avg", "MYO_thickness_mean_std"])
    interpretable_feats_tra.iloc[:,0] = interpretable_volume_lv.values
    interpretable_feats_tra.iloc[:,1] = interpretable_volume_myo.values
    interpretable_feats_tra.iloc[:,2] = myo_mass.values
    interpretable_feats_tra.iloc[:,3] = myo_thickness_std_avg
    interpretable_feats_tra.iloc[:,4] = myo_thickness_mean_std
    
    return images_tra_, segs_tra_, labels_tra, indices, interpretable_feats_tra, radiomics_tra, clinical_info_training
