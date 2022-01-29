# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:26:26 2022

@author: iremc
"""
from sklearn.model_selection import train_test_split

def tra_val_oversample(images_tra_, segs_tra_, labels_tra, indices, interpretable_feats_tra, do_oversampling=True):
    train_files, val_files, train_segs, val_segs, train_labels,  val_labels, train_interpretable, val_interpretable, train_clinical_info, val_clinical_info, indices_train, indices_val = train_test_split(images_tra_, segs_tra_, labels_tra_,interpretable_feats_tra, clinical_info_training , indices, stratify = labels_tra_, test_size=0.3, random_state=11)
    print(f" Training has {np.sum(train_labels)} MINF and {np.abs(len(train_labels)-np.sum(train_labels))} NOR")
    print(f" Validation has {np.sum(val_labels)} MINF and {np.abs(len(val_labels)-np.sum(val_labels))} NOR")
    ###oversampling class 0 ###
    if do_oversampling == True:
        difference = np.sum(train_labels) - np.abs(len(train_labels)-np.sum(train_labels)) # difference between 1s(MINF) and 0s (NOR)
        indcs = np.array([i for i,v in enumerate(train_labels) if v == 0])
        rand_subsample = np.random.choice(indcs, difference) # randomly select the indices "difference" times
        
        train_files = np.concatenate([train_files, np.take(train_files, rand_subsample)])
        train_labels=np.concatenate([train_labels, np.take(train_labels, rand_subsample, axis=0)])
        train_interpretable = pd.concat((train_interpretable, train_interpretable.iloc[rand_subsample]), axis=0)
        train_clinical_info= pd.concat((train_clinical_info, train_clinical_info.iloc[rand_subsample]), axis=0)
        indices_train = np.concatenate([indices_train, np.take(indices_train, rand_subsample)])
    print(f" After oversampling training has {np.sum(train_labels)} MINF and {np.abs(len(train_labels)-np.sum(train_labels))} NOR")
    print(f" After oversampling validation has {np.sum(val_labels)} MINF and {np.abs(len(val_labels)-np.sum(val_labels))} NOR")
    
    return train_files, val_files, train_segs, val_segs, train_labels,  val_labels, train_interpretable, val_interpretable, train_clinical_info, val_clinical_info, indices_train, indices_val