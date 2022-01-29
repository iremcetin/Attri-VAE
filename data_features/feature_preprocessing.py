# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:32:34 2022

@author: iremc
"""

from sklearn.preprocessing import MinMaxScaler

def feature_preprocess(train_feats, val_feats):
     scaler = MinMaxScaler(feature_range=(0,1))
     train_feats_scl = scaler.fit_transform(train_feats)
     val_feats_scl = scaler.transform(val_feats)
     return train_feats_scl, val_feats_scl

def feature_preprocessing_main(train_interpretable, val_interpretable, train_clinical_info, val_clinical_info  )
    interpretable_train_scl, interpretable_val_scl = feature_preprocess(train_interpretable, val_interpretable)
    clinical_info_train_scl, clinical_info_val_scl  = feature_preprocess(train_clinical_info, val_clinical_info)
    return interpretable_train_scl, interpretable_val_scl, clinical_info_train_scl, clinical_info_val_scl