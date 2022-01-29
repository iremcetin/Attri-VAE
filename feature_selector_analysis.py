# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:36:43 2022

@author: iremc
"""

from sklearn.feature_selection import SelectKBest, chi2, RFE, f_classif, VarianceThreshold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np

def feature_selector(train_scl, train_labels, df, n_feats):
    clf = SVC(kernel = "linear", C=1)
    #clf = RFC(random_state=0)
    selector = RFE(estimator = clf, n_features_to_select=n_feats)
    selected_rad =  selector.fit(train_scl, train_labels)
    col_indices  = selector.get_support(indices=True)
    col_names = df.columns[col_indices]
    return col_names, col_indices

def variance_feature_selector(df, threshold = 5e+5):
    var_threshold = VarianceThreshold(threshold = threshold)
    var_threshold.fit(df)
    col_indices = var_threshold.get_support(indices=True)
    col_names = df.columns[col_indices]
    return col_names, col_indices

def simple_eval(train_scl, val_scl, train_labels, val_labels):
    clf_2 = SVC(kernel = "linear", C=1)
    clf_2 = clf_2.fit(train_scl, train_labels )
    # 3. Make the prediction
    y_pred_svm = clf_2.predict(val_scl)
    # 4. Evaluation
    acc_svm = accuracy_score(val_labels, y_pred_svm)
    auc_svm = roc_auc_score(val_labels, y_pred_svm)
    print(f"Results of the analysis...")
    print(f"Accuracy is {acc_svm} and AUC score is {auc_svm}")
    
def feature_sel(clinical_info_train_scl, clinical_info_training, interpretable_train_scl, interpretable_val_scl, train_labels, val_labels, add_feat = True)
    #col_names_rad, col_indices_rad = feature_selector(radiomics_train_scl, train_labels, radiomics_tra, n_feats = 4)
    col_names_clnc, col_indices_clnc = feature_selector(clinical_info_train_scl, train_labels, clinical_info_training, n_feats=1)
    #col_names_rad, col_indices_rad = variance_feature_selector(radiomics_tra, threshold=5e+16)
    #col_names_clnc, col_indices_clnc = variance_feature_selector(clinical_info_training, threshold=5e+4)

    if add_feat == True:
        #ADD EXTRA CLNC FEATURES for the first interpretable experiment
        ef = 11
        sex = 1
        age = 2
        tobacco = 3

        add_feats_clnc = 4 # how many features you want to add additionally
        col_indices_all_clnc = np.zeros((col_indices_clnc.shape[0] + add_feats_clnc))
        col_indices_all_clnc[:len(col_indices_clnc)] = col_indices_clnc
        col_indices_all_clnc[-4] = ef
        col_indices_all_clnc[-3] = sex
        col_indices_all_clnc[-2] = age
        col_indices_all_clnc[-1] = tobacco
        clinical_tra_selected = train_clinical_info.iloc[:, col_indices_all_clnc]
    clinical_val_selected = val_clinical_info.iloc[:, col_indices_all_clnc]
    clinical_train_scl_selected, clinical_val_scl_selected = feature_preprocess(clinical_tra_selected, clinical_val_selected)
    simple_eval(interpretable_train_scl, interpretable_val_scl, train_labels, val_labels)
    simple_eval(clinical_train_scl_selected, clinical_val_scl_selected, train_labels, val_labels)
    ##### Define and return attribute dataframe 
    attributes_tra = pd.concat((train_interpretable, clinical_tra_selected), axis=1)
    attributes_val = pd.concat((val_interpretable, clinical_val_selected), axis=1)
    scl_values_tra, scl_values_val = feature_preprocess(attributes_tra.values, attributes_val.values)
    attributes_tra_scl = pd.DataFrame(data=scl_values_tra, columns=attributes_tra.columns)
    attributes_val_scl = pd.DataFrame(data=scl_values_val, columns=attributes_val.columns)
    #print(attributes_tra.columns)
    return attribute_tra_scl, attribute_val_scl, attributes_tra_scl, attributes_val_scl