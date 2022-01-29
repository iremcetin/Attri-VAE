
# **Interpretable Features (UTILS)**

from scipy import ndimage
import scipy
from skimage import feature
import numpy as np
import nibabel as nib
def ejection_fraction(ed_vol, es_vol):
    """
    Calculate ejection fraction
    """
    stroke_vol = ed_vol - es_vol
    return (np.float(stroke_vol)/np.float(ed_vol))

def myocardialmass(myocardvol):
    """
    Specific gravity of heart muscle (1.05 g/ml)
    """ 
    return myocardvol*1.05

def bsa(height, weight):
    """
    Body surface Area
    """
    return np.sqrt((height*weight)/3600)

def myocardial_thickness(data_path, slices_to_skip=(0,0), myo_label=2):
    """
    Calculate myocardial thickness of mid-slices, excluding a few apex and basal slices
    since myocardium is difficult to identify
    """
    label_obj = nib.load(data_path)
    myocardial_mask = (label_obj.get_data()==myo_label)
    # pixel spacing in X and Y
    pixel_spacing = label_obj.header.get_zooms()[:2]
    assert pixel_spacing[0] == pixel_spacing[1]

    holes_filles = np.zeros(myocardial_mask.shape)
    interior_circle = np.zeros(myocardial_mask.shape)

    cinterior_circle_edge=np.zeros(myocardial_mask.shape)
    cexterior_circle_edge=np.zeros(myocardial_mask.shape)

    overall_avg_thickness= []
    overall_std_thickness= []
    for i in range(slices_to_skip[0], myocardial_mask.shape[2]-slices_to_skip[1]):
        holes_filles[:,:,i] = ndimage.morphology.binary_fill_holes(myocardial_mask[:,:,i])
        interior_circle[:,:,i] = holes_filles[:,:,i] - myocardial_mask[:,:,i]
        cinterior_circle_edge[:,:,i] = feature.canny(interior_circle[:,:,i])
        cexterior_circle_edge[:,:,i] = feature.canny(holes_filles[:,:,i])
       
        x_in, y_in = np.where(cinterior_circle_edge[:,:,i] != 0)
        number_of_interior_points = len(x_in)
        x_ex,y_ex=np.where(cexterior_circle_edge[:,:,i] != 0)
        number_of_exterior_points=len(x_ex)
        if len(x_ex) and len(x_in) !=0:
            total_distance_in_slice=[]
            for z in range(number_of_interior_points):
                distance=[]
                for k in range(number_of_exterior_points):
                    a  = [x_in[z], y_in[z]]
                    a=np.array(a)
                    b  = [x_ex[k], y_ex[k]]
                    b=np.array(b)
                    dst = scipy.spatial.distance.euclidean(a, b)

                    distance = np.append(distance, dst)
                distance = np.array(distance)
                min_dist = np.min(distance)
                total_distance_in_slice = np.append(total_distance_in_slice,min_dist)
                total_distance_in_slice = np.array(total_distance_in_slice)

            average_distance_in_slice = np.mean(total_distance_in_slice)*pixel_spacing[0]
            overall_avg_thickness = np.append(overall_avg_thickness, average_distance_in_slice)

            std_distance_in_slice = np.std(total_distance_in_slice)*pixel_spacing[0]
            overall_std_thickness = np.append(overall_std_thickness, std_distance_in_slice)


    return (overall_avg_thickness, overall_std_thickness)