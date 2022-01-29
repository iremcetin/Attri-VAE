
# import monai - for the image transformations
import monai
import torch
#from monai.networks.utils import eval_mode
from monai.transforms import (
    AddChannel, Compose, LoadImage, RandRotate90, 
    Resize, ScaleIntensityRange, ToTensor, NormalizeIntensity,
    RandFlip, RandSpatialCrop, ResizeWithPadOrCrop, ThresholdIntensity, ScaleIntensity
)
from monai.metrics import compute_roc_auc
monai.config.print_config()
random_seed = 42
monai.utils.set_determinism(random_seed)
#np.random.seed(random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from monai.transforms import AdjustContrast
import numpy as np

class EmidecDataset(torch.utils.data.Dataset):
  def __init__(self, image_files,  labels, rad_df, clinical_info_df, transforms):
    self.image_files = image_files
    self.labels = labels
    self.transforms = transforms
    self.clinical_info = np.asarray(clinical_info_df, dtype=float)
    self.rad = np.asarray(rad_df, dtype=float) # this is the attribute dataframe = either interpretable or radiomics
  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, index):
    return self.transforms(self.image_files[index]), self.labels[index], self.rad[index], self.clinical_info[index], self.image_files[index]


def dataset_dataloader(train_files, val_files, attributes_tra_scl, attributes_val_scl, train_clinical_info,val_clinical_info, train_labels, val_labels, win_size, batch_size, num_workers ):
# Define transforms for image
    train_transforms = Compose(
            [
                    LoadImage(image_only=True),
                    AddChannel(),
                    AdjustContrast(gamma = 5.0), # This transformation is great for EMIDEC
                    ScaleIntensity(minv=0.0, maxv=1.0), # this is important!!
                    #RandSpatialCrop( roi_size = win_size),
                    RandFlip( spatial_axis=0),
                    RandFlip( spatial_axis=1),
                    RandRotate90(),
        
                    Resize(win_size, mode = "area"),
      
                    ToTensor(),
            ]
            )
    val_transforms = Compose(
            [
                    LoadImage(image_only=True),
                    AddChannel(),
                    AdjustContrast(gamma = 5.0),
                    ScaleIntensity(minv=0.0, maxv=1.0),
                    Resize(win_size, mode = "area"),
   
                    ToTensor(),
            ]
            )


    # Define data loaders (Training and Validation dataset)
    
    train_ds = EmidecDataset(train_files, train_labels, attributes_tra_scl.values, train_clinical_info, train_transforms) 
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_ds = EmidecDataset(val_files, val_labels,  attributes_val_scl.values, val_clinical_info, val_transforms) 
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    
    return train_ds, train_loader, val_ds, val_loader
    