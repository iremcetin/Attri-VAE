import torch
import monai
from monai.transforms import (
    AddChannel, Compose, LoadImage,AdjustContrast,
    Resize, ScaleIntensityRange, ToTensor, NormalizeIntensity,
    RandFlip, RandSpatialCrop, ResizeWithPadOrCrop, ThresholdIntensity, ScaleIntensity
)
from monai.metrics import compute_roc_auc
monai.config.print_config()
random_seed = 42
monai.utils.set_determinism(random_seed)

from utils.utils_acdc_testing import acdc_dataload_process

drive_path_acdc ='/content/gdrive/My Drive/DATA/ACDC/processed_acdc_dataset/Patient_Groups/'

#DCM_path = os.path.join(drive_path, "DCM")
#HCM_path = os.path.join(drive_path, "HCM")
MINF_path = os.path.join(drive_path_acdc, "MINF")
NOR_path = os.path.join(drive_path_acdc, "NOR")
#RV_path = os.path.join(drive_path, "RV")

class ACDCDataset(torch.utils.data.Dataset):
  def __init__(self, image_files, labels, transforms):
    self.image_files = image_files
    self.labels = labels
    self.transforms = transforms

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, index):
    return self.transforms(self.image_files[index]), self.labels[index], self.image_files[index]

    #Note: to get the name of the files? -- when visualize with gradcam


images_ED, images_ES, segs_ED, segs_ES, labels_acdc = acdc_dataload_process(save_path_testing, MINF_path, NOR_path)
images_acdc = images_ED + images_ES
segs_acdc = segs_ED + segs_ES
labels_acdc = labels_acdc + labels_acdc

test_transforms = Compose(
    [
        LoadImage(image_only=True),
        AddChannel(),
        AdjustContrast(gamma = 1.1),
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize(win_size, mode = "area"),
        #NormalizeIntensity(nonzero=False, channel_wise=True),
        #ResizeWithPadOrCrop(spatial_size=win_size, mode="constant"),
        ToTensor(),
    ]
)

# Define the test loader
test_ds = ACDCDataset(images_acdc, labels_acdc,test_transforms )
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

# load the best hyperparameters
vis_vae_test = ConvVAE().cuda()
#vis_vae_test.load_state_dict(torch.load( path_saved_nets + "best_test_loss_model.pth")) # Best loss
vis_vae_test.load_state_dict(torch.load( path_saved_nets + "best_metric_model.pth")) #Best acc
#vis_vae_test.load_state_dict(torch.load( path_saved_nets + "best_AUC_model.pth")) #Best auc

acc_test, auc_test = acdc_test(0, vis_vae_test, test_loader)
print(f"Testing accuracy is {acc_test}, AUC is {auc_test}")
#