from training_main import main_train
from utils_evaluation import *
import nibabel as nib
import torch
'''
Define the configurations
'''
# Configurations for the training/testing ###################
batch_size = 16
num_workers = 1
seed = 3
epochs = 5000
learning_rate = 0.0001
# Configurations for VAE ####################################
IMG = 1 # 1 channel input = IMG
num_class = 2 # enter the number of classes in the dataset 
win_size = (80, 80, 80)  # Dimensions of the input to the net
HDIM =  96 # LVAE + MLP = h_dims = [96, 48, 24] # dim of the FC layer before the latent space (mu and sigma)
latent_size = 64
unflatten_channel = 2 # number of channels before unflatten : if IMG ==2 then 2, if IMG ==1 then 1
dim_start_up_decoder = [5,5,5] # in unflatten (inside the decoder), the dimensions will be [1, unflatten_channel, dim_startup_decoder]

'''
Train the model
'''
model, train_losses, test_losses, acc_metrices, auc_metrices, train_loader, val_loader = main_train(HDIM, IMG, latent_size, epochs, is_L1 = True, use_AR_LOSS = True, resume=False)
# ###########################
path_saved_nets = "put the path where the trained networks saved
path_example_img = ".../Case_N006/Images/Case_N006.nii.gz"
img_example = nib.load(path_example_img)
# load the train network
vis_vae = ConvVAE().cuda()
vis_vae.load_state_dict(torch.load( path_saved_nets + "best_test_loss_model.pth")) # Best loss
#vis_vae.load_state_dict(torch.load( path_saved_nets + "best_metric_model.pth")) #Best acc
#vis_vae.load_state_dict(torch.load( path_saved_nets + "best_AUC_model.pth")) #Best auc
path_tosave_nifti = path_saved_nets + "/" + "NIFTI/"
check_dir(path_tosave_nifti)
# #################################
'''
Evaluate the results
'''
z_dist, latent_z, z_prior, latent_mu, output_images, input_images, targets, clnc_info, filenames, z_gpu, radiomics_vals= compute_representations(train_loader, vis_vae, batch_size=batch_size)

### Experiment: attribute manipulation
min1 = -5
max1 = 5
dim1 = 0
num_points1D_vis = 10
sample_to_manipulate = 66
slc= 40
print(f"Selected sample: \n {filenames[sample_to_manipulate]}")
print(f" Interval of the selected dimensions: \n dim1:  min: {latent_z[:,dim1].min()} and max: {latent_z[:,dim1].max()} \n dim2: min: {latent_z[:,dim2].min()} and max: {latent_z[:,dim2].max()}")
print(f"Selected sample # {sample_to_manipulate} \n dim1 ({attributes_tra.columns[dim1]}) value: {latent_z[sample_to_manipulate, dim1]} \n dim2 ({attributes_tra.columns[dim2]}) value: {latent_z[sample_to_manipulate, dim2]}")
interp1, interp1Dim_dim1, z_1D = compute_latent_interpolations(latent_z, z_gpu, sample_to_manipulate, vis_vae, dim1=dim1, num_points = num_points1D_vis, min_val = min1, max_val =max1, slc=slc)
show(interp1, dim1)

### Experiment : Interpolation between two data points
case1 = 17
case2 = 66
latent_code2 = latent_z[case1]
latent_code1 = latent_z[case2]
print(f"case 1 : {filenames[case1]}\ncase 2 : {filenames[case2]}")
interpolated_images = make_latent_interp_animation(latent_code1, latent_code2, vis_vae, num_interps = 10, slc = 40)
show_(interpolated_images)
#### Experiment: attribute-wise attention map generation
target_layer = "conv5_enc" # target layer to be visualized
#labels_test  = np.zeros(len(img_dir_all))

model = vis_vae
gcam = GradCAM(model, target_layer=target_layer, cuda=True) 
show_gradcam(interp1Dim_dim1, model, gcam, interp_dim =dim1, attr_dim = dim1, the_slice = slc, visualize_ex = num_points1D_vis)

### Experiment: Latent space projection - to demonstrate how the networks constrains the latent space between pathological and healthy cases
axis_1 = 0
axis_2 = 2

df_rad = pd.DataFrame()
df_rad["Classes"] = classes = np.asanyarray(targets)
x_label = attributes_tra.columns[axis_1]
z_1 = latent_z[:,axis_1]
z_2 = latent_z[:,axis_2]
y_label = attributes_tra.columns[axis_2]

df_rad[x_label] = z_1
df_rad[y_label] = z_2
sns.scatterplot(x=x_label, y=y_label, hue=df_rad.Classes,
                palette=sns.color_palette("Set1", 2),
                data=df_rad).set(title="Latent space (z) projection") 
