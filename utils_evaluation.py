# **1. Utils**

import torchvision.transforms.functional as TF
def show(imgs, dim):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), figsize=(25,25), squeeze=False)
    for i, img in enumerate(imgs): 
        img = img.detach()
        img = TF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), cmap = "gray")
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
       # plt.xlabel(radiomics_tra_selected.columns[dim])

def show2d(imgs, dim_1, dim_2):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), figsize=(10,10), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = TF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), cmap ="gray")
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.xlabel(attributes_tra.columns[dim_1])
        plt.ylabel(attributes_tra.columns[dim_2])

def from_nested_to_(nested_list):
    normal_list = np.concatenate((nested_list[0], nested_list[1], nested_list[2], nested_list[3], nested_list[4]))
   
    return normal_list

def compute_representations(train_loader, model, batch_size ):
    model.eval()
    batch_1 = batch_size
    latent_z = []
    latent_mu = []
    targets = []
    clnc_info=[]
    rad_vals = []
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
        rad_vals.append(batch_1[2].detach().cpu().numpy())
        
        targets.append(batch_1[1].detach().cpu().numpy())
        clnc_info.append(batch_1[3].detach().cpu().numpy())
        filenames.append(batch_1[4])

    z_gpu = torch.cat((z_gpu[0], z_gpu[1], z_gpu[2], z_gpu[3], z_gpu[4]), dim=0)
    return z_distribution, from_nested_to_(latent_z), from_nested_to_(z_nonregularized), from_nested_to_(latent_mu), from_nested_to_(output_images), from_nested_to_(input_images), from_nested_to_(targets), from_nested_to_(clnc_info), from_nested_to_(filenames), z_gpu, from_nested_to_(rad_vals)


def compute_latent_interpolations(latent_z, z_gpu, sample_to_manipulate, model, dim1=0, num_points=15, min_val=-1, max_val=2, slc = 40):
    model.eval()
    min_value = min_val
    #latent_z[:,dim1].min() + 1
    max_value = max_val
    #latent_z[:,dim1].max() + tol
    latent_code = latent_z[sample_to_manipulate]
    latent_code_gpu = z_gpu[sample_to_manipulate]## this is to obtain attention maps with the same z ##################
    x1 = torch.linspace(min_value,max_value, num_points)
    print(f"Linspace: {x1}")
    num_points = x1.size(0)
    z = to_cuda_variable(torch.from_numpy(np.asarray(latent_code))) # 1 sampled point with size 1x64 and 64:latent dim
    z = z.repeat(num_points, 1)
    z_gpu_interp = latent_code_gpu.repeat(num_points, 1) ##########Gradcam
    z[:, dim1] = x1.contiguous()
    z_gpu_interp[:, dim1] = x1.contiguous()
    z_1D = z_gpu_interp
    outputs = model.decode(z)  # If you added sigmoid after decoder's last layer remove here if not add here
    print(f"Shape of outputs {outputs.shape}")
    deneme = outputs
    outputs = outputs[:,:,:,:,slc] # to show the middle slice for each image
    interp = make_grid(outputs.cpu(), nrow=num_points, pad_value=1.0)
    return interp, deneme, z_1D

def one_feat_radiomic_extractor(img_dir, mask_dir, rad_name):
    
     rad_name_split = rad_name.strip().split("_")
     rad_class = rad_name_split[1]
     rad_name = rad_name_split[2]
     #arg = rad_class + "=" + "["+ "\"" + rad_name + "\""+"]"
     radiomics_extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
     radiomics_extractor.disableAllFeatures()
     #radiomics_extractor.enableFeaturesByName(glszm = ["{arg2}".format(arg2 = rad_name) ]) #
     radiomics_extractor.enableFeatureClassByName(featureClass=rad_class) #enable the class radiomics feat. you want to extract belongs to
     img = sitk.ReadImage(img_dir,sitk.sitkFloat32 )
     mask = sitk.ReadImage(mask_dir,sitk.sitkFloat32 )
     
     rad_value_df = pd.DataFrame([ radiomics_extractor.execute(img, mask, label=1) ] )
     sub_df = rad_value_df.filter(regex=rad_name)
     rad_value = sub_df
     #rad_value = np.int(rad_value_df.iloc[:,-1].values)
     rad_value = int(sub_df.values)
     return rad_value



def threshold_3D_GCAM(gcam_map):
    scaler = MinMaxScaler(feature_range = (0,1)) #

    gcam_mask_Arr = gcam_map[0][0]
    gcam_mask_Arr[gcam_mask_Arr<0.3] = 0.0
    gcam_mask_Arr_ = np.zeros((80,80,80))
    for slc in range(80):

#
          gcam_mask_Arr_[:,:,slc] = scaler.fit_transform(gcam_mask_Arr[:,:,slc])
    gcam_mask_Arr_[gcam_mask_Arr_<=0.75] = 0.0
    gcam_mask_Arr_[gcam_mask_Arr_>=0.75] = 1.0
    return gcam_mask_Arr_

def IoU_gcam_scar(gcam_mask_Arr_, scar_img):
    scar_only_img = scar_img[0,0,...].copy()
    scar_only_img[scar_only_img < 2.5] = 0.0
    scar_only_img[scar_only_img >= 2.5] = 1.0
    #IoU calculation

    intersection = np.logical_and(scar_only_img, gcam_mask_Arr_)
    union = np.logical_or(scar_only_img, gcam_mask_Arr_)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def show_(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), figsize=(25,25), squeeze=False)
    for i, ax in enumerate(axs.flatten()): 
        img = imgs[i]
        img = img.detach()
        img = TF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), cmap = "gray")
        axs[0, i].axis("off")
    plt.show()

def linear_interpolate(code1, code2, alpha): #returns  interpolation between two points
    return code1 * alpha + code2 * (1 - alpha)

def make_latent_interp_animation(code1, code2 ,model, num_interps, slc): #between 2 points
    model.eval()
    code_1 = to_cuda_variable(torch.from_numpy(np.asarray(code1)))
    img1 = model.decode(code_1.unsqueeze(0))
    img1_ = img1.detach().cpu().numpy()

    code_2 = to_cuda_variable(torch.from_numpy(np.asarray(code2)))
    img2 = model.decode(code_2.unsqueeze(0))
    img2_ = img2.detach().cpu().numpy()

    step_size = 1.0/num_interps
    
    all_imgs = []
    
    amounts = np.arange(0, 1, step_size)
    all_imgs.append(img2[0,0,:,:,slc])
    for alpha in tqdm(amounts):
        interpolated_latent_code = linear_interpolate(code1, code2, alpha)
        interpolated_latent_code_ = to_cuda_variable(torch.from_numpy(np.asarray(interpolated_latent_code)))
        images = model.decode(interpolated_latent_code_.unsqueeze(0))
        #images_ = images.detach().cpu().numpy()
        interp_latent_image = images[0,0,:,:,slc]
        all_imgs.append(interp_latent_image)

    all_imgs.append(img1[0,0,:,:,slc])
    #interp = make_grid(all_imgs, nrow=num_interps, pad_value=1.0)
    
    return all_imgs

def show_gradcam(deneme, model, gcam, interp_dim,  attr_dim, the_slice = 40, visualize_ex = 8):
#the_slice = 40
# Visualize Results and Gradcam images
  n_examples = visualize_ex
  #visualize_ex = 10
  subplot_shape = [1, visualize_ex]
  fig, axes = plt.subplots(*subplot_shape, figsize=(25,25), facecolor='white')
  plt.subplots_adjust(wspace = 0.02, hspace=0.)
  items = range(0, visualize_ex) #*
  example = 0
  vis_ex = 0
  for item in range(deneme.shape[0]):
        x = deneme[item,:,:,:,:].unsqueeze(0)

        img_ = x.detach().cpu().numpy()  
        #######################################################################
        ### ATTENTION GENERATION ###############################################        
        model.eval()
        x_rec, mu, logvar, out_mlp= gcam.forward(x)
        model.zero_grad()
        if item==0:
                print(f"Input is interpolated w.r.t. the attribute = {interp_dim} / {attributes_tra.columns[interp_dim]} ")
                print(f"Attention map is being generated for the attribute = {attr_dim} / {attributes_tra.columns[attr_dim]}")   

        gcam.backward(mu, logvar, mu_avg, logvar_avg, attr_dim)
        gcam_map = gcam.generate() 
        #### DONE!!! ###########################################################
        gcam_map = gcam_map.detach().cpu().data.numpy()

        if vis_ex < visualize_ex: #and label ==0: ### To visualize MINF cases

            for row, (im, title) in enumerate(zip(
                [img_[0,0,:,:,the_slice]],
                ["Attention maps"],
            )):
                #cmap = 'gray' if row == 0  else 'jet'
                ax = axes[vis_ex]
                if isinstance(im, torch.Tensor):
                    im = im.cpu().detach()
                #if row==0:
                #    ax.imshow(im, cmap="gray")
                if row ==0:
                    ax.imshow( gcam_map[0,0,:,:,the_slice], alpha= 0.9, cmap="jet")
                    ax.imshow( img_[0,0,:,:,the_slice],alpha=0.7, cmap="gray")
                    

                #ax.set_title(title, fontsize=25)
                ax.axis('off')           
                #fig.colorbar(im_show, fraction=0.046, ax=ax)

            vis_ex += 1

        if example == n_examples:
           break


