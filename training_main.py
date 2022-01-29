import data_load_feature_extract
from tra_val_split_oversample import tra_val_oversample
from feature_preprocessing import feature_preprocessing_main
from feature_selector_analysis import feature_sel
from dataset_dataloader import dataset_dataloader
from training_testing_functions import train, test
from model import ConvVAE, initialize_weights
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

n_filters_ENC = (8, 16, 32, 64, 2)
n_filters_DEC = (64, 32, 16, 8, 4, 2)
#Hyperparameters ###########################################
recon_param = 1.0
beta = 2.0  # [0.02, 0.001, 0.0001], -> (0.55, 0.0275, 0.00275)  # this values was taken from biffi et. al. = beta value of the beta-VAE 
alpha = 1 # alpha * mlp_loss # 
# Initialize the parameters for AR-LOSS
gamma = 10.0 # AR-LOSS = gamma * reg_loss
factor = 100.0 # this is the factor in AR-loss to scale the regularized latent dimension
print(f"Parameter List: \n recon_param = {recon_param} ; beta = {beta} ; alpha = {alpha} ; gamma = {gamma} ; factor = {factor} \n epoch number is {epochs} and latent dimension is {latent_size} ")
path_tosave_nets ="/content/gdrive/MyDrive/Colab Notebooks/Radiomics_latent_visualization/January2022/FINAL_Old_Model_oldSETUP_INTERPRETABLE_FEATS_TraVal_2901" + "/Experiment1_WHOLE_betaVAE_mlp_ar_woL1_VarianceMI_onlyVolume_woCapWoAnnealing" + "_epochs%d_batchs%d"%(epochs, batch_size) + "_beta%2f_alpha%2f_gamma%2f_factor%2f_reconparam%d_latentsize%d_batchsize%d"%(beta,alpha,gamma,factor,recon_param, latent_size, batch_size) + "/"
check_dir(path_tosave_nets)
###################################################################################################################
training_path = "/content/gdrive/MyDrive/EMIDEC/Emidec_DATASET/Training"  
#1. Load data and extract attributes(clinical, interpretable and radiomics)
images_tra_, segs_tra_, labels_tra, indices, interpretable_feats_tra, radiomics_tra, clinical_info_training = data_load_feature_extract(training_path)
#2. Define training and validation samples and attributes
train_files, val_files, train_segs, val_segs, train_labels,  val_labels, train_interpretable, val_interpretable, train_clinical_info, val_clinical_info, indices_train, indices_val = tra_val_oversample(images_tra_, segs_tra_, labels_tra, indices, interpretable_feats_tra, do_oversampling = True)
#3. Preprocessing of the features
interpretable_train_scl, interpretable_val_scl, clinical_info_train_scl, clinical_info_val_scl = feature_preprocessing_main(train_interpretable, val_interpretable, train_clinical_info, val_clinical_info)
#4. Feature selection, analysis (Radiomics) - returns to a dataframe with selected attributes
attribute_tra_scl, attribute_val_scl, attributes_tra_scl, attributes_val_scl = feature_sel(clinical_info_train_scl, clinical_info_training, interpretable_train_scl, interpretable_val_scl, train_labels, val_labels, add_feat = True)
#5. Define datasets (and data loaders)
train_ds, train_loader, val_ds, val_loader = dataset_dataloader(train_files, val_files, attributes_tra_scl, attributes_val_scl, train_clinical_info,val_clinical_info, train_labels, val_labels, win_size, batch_size, num_workers )


def main_train(HDIM, IMG, latent_size, epoch, is_L1, use_AR_LOSS,  resume = False):
 
 train_losses =[]
 test_losses = []
 acc_metrices = []
 auc_metrices = []

 start_epoch = 0
 best_test_loss = np.finfo('f').max
 best_test_loss_epoch = -1
 best_metric = -1
 best_metric_epoch = -1
 best_auc = -1
 best_auc_epoch = -1
 
 model = ConvVAE(image_channels=IMG, h_dim=HDIM, latent_size= latent_size, n_filters_ENC, n_filters_DEC).to(device)
 model.apply(initialize_weights) ### Initializa the weights
 optimizer = optim.Adam(model.parameters(), lr=learning_rate) 


 if resume:
   
    resume_path = path_tosave_nets + "checkpoint.pth"
    print('=> loading checkpoint %s' % resume)
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch'] + 1
    best_test_loss = checkpoint['best_test_loss']
    best_test_loss_epoch = checkpoint['best_test_loss_epoch']
    best_metric = checkpoint['best_metric']
    best_metric_epoch = checkpoint['best_metric_epoch']
    best_auc = checkpoint['best_auc']
    best_auc_epoch = checkpoint['best_auc_epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('=> loaded checkpoint %s' % resume)
   
 for epoch in range(start_epoch, epochs):

        train_loss = train(epoch, model, train_loader, optimizer, is_L1 = is_L1)
        test_loss, acc_metric, auc_metric = test(epoch, model, val_loader) 
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        acc_metrices.append(acc_metric)
        auc_metrices.append(auc_metric)
        print('Epoch [%d/%d] train loss: %.3f validation loss: %.3f' % (epoch + 1, epochs, train_loss, test_loss))
        print(f"Validation accuracy is {acc_metric} & AUC is {auc_metric} in epoch {epoch+1}")
        ############### Save checkpoint 
        checkpoint = { 'epoch': epoch + 1, 'best_test_loss': best_test_loss, 'best_test_loss_epoch': best_test_loss_epoch, 'best_metric': best_metric,'best_metric_epoch':  best_metric_epoch, 'best_auc': best_auc, 'best_auc_epoch':  best_auc_epoch, 'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
        save_ckp(checkpoint, path_tosave_nets)
        #is_best = test_loss < best_test_loss ### COMMENTED OUT @27.08
        if acc_metric > best_metric:
            best_metric = acc_metric
            best_metric_epoch = epoch + 1
            print(f"Best accuracy is achieved {best_metric} in the epoch: {best_metric_epoch}")
            torch.save(model.state_dict(), path_tosave_nets +  "best_metric_model.pth")
        ### Save the model with the best loss ###########################
        #best_test_loss = min(test_loss, best_test_loss) # all loss # COMMENTED OUT @27.08
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_loss_epoch = epoch+1
            print(f"Best loss is achieved {best_test_loss} in the epoch: {best_test_loss_epoch}")
            torch.save(model.state_dict(), path_tosave_nets + "best_test_loss_model.pth")
        if auc_metric > best_auc:
            best_auc = auc_metric
            best_auc_epoch = epoch+1
            print(f"Best AUC is achieved {best_auc} in the epoch: {best_auc_epoch}")
            torch.save(model.state_dict(), path_tosave_nets +  "best_AUC_model.pth")
 print("OUTCOME")
 print(f"Best accuracy of {best_metric} was achieved in epoch {best_metric_epoch}")
 print(f"Best loss of {best_test_loss} was achieved in epoch {best_test_loss_epoch}")
 print(f"Best AUC of {best_auc} was achieved in epoch {best_auc_epoch}")

 return model, train_losses, test_losses, acc_metrices, auc_metrices, train_loader, val_loader


      
