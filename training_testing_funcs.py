### Training #####
import torch
import shutil
cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')

device = torch.device("cuda" if cuda else "cpu")

def save_ckp(state,checkpoint_dir):
    f_path = checkpoint_dir + 'checkpoint.pth'
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

def train(epoch, model, train_loader, optimizer, is_L1, use_AR_LOSS ):
    model.train()
    roc_last = 0
    acc_last = 0
   
    train_loss = 0
    iter = 0
    for batch_idx, (data, label, rad_, clinic, fname) in enumerate(train_loader):
        iter = iter + 1
        data = data.to(device)
        label = label.to(device)
        rad_ = rad_.to(device)
        clinic_ = clinic.to(device)
        optimizer.zero_grad()
        # 1. Forward #############################################################################
        recon_batch, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist = model(data)
        #########################################################
        #z_sampled_es : sampled using reparam trick implemented from mu and logvar
        #z_tilde : sampled using torch reparam trick from z_dist
        ########################################################
        
        # 2. Loss ################################################################################
        recon_loss = reconstruction_loss(recon_batch, data,recon_param, dist = 'gaussian')
        mlp_loss = mlp_loss_function(label, out_mlp, alpha)
        
        kl_loss1, kl_loss2 = KL_loss(mu, logvar, z_dist, prior_dist, beta, c=0.0)
        loss = recon_loss + mlp_loss +  kl_loss2

        if use_AR_LOSS :
            attr_reg_loss = reg_loss(z_tilde, rad_, len(data), gamma = gamma, factor = factor)
            #print(f" Attr Reg Los {attr_reg_loss}")
            loss += attr_reg_loss
        # 2.1 Weight Regularization #############################################################
        ## L1 Regularization 
        if (is_L1==True):
      
            l1_crit = nn.L1Loss(reduction="sum")
            weight_reg_loss = 0
            for param in model.parameters():
                weight_reg_loss += l1_crit(param,target=torch.zeros_like(param))

            fctr = 0.00005
            loss += fctr * weight_reg_loss
            #train_losses.append(loss)

        else:
            pass
        #print(f"Total loss {loss}")
        # 3. Backward ###########################################################################
        loss.backward()
        # 4. Update #############################################################################
        optimizer.step()

        accuracy, roc = mean_accuracy(label, out_mlp)
        
        acc_last +=accuracy
        roc_last += roc
        train_loss += loss

    train_loss = train_loss/iter
    print(f"Training Accuracy (%) {acc_last/iter} AUC: {roc_last/iter} in epoch {epoch+1}")
    return train_loss

### Validating ####
def test(epoch, model, test_loader):
    model.eval()
    test_loss_ = 0
    iter = 0
    test_loss = 0
    acc_metric = 0
    acc_value = 0
    auc_value=0
    auc_metric = 0
    with torch.no_grad():
        for batch_idx, (data_test, label, rad_, clinic, fname) in enumerate(test_loader):
            
            iter = iter + 1
            
            data_test = data_test.to(device)
            label = label.to(device)
            rad_ = rad_.to(device)
            clinic_ = clinic.to(device)
            recon_batch, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist = model(data_test)
            
            recon_loss = reconstruction_loss(recon_batch, data_test,recon_param, dist = 'gaussian')
            kl_loss1, kl_loss2 = KL_loss(mu, logvar, z_dist, prior_dist, beta, c=0.0)
            mlp_loss = mlp_loss_function(label, out_mlp, alpha)
            loss_ = recon_loss + mlp_loss +  kl_loss2

            if use_AR_LOSS :
                
                attr_reg_loss = reg_loss(z_tilde, rad_, len(data_test), gamma = gamma, factor = factor)
                loss_ += attr_reg_loss
            if (epoch == 1) or (epoch == 50) or (epoch  == 100) or (epoch == 200): 
                  print(f" Training AR loss is {attr_reg_loss} and MLP loss {mlp_loss} and reconstruction loss {recon_loss} and KL loss {kl_loss2}")
            test_loss_ += loss_
            accuracy,  roc = mean_accuracy(label,  out_mlp)

            acc_value += accuracy
            auc_value += roc

           
    
    
    acc_metric = acc_value /iter
    auc_metric = auc_value /iter
    test_loss = test_loss_ /iter
    return test_loss, acc_metric, auc_metric
