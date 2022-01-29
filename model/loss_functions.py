from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F

def mean_accuracy(targets, weights):
        """
        Evaluates the mean accuracy in prediction

        """
        weights = weights.squeeze()
        score_roc_auc = 0.0
        predictions = torch.zeros_like(weights)
        predictions[weights >= 0.5] = 1
        binary_targets = torch.zeros_like(targets)
        binary_targets[targets >= 0.5] = 1
        correct = predictions == binary_targets
        acc = torch.sum(correct.float()) /binary_targets.view(-1).size(0)
        try:
            score_roc_auc = roc_auc_score(binary_targets.detach().cpu().numpy(), predictions.detach().cpu().numpy())
        except ValueError:
            pass
        return acc,  score_roc_auc
    
def reconstruction_loss(recon_x, x, recon_param , dist):
    BCE = torch.nn.BCELoss(reduction="sum") 
    batch_size = recon_x.shape[0]
    if dist == 'bernoulli':

            recons_loss = BCE(recon_x, x) / batch_size
    elif dist == 'gaussian':
            x_recons = recon_x
            recons_loss = F.mse_loss(x_recons, x, reduction='sum') /batch_size
    else:
        raise AttributeError("invalid dist")
    return recon_param * recons_loss

def KL_loss(mu, logvar, z_dist, prior_dist, beta, c=0.0):
    
    # KL divergence loss
    KLD_1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ############################

    KLD_2 = torch.distributions.kl.kl_divergence(z_dist, prior_dist)
    KLD_2 = KLD_2.sum(1).mean()
    KLD_2 = beta * (KLD_2 - c).abs()
    return beta * KLD_1, KLD_2


def mlp_loss_function(y, out_mlp, alpha):
    criterion = torch.nn.BCELoss()
    #batch_size = out_mlp.shape[0]
    targets = y.reshape(-1,1)
    mean_loss = criterion( out_mlp.type(torch.float),  targets.type(torch.float) ) 
    
    return alpha * mean_loss

def reg_loss_sign(latent_code, attribute, factor=1.0):
        # compute latent distance matrix
        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        #print(f"latent code shape: {latent_code.shape}")
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

        # compute attribute distance matrix
        attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
        #print(attribute.shape)
        attribute_dist_mat = (attribute - attribute.transpose(1, 0)).view(-1, 1)

        # compute regularization loss
        loss_fn = torch.nn.L1Loss()
        lc_tanh = torch.tanh(lc_dist_mat * factor) # factor: tunable hyperparameter
        attribute_sign = torch.sign(attribute_dist_mat)
        sign_loss = loss_fn(lc_tanh, attribute_sign.float())

        return sign_loss

def reg_loss(latent_code, radiomics_, mini_batch_size, gamma = 1.0, factor = 1.0):
    AR_loss = 0.0
    for dim in range(radiomics_.shape[1]):
        x = latent_code[:, dim]
        radiomics_dim = radiomics_[:, dim]    
        AR_loss += reg_loss_sign(x, radiomics_dim, factor=factor )
    return gamma * AR_loss