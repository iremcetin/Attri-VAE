# -*- coding: utf-8 -*-
"""
@author: iremc
"""
import torch
from torch.nn import functional as F
import torch.nn as nn

def initialize_weights(m):
  if isinstance(m, nn.Conv3d):
     nn.init.xavier_uniform_(m.weight)
     m.bias.data.fill_(0.01)
  elif isinstance(m, nn.BatchNorm3d):
     nn.init.constant_(m.weight.data, 1)
     nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm1d):
     nn.init.constant_(m.weight.data, 1)
     nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
     nn.init.xavier_uniform_(m.weight.data)
     nn.init.constant_(m.bias.data, 0)
     
unflatten_channel = 2
dim_start_up_decoder = [5,5,5]
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Unflatten(nn.Module):
     def forward(self, input, size= unflatten_channel):
        return input.view(input.size(0), size, dim_start_up_decoder[0], dim_start_up_decoder[1], dim_start_up_decoder[2])

class ConvVAE(nn.Module):

    def __init__(self, image_channels, h_dim, latent_size, n_filters_ENC, n_filters_DEC):
        super(ConvVAE, self).__init__()
        self.image_channels = image_channels
        self.h_dim = h_dim
        self.latent_size = latent_size
        self.n_filters_ENC = n_filters_ENC
        self.n_filters_DEC = n_filters_DEC
        
    ##############
    ## ENCODER ##
    ##############
        self.conv1_enc = nn.Conv3d(in_channels = image_channels, out_channels = n_filters_ENC[0],  kernel_size = [3,3,3] , stride = 2, padding =1 )
        self.bn1_enc = nn.BatchNorm3d( n_filters_ENC[0])
        self.conv2_enc = nn.Conv3d(in_channels = n_filters_ENC[0], out_channels = n_filters_ENC[1], kernel_size = [3,3,3] , stride = 2, padding =1)
        self.bn2_enc = nn.BatchNorm3d( n_filters_ENC[1])
        self.conv3_enc = nn.Conv3d(in_channels = n_filters_ENC[1], out_channels = n_filters_ENC[2],  kernel_size = [3,3,3] , stride = 2, padding =1)
        self.bn3_enc = nn.BatchNorm3d( n_filters_ENC[2])
        self.conv4_enc = nn.Conv3d(in_channels = n_filters_ENC[2], out_channels = n_filters_ENC[3],  kernel_size = [3,3,3] , stride = 2, padding =1)
        self.bn4_enc = nn.BatchNorm3d( n_filters_ENC[3])
        self.conv5_enc = nn.Conv3d(in_channels = n_filters_ENC[3], out_channels = n_filters_ENC[4], kernel_size = [3,3,3], stride = 1 , padding =1)
        self.bn5_enc = nn.BatchNorm3d( n_filters_ENC[4])
        
        self.flatten = Flatten() 
             
        self.fc1 = nn.Linear(250, 128)     

        self.fc2 = nn.Linear(128, h_dim)
              
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.25)

        # icetin: hidden => mu
        self.mu = nn.Linear(h_dim, latent_size)
        # icetin: hidden => logvar
        self.logvar = nn.Linear(h_dim, latent_size)# icetin: same above

        # icetin: MLP
        self.mlp1 = nn.Linear(latent_size, int(latent_size/2))
        self.bn1_mlp = nn.BatchNorm1d(int(latent_size/2))
        self.mlp2 = nn.Linear(int(latent_size/2), int(latent_size/4)) 
        self.bn2_mlp = nn.BatchNorm1d(int(latent_size/4))
        self.mlp3 = nn.Linear(int(latent_size/4), 1) 
        #self.bn3_mlp = nn.BatchNorm1d(1)
        self.sigmoid_mlp = nn.Sigmoid()
    ###################
    ### END OF ENCODER
    ###################
      
    ###############
    ### DECODER 
    ###############
        #icetin: biffi et. al decoder, LVAE + MLP
        self.fc3 = nn.Linear(latent_size, 250) #icetin: pulls from bottleneck to hidden # dim_start_up_decoder = [5,5,5]
        self.unflatten = Unflatten()
        
        
        self.conv1_dec = nn.Conv3d(in_channels = unflatten_channel, out_channels = n_filters_DEC[0],  kernel_size = [3,3,3], stride = 1, padding =1)
        self.bn1_dec = nn.BatchNorm3d(n_filters_DEC[0])
        self.deconv1_dec = nn.ConvTranspose3d(in_channels =  n_filters_DEC[0], out_channels =  n_filters_DEC[0], kernel_size = [3,3,3], stride = 2, padding =1, output_padding=1)
        self.bn2_dec = nn.BatchNorm3d(n_filters_DEC[0])
        self.deconv2_dec = nn.ConvTranspose3d(in_channels =  n_filters_DEC[0], out_channels =  n_filters_DEC[1], kernel_size = [3,3,3], stride = 2, padding =1, output_padding=1)
        self.bn3_dec = nn.BatchNorm3d(n_filters_DEC[1])
        self.deconv3_dec = nn.ConvTranspose3d(in_channels =  n_filters_DEC[1], out_channels =  n_filters_DEC[2], kernel_size = [3,3,3], stride = 2, padding =1, output_padding=1)
        self.bn4_dec = nn.BatchNorm3d(n_filters_DEC[2])
        self.deconv4_dec = nn.ConvTranspose3d(in_channels =  n_filters_DEC[2], out_channels =  n_filters_DEC[3], kernel_size = [3,3,3], stride = 2, padding =1, output_padding=1)
        self.bn5_dec = nn.BatchNorm3d(n_filters_DEC[3])
        self.conv2_dec = nn.Conv3d(in_channels = n_filters_DEC[3], out_channels = n_filters_DEC[4], kernel_size = [3,3,3], stride = 1, padding =1)
        self.bn6_dec = nn.BatchNorm3d(n_filters_DEC[4])
        self.conv3_dec = nn.Conv3d(in_channels = n_filters_DEC[4], out_channels = IMG, kernel_size = [3,3,3], stride = 1, padding =1)
        self.bn7_dec = nn.BatchNorm3d(image_channels)
        

        self.sigmoid = nn.Sigmoid() # No need : sigmoid is used in the loss - when to set 'gaussian'
        #self.tanh = nn.Tanh()
   ##################
   ### END OF DECODER
   ##################

    def encode(self, x): # encoder returns mu and logvar
        
        h = F.relu(self.bn1_enc(self.conv1_enc(x)))
        h = F.relu(self.bn2_enc(self.conv2_enc(h)))
        h = F.relu(self.bn3_enc(self.conv3_enc(h)))
        h = F.relu(self.bn4_enc(self.conv4_enc(h)))
        h = F.relu(self.bn5_enc(self.conv5_enc(h)))

        h = self.dropout(self.flatten(h))
        
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        
        mu, logvar = self.mu(h), self.logvar(h)
        
        ####define the distribution from mu and logvar
        z_distribution = torch.distributions.Normal(loc=mu, scale=torch.exp(logvar))
        return mu, logvar, z_distribution

    def decode(self, z): # input of the decoder is z and returns reconstructed image.
        z = F.relu(self.fc3(z))
        z = self.unflatten(z) # 

        z = F.relu(self.bn1_dec(self.conv1_dec(z)))
        z = F.relu(self.bn2_dec(self.deconv1_dec(z)))
        z = F.relu(self.bn3_dec(self.deconv2_dec(z)))
        z = F.relu(self.bn4_dec(self.deconv3_dec(z)))
        z = F.relu(self.bn5_dec(self.deconv4_dec(z)))
        z = F.relu(self.bn6_dec(self.conv2_dec(z)))
        z = self.conv3_dec(z)

        z = self.sigmoid(z)
        
        return z

    def mlp_predict(self, z): #icetin: mlp part that is connected to z
        out_mlp = F.relu(self.bn1_mlp(self.mlp1(z))) # input: z output: prediction
        out_mlp = F.relu(self.bn2_mlp(self.mlp2(out_mlp)))
        out_mlp = self.sigmoid_mlp(self.mlp3(out_mlp))
        return out_mlp

    def reparameterize(self, mu, logvar, z_dist):
       # reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_sampled_eq = eps.mul(std).add_(mu) # sample
        
        # compute prior : normal distribution
        prior_dist = torch.distributions.Normal(loc=torch.zeros_like(z_dist.loc),scale=torch.ones_like(z_dist.scale)        )
        z_prior = prior_dist.sample()

        ### sample from the defined (in encoder) distribution
        z_tilde = z_dist.rsample() # implemented reparameterization trick
        return z_tilde, z_sampled_eq, z_prior, prior_dist
 
    def reparameterize_eval(self, mu, logvar):
        #print("REPARAMETERIZE EVAL...")
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    def z_return(self, z):
        return z

    def forward(self, x): # forward prop of the network.
        mu, logvar, z_dist = self.encode(x) # encoder returns mu and sigma
        
        z_tilde, z_sampled_eq, z_prior, prior_dist  = self.reparameterize(mu, logvar, z_dist) # reparameterization trick returns sample, z
        out_mlp = self.mlp_predict(z_tilde) # mlp branch takes z and outputs the predictions

        output = self.decode(z_tilde) # before z_sampled_eq was inputted
        return output, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist # reconstructed x, mu, logvar, mlp output