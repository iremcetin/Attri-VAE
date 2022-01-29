
import numpy as np
from __future__ import print_function
from collections import OrderedDict
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import os

def default_normalizer(x) -> np.ndarray:
    """
    A linear intensity scaling by mapping the (min, max) to (0, 1).

    N.B.: This will flip magnitudes (i.e., smallest will become biggest and vice versa).
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    scaler = ScaleIntensity(minv=0.0, maxv=1.0)
    x = [scaler(x) for x in x]
    return np.stack(x, axis=0)

class PropBase(object):

    def __init__(self, model, target_layer, cuda=True):
        self.model = model
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
        self.model.eval()
        self.target_layer = target_layer
        self.outputs_backward = OrderedDict()
        self.outputs_forward = OrderedDict()
        self.set_hook_func()

    def set_hook_func(self):
        raise NotImplementedError

    # set the target latent dim as one others as zero. use this vector for back prop
    def encode_one_hot_batch(self,z, idx):
         #one_hot = torch.FloatTensor(1, self.n_class).zero_()
         one_hot_batch = torch.FloatTensor(z.size()).zero_()
         one_hot_batch[0][idx] = 1.0
         return one_hot_batch

    def forward(self, x): # run the VAE and return the outputs
        self.image_size = x.size(-1) 
        recon_batch, self.mu, self.logvar, self.out_mlp, self.z_sampled_eq, self.z_prior, self.prior_dist, self.z_tilde, self.z_dist = self.model(x) 
        return recon_batch, self.mu, self.logvar, self.out_mlp

    # back prop the one_hot signal
    def backward(self, mu, logvar, mu_avg, logvar_avg, idx):
        self.model.zero_grad()
        z = self.model.reparameterize_eval(mu, logvar).cuda() # use the mu and logvar from forward pass and sample z         
        one_hot = self.encode_one_hot_batch(z, idx)# this returns mu
        #print(f"selected z dimension's value (w.r.t. idx= {idx}) {z[0,idx]}")

        if self.cuda:
            one_hot = one_hot.cuda()


        flag=2
        if flag == 1:
            self.score_fc = torch.sum(F.relu(one_hot * z))
            #print(f" flag==1 and score_fc is : {self.score_fc}")
        else:
            #self.score_fc = torch.sum(one_hot)
            self.score_fc = z[0,idx] # backprop the selected z
            #print(f" flag !=1 and score_fc is : {self.score_fc}")
        
        self.score_fc.backward(retain_graph=True)

    def get_conv_outputs(self, outputs, target_layer): # this function outputs the selected conv layer's output : Feature maps
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        
                        return value
        raise ValueError('invalid layer name: {}'.format(target_layer))

class attribute_wise_attn(PropBase):

    def set_hook_func(self):
        def func_b(module, grad_in, grad_out):
            self.outputs_backward[id(module)] = grad_out[0].cpu()

        def func_f(module, input, f_output):
            self.outputs_forward[id(module)] = f_output

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)
            module[1].register_forward_hook(func_f)

    def normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.item()

    def compute_gradient_weights(self):
        self.grads = self.normalize(self.grads) #icetin:  grads.size() = [1,2,5,5,5] [batch_size, selected_layers_output_channel, 5,5,5]
        #print(f" grads inside compute_gradient_weights: {self.grads.size()}")
        self.map_size = self.grads.size()[2:] # map_size = [1,2, 5,5,5]
        #print(f" map_size inside compute_gradient_weights: {self.map_size}")
        self.weights = nn.AvgPool3d(self.map_size)(self.grads) #this is the weight

    def generate(self):
        # icetin: get gradient 
        self.grads = self.get_conv_outputs(self.outputs_backward, self.target_layer)

        self.compute_gradient_weights()

        # icetin: get activation 
        self.activation = self.get_conv_outputs(self.outputs_forward, self.target_layer)

        self.weights.volatile = False
        attn_map = F.conv3d(self.activation, weight = (self.weights.cuda()), stride=1, padding=0, groups=len(self.weights))    
        attn_map = F.upsample(attn_map, (self.image_size, self.image_size, self.image_size), mode="trilinear") #icetin: Upsample the attention map to the size of the image
        attn_map = torch.relu(attn_map) # use ReLu
        return attn_map

