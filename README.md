# Attri-VAE: attribute-based, disentangled and interpretable representations of medical images with variational autoencoders


### Training
<img src="https://user-images.githubusercontent.com/26603738/151681620-7a4f7705-cd40-4802-bf7a-62344e65725e.png" width="800" height="600">
Training framework of the Attri-VAE with the corresponding loss functions shown in red arrows. (a) shows the losses that are computed for each data sample: MLP loss , KL loss and reconstruction loss , and (b) shows the loss that is computed inside a training batch : attribute-regularization loss. The input, a 3D image, first goes through the 3D convolutional encoder which learns to map to the low dimensional space by outputting mean and variance of the latent space distributions. The decoder, then takes Z and outputs the reconstruction of the original input. The predicted classes of the inputs, y_c, are computed with MLP that consists of three fully connected (FC) layers. The corresponding MLP loss function is computed between y_c and the ground truth label y_{GT}. In (b), L_AR is shown to regularize the first dimension of the latent space with the attribute, a_1 (a_1 and a_2 represent the first and the second attributes).

### Testing 
<img src="https://user-images.githubusercontent.com/26603738/151681675-06c0f620-c596-4d7f-8d81-8a04e0e881c9.png" width="800" height="600">
The trained network offers two use cases: (a) shows latent space manipulation and (b) demonstrates attribute-wise attention generation. For a given 3D data sample, X,  the trained 3D convolutional encoder, outputs two vectors: Z is sampled with the reparameterization trick. (a) demonstrates the new data generation process by changing only first and second regularized latent dimensions of Z which correspond to two different data attributes (volume and max 2D diameter, respectively). Then the decoder, generates 3D outputs using the manipulated latent vectors Z_1 and Z_2, respectively. In (b) attribute-wise attention map generation is demonstrated for a given attribute that was encoded in the first latent dimension (Z^1). First, (Z^1) is backpropagated to the encoder´s last convolutional layer to obtain the gradient maps with respect to the feature maps. The gradient maps of measure the linear effect of each pixel in the corresponding feature map on the latent values. After that we compute the weights using GAP on each gradient map. A heat map is generated by multiplying these values with the corresponding feature map, summing them up and applying a ReLU. Finally we upsample the heat map and overlay it with the input image to obtain the superimposed image (3D attention map).  Additionally, the class score, y_c of the input is computed with MLP that is connected to latent_vector. Note that, in the figure it is assumed that the last convolutional layer of the encoder has 2 feature maps.

#### Architectural Details of Attri-VAE

<img src="https://user-images.githubusercontent.com/26603738/158028398-ddebe493-1e85-4d0f-bf96-0ee7b69cd941.png" width="800" height="400">
Architectural details of the proposed Attri-VAE. Conv: convolutional layer, Trconv : transposed convolutional layer,  BN: batch normalization,  fc: fully connected layer, ReLU: Rectified linear unit.


<img src="https://user-images.githubusercontent.com/26603738/158028645-a2b3db86-f9fe-47e5-a7ae-9b5cb90a1105.PNG"  width="700" height="600">
Conv3D: 3-dimensional convolutional layer, Trconv3D: 3-dimensional transposed convolutional layer, input: input channels, output: output channels, ks: kernel size, s: stride, pad: padding, BN: batch normalization, d: dropout probability, ReLU: Rectified linear unit.
