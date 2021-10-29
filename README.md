# Spatial-Transformer-Networks

Repository with the implementation in PyTorch of a visual attention mechanism called Spatial Transformer
Networks (STN) for MNIST classification.
Based on the paper: "Spatial transformer networks". Max Jaderberg et al., Advances in neural information processing systems, 2015, vol. 28, p. 2017-2025. https://arxiv.org/abs/1506.02025 

![Header](images/STN.PNG)

STNs allows a neural network to perform spatial manipulation on the input data within the network to enhance the geometric invariance of the model. 

CNNs are not invariant to rotation and scale and more general affine transformations. In STNs the localization network is a regular CNN which regresses the transformation parameters. The transformation network learns automatically the spatial transformations that enhances the global accuracy on a specific dataset.

STNs can be simply inserted into existing convolutional architectures without any extra training supervision or modification to the optimization process.

![Example](images/CoordConv.PNG)

Modify the convolutional layers by adding information about the coordinates of the input images to the input tensor. The CoordConv layer is designed to be used a substitute of the regular Conv2D layer.

![Example](./images/MNIST_example.png)

Results of image warping for the MNIST dataset. 


