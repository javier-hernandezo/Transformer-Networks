# Spatial-Transformer-Networks
Baseline and improved versions (using CoordConv) of a STN for MNIST classification


![Header](images/STN.PNG)

The localization network is a regular CNN which regresses the transformation parameters. The transformation network learns automatically the spatial transformations that enhances the global accuracy on a specific dataset.

![Example](images/CoordConv.PNG)

Modify the convolutional layers by adding information about the coordinates of the input images to the input tensor. The CoordConv layer is designed to be used a substitute of the regular Conv2D layer.

![Example](./images/MNIST_example.png)

Results of image warping for the MNIST dataset. 


