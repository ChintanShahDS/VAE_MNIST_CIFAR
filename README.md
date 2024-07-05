# Variational Auto Encoder trained for CIFAR and MNIST datasets
Trained a Variational Auto Encoder using pytorch lightning using CIFAR and MNIST data

## Achievements
- label appended to the Encoder imput
- label hot encoded and resized to append
- Created a batch of images with different labels (except the original to be sent for inference to trained model
- Combined the input and output to be displayed on a grid and dumped into a output image with labels
- Padding to achieve the desired size of inputs like in MNIST
- Understanding the need of the ready model and Increasing the channel size to handle the same
- Handling the display of 1 channel to 3 channels and vice versa

## Files
- VAE_CIFAR.py
  - Training file for VAE CIFAR with label also appended along with image
  - Hotencoded the label and then resized to the image dimensions and added
- VAE_CIFAR_inference.py
  - Inference for CIFAR VAE trained using the earlier code
  - Pass a different label to see what we get as outcome for each of the images
  - Handled the multiple nuances with passing not correct labels and displaying the outcomes in a proper format with labels
- VAE_MNIST.py
  - Training file for VAE MNIST with label also appended along with image - On same lines as VAE_CIFAR
  - Hotencoded the label and then resized to the image dimensions and added
  - Combined the inference part and generation of output images also internally in this
  - Additional Complexity to convert the MNIST to 3 channels as well as to 32 by padding due to the Resnet model used by pl_bolts implementation
  - Also comparison of the output with the input for loss calculation
  - End to show the outputs in a proper format with the changed dimensions

## Hyperparameters used
- Epochs=30
- enc_out_dim=512
- latent_dim=256
- input_height=32
- optimizer=torch.optim.Adam(self.parameters(), lr=1e-4)
- batch_size=16
- num_workers=16

## Hardware used
- Google Colab T4
- Some experimentation on CPU as well

## Detailed Training detials
- Important Excerpts from the tensorboard logs

### Elbo CIFAR training
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/assets/94432132/b885302b-46bc-469c-8fdb-c86cd6423505)

### KL Loss CIFAR training
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/assets/94432132/9efc34bd-8034-46e7-8e97-c35bdeaa34a4)

### Recon loss CIFAR training
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/assets/94432132/c0869e8e-8152-4b95-9f48-8f43ef1a59cb)

### Reconstruction CIFAR training
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/assets/94432132/c1c268df-bd2b-430b-8eaa-b15c5707087b)


## Observations
- Output with different labels do not differ
- 
