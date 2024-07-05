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

## Outputs

### VAE CIFAR 25 image outputs giving the non correct label as input along with the image to encoder
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/blob/main/CIFAR_Output_25x10_DiffLabel.png)

### VAE CIFAR outputs along with the input image with correct label
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/blob/main/CIFAR_output%20(1).png)
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/blob/main/CIFAR_output%20(2).png)
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/blob/main/CIFAR_output%20(3).png)
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/blob/main/CIFAR_output%20(4).png)
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/blob/main/CIFAR_output%20(5).png)
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/blob/main/CIFAR_output%20(6).png)

### VAE CIFAR output less training
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/blob/main/CIFAR_output.png)

### VAE MNIST 25 image outputs giving the non correct label as input along with the image to encoder
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/blob/main/MNIST_Output_25x10_DiffLabel.png)

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
- *.png
  - Various outputs

## Hyperparameters used
- CIFAR
  - Epochs=30
  - enc_out_dim=512
  - latent_dim=256
  - input_height=32
  - optimizer=torch.optim.Adam(self.parameters(), lr=1e-4)
  - batch_size=16
  - num_workers=16
- MNIST
  - Epochs=50
  - enc_out_dim=512
  - latent_dim=256
  - input_height=32 (Adjusted size of the image)
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

### Elbo MNIST training
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/assets/94432132/119610d7-887e-48e7-9694-f25b8e8659a6)

### KL Loss MNIST training
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/assets/94432132/d70f3dc7-90dd-4751-b06c-457ac29cf8a2)

### Recon loss MNIST training
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/assets/94432132/852e678e-4d4e-4710-9dcb-8138cb5099fb)

### Reconstruction MNIST training
![image](https://github.com/ChintanShahDS/VAE_MNIST_CIFAR/assets/94432132/008ef09b-2ad7-4276-8290-30c30445ce77)


## Observations
- Output with different labels do not differ
- When using 28 x 28 image the output we were getting was 24 x 24 causing issues with comparison (So switched to 32 x 32 by padding)
- Also facing issues due to single channel in MNIST as model was defined accordingly
  - Instead of changing the model copied the channels 3 times to create a 3 channel input
