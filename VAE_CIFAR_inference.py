
import lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
import torchvision

class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=32):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim, 
            input_height=input_height, 
            first_conv=False, 
            maxpool1=False
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def getUpdatedInput(self, x, y):
        # print("Label shape:", data[1].shape)
        # print("Shape data[0]:", data[0].shape)
        # print("Data labels:", data[1])
        y_hot_encoded = F.one_hot(y.long(),10)
        # print("One hot encoded vector shape:", y_hot_encoded.shape)
        # print("One hod encoded:", y_hot_encoded[0])
        totallen = x.shape[-1]
        curlen = y_hot_encoded.shape[-1]
        y_hot_encoded_pad = F.pad(y_hot_encoded, (0,totallen-curlen))
        # print("One hod encoded padded shape:", y_hot_encoded_pad.shape)
        # print("One hod encoded padded:", y_hot_encoded_pad[0])
        y_hot_encoded_reshaped = y_hot_encoded_pad.reshape([y_hot_encoded_pad.shape[0],1,1,y_hot_encoded_pad.shape[-1]])
        # print("y_hot_encoded_reshaped shape:", y_hot_encoded_reshaped.shape)
        y_hot_concat = torch.cat((y_hot_encoded_reshaped,y_hot_encoded_reshaped,y_hot_encoded_reshaped), 1)
        # print("y_hot_concat shape:", y_hot_concat.shape)

        concatdata = torch.cat((x,y_hot_concat), -2)
        # print("Concatenaned:", concatdata.shape)
        # print("Concatenaned [0]:", concatdata[0])

        return concatdata

    def training_step(self, batch, batch_idx):
        x, y = batch

        updatedInput = self.getUpdatedInput(x, y)

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(updatedInput)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded 
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(), 
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo

    def inference(self, x, y):
        updatedInput = self.getUpdatedInput(x, y)

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(updatedInput)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded 
        x_hat = self.decoder(z)

        return x_hat


import torchvision.transforms as transforms

trainset = torchvision.datasets.CIFAR10(root='./data', 
                                        train=True, 
                                        download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=1, num_workers=16,
                                          shuffle=True)
# images, labels = next(iter(trainloader))
# plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0) / 2 + 0.5); 
# plt.title(' '.join(trainset.classes[label] for label in labels)); plt.show(

model = VAE.load_from_checkpoint("./lightning_logs/version_23/checkpoints/epoch=29-step=93750.ckpt")

# disable randomness, dropout, etc...
model.eval()

from matplotlib.pyplot import imshow, figure, imsave
from torchvision.utils import make_grid
from matplotlib import pyplot as plt 

xfull = None
# create figure 
fig = plt.figure(figsize=(3.5, 20)) 
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# setting values to rows and column variables 
rows = 26
columns = 1

for i, data in enumerate(trainloader):
    x, y = data
    # print("x shape:", x.shape)
    # print("y shape:", y.shape)
    # print("y:", y)

    xnew = None
    ynew = None
    for yall in range(10):
        if yall != y.item():
            if xnew == None:
                xnew = x
                ynew= torch.tensor([yall])
            else:
                xnew = torch.cat((xnew, x),0)
                ynew=torch.cat((ynew, torch.tensor([yall])))

    # print("xnew shape:", xnew.shape)
    # print("ynew shape:", ynew.shape)
    xnew = xnew.to(model.device)
    ynew = ynew.to(model.device)
    # print("xnew:", xnew)

    # predict with the model
    with torch.no_grad():
        # encode x to get the mu and variance parameters
        # decoded 
        x_hat = model.inference(xnew,ynew).cpu()
        # print("pred shape:", x_hat.shape)

    # filename = 'output_'+str(i)+'.png'
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    # print("x_hat shape:", x_hat.shape)
    # print("x_hat:", x_hat)

    x_concat = torch.cat((x.cpu(),x_hat),0)
    gridOutput = make_grid(x_concat,nrow=10).permute(1, 2, 0).numpy() * std + mean

    gridOutput[gridOutput>1.0] = 1.0
    gridOutput[gridOutput<0.0] = 0.0

    # Adds a subplot at the 1st position 
    fig.add_subplot(rows, columns, i+1) 
    classnames = ', '.join(f'{classes[j]:5s}' for j in ynew.cpu().tolist())
    classnames = "Labels passed\nOrig:" + str(classes[y.item()]) + ", " + classnames

    # showing image 
    plt.imshow(gridOutput) 
    plt.axis('off') 
    plt.title(classnames, fontsize=6, color= 'black', fontweight='bold') 
  
    if i >= 25:
        print("x_concat shape:", x_concat.shape)
        print("gridOutput after shape:", gridOutput.shape)
        break

plt.tight_layout()
filename = 'fulloutput.png'

fig.savefig(filename)
