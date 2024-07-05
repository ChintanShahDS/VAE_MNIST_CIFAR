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
import torchvision.transforms as transforms

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

        # Resnet model reduces the size so incrase to 32
        totallen = 32
        # print("Label shape:", y.shape)
        # print("Shape x:", x.shape)
        padReqEachSide = int((totallen-x.shape[-1])/2)
        # print("padReqEachSide:", padReqEachSide)
        # Pad x to increase to totallen as needed by Resnet to avoid shrinking in decoder
        xpadded = F.pad(x, (padReqEachSide,padReqEachSide, padReqEachSide,padReqEachSide))
        # print("xpadded shape:", xpadded.shape)
        # Increase to 3 channels
        updatedx = torch.cat((xpadded,xpadded,xpadded), 1)
        # print("Data labels:", data[1])
        y_hot_encoded = F.one_hot(y.long(),10)
        # print("One hot encoded vector shape:", y_hot_encoded.shape)
        # print("One hod encoded:", y_hot_encoded[0])
        curlen = y_hot_encoded.shape[-1]
        y_hot_encoded_pad = F.pad(y_hot_encoded, (0,totallen-curlen))
        # print("One hod encoded padded shape:", y_hot_encoded_pad.shape)
        # print("One hod encoded padded:", y_hot_encoded_pad[0])
        y_hot_encoded_reshaped = y_hot_encoded_pad.reshape([y_hot_encoded_pad.shape[0],1,1,y_hot_encoded_pad.shape[-1]])
        # print("y_hot_encoded_reshaped shape:", y_hot_encoded_reshaped.shape)
        # Increase to 3 channels
        updatedy = torch.cat((y_hot_encoded_reshaped,y_hot_encoded_reshaped,y_hot_encoded_reshaped), 1)
        concatdata = torch.cat((updatedx,updatedy), -2)
        # print("Concatenaned end:", concatdata.shape)
        # print("Concatenaned [0]:", concatdata[0])

        return updatedx, concatdata

    def training_step(self, batch, batch_idx):
        x, y = batch

        updatedx, updatedInput = self.getUpdatedInput(x, y)

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(updatedInput)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded 
        x_hat = vae.decoder(z)
        # print("mu shape:", mu.shape)
        # print("std shape:", std.shape)
        # print("q shape:", q.shape)
        # print("z shape:", z.shape)
        # print("xhat shape:", x_hat.shape)
        # print("updated x shape:", updatedx.shape)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, updatedx)

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
        updatedx, updatedInput = self.getUpdatedInput(x, y)

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(updatedInput)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded 
        x_hat = self.decoder(z)

        return updatedx, x_hat

trainset = torchvision.datasets.MNIST(root='./data', 
                                        train=True, 
                                        download=True,
                                        transform=transforms.Compose([transforms.ToTensor()]))
trainloader = DataLoader(trainset,
                            batch_size=16, num_workers=16,
                            shuffle=True)
# images, labels = next(iter(trainloader))
# plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0) / 2 + 0.5); 
# plt.title(' '.join(trainset.classes[label] for label in labels)); plt.show(



print("------------------------------------------------------------------------------------------------------")

pl.seed_everything(1234)

vae = VAE()
mu = 0.0
std = 0.0

# DATA
# we're pretending to have an image from mnist (1 channel, 28x28 pixels)
x = torch.rand(1, 3, 28, 28)

# print('image shape:', x.shape)

# GET Q(z|x) PARAMETERS
# encode x to get the mu and variance parameters
x_encoded = vae.encoder(x)
mu, log_var = vae.fc_mu(x_encoded), vae.fc_var(x_encoded)

print('mu:', mu.shape)
print('log_var:', log_var.shape)

# SAMPLE Z from Q(Z|x)
std = torch.exp(log_var / 2)
q = torch.distributions.Normal(mu, std)
z = q.rsample()

print('z shape:', z.shape)


trainer = pl.Trainer(max_epochs=50, accelerator="auto") # progress_bar_refresh_rate=10, 
trainer.fit(vae, trainloader)

trainloader = DataLoader(trainset,
                    batch_size=1, num_workers=16,
                    shuffle=True)

vae.eval()

from matplotlib.pyplot import imshow, figure, imsave
from torchvision.utils import make_grid
from matplotlib import pyplot as plt 

xfull = None
# create figure 
fig = plt.figure(figsize=(3.5, 20)) 

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
    xnew = xnew.to(vae.device)
    ynew = ynew.to(vae.device)
    # print("xnew:", xnew)

    # predict with the model
    with torch.no_grad():
        # encode x to get the mu and variance parameters
        # decoded 
        updatedx, x_hat = vae.inference(xnew,ynew)
        updatedx = updatedx.cpu()
        x_hat = x_hat.cpu()
        # print("pred shape:", x_hat.shape)

    # print("x_hat shape:", x_hat.shape)
    # print("x_hat:", x_hat)

    x_concat = torch.cat((x,x_hat),0)
    gridOutput = make_grid(x_concat,nrow=10).permute(1, 2, 0).numpy()

    gridOutput[gridOutput>1.0] = 1.0
    gridOutput[gridOutput<0.0] = 0.0

    # Adds a subplot at the 1st position 
    fig.add_subplot(rows, columns, i+1) 
    # classnames = ', '.join(f'{classes[j]:5s}' for j in ynew.cpu().tolist())
    classnames = "Labels passed\nOrig:" + str(y.item()) + ", " + str(ynew.cpu().tolist())

    # showing image 
    plt.imshow(gridOutput) 
    plt.axis('off') 
    plt.title(classnames, fontsize=6, color= 'black', fontweight='bold') 
  
    if i >= 25:
        print("x_concat shape:", x_concat.shape)
        print("gridOutput after shape:", gridOutput.shape)
        break

plt.tight_layout()
filename = 'fullMnistOutput.png'

fig.savefig(filename)
