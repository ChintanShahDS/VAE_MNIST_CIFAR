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
        x_hat = vae.decoder(z)

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


import torchvision.transforms as transforms

trainset = torchvision.datasets.CIFAR10(root='./data', 
                                        train=True, 
                                        download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
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

# for data in trainloader:
#         # encode x to get the mu and variance parameters
#     # print("Data first:", data)
#     print("Label shape:", data[1].shape)
#     print("Shape data[0]:", data[0].shape)
#     print("Data labels:", data[1])
#     y_hot_encoded = F.one_hot(data[1].long(),10)
#     print("One hot encoded vector shape:", y_hot_encoded.shape)
#     print("One hod encoded:", y_hot_encoded[0])
#     totallen = data[0].shape[-1]
#     curlen = y_hot_encoded.shape[-1]
#     y_hot_encoded_pad = F.pad(y_hot_encoded, (0,totallen-curlen))
#     print("One hod encoded padded shape:", y_hot_encoded_pad.shape)
#     print("One hod encoded padded:", y_hot_encoded_pad[0])
#     y_hot_encoded_reshaped = y_hot_encoded_pad.reshape([y_hot_encoded_pad.shape[0],1,1,y_hot_encoded_pad.shape[-1]])
#     print("y_hot_encoded_reshaped shape:", y_hot_encoded_reshaped.shape)
#     y_hot_concat = torch.cat((y_hot_encoded_reshaped,y_hot_encoded_reshaped,y_hot_encoded_reshaped), 1)
#     print("y_hot_concat shape:", y_hot_concat.shape)

#     concatdata = torch.cat((data[0],y_hot_concat), -2)
#     print("Concatenaned:", concatdata.shape)
#     print("Concatenaned [0]:", concatdata[0])
#     # mu, log_var = vae.fc_mu(x_encoded), vae.fc_var(x_encoded)
#     # std = torch.exp(log_var / 2)
#     # print("Shape:", len(data))
#     # print("Data:", data)
#     # print(data[0])
#     # print(type(data[0]))
#     break

# DATA
# we're pretending to have an image from cifar-10 (3 channels, 32x32 pixels)
x = torch.rand(1, 3, 32, 32)

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

trainer = pl.Trainer(max_epochs=30, accelerator="auto") # progress_bar_refresh_rate=10, 
trainer.fit(vae, trainloader)

from matplotlib.pyplot import imshow, figure, imsave
from torchvision.utils import make_grid
figure(figsize=(8, 3), dpi=300)

# print("mu:", mu, "std:", std, "mu shape:", mu.shape, "std.shape:", std.shape)

# Z COMES FROM NORMAL(0, 1)
num_preds = 16
p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
# print("p:", p)
z = p.rsample((num_preds,))
print("z shape:", z.shape)

# SAMPLE IMAGES
with torch.no_grad():
    pred = vae.decoder(z.to(vae.device)).cpu()
    print("pred shape:", pred.shape)

# UNDO DATA NORMALIZATION
# normalize = cifar10_normalization()
# mean, std = np.array(normalize.mean), np.array(normalize.std)
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
img = make_grid(pred).permute(1, 2, 0).numpy() * std + mean

# PLOT IMAGES
# imshow(img);

imsave('outputcifar.png', img)