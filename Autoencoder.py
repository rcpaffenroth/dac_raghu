# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/rcpaffenroth/dac_raghu/blob/main/Autoencoder.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="f6828844-3a74-49bd-8348-ceae45948da7"
# # Load data

# %% id="a5713c2d-0108-4708-9f0d-021e183f6d1e"
import glob
import os
import numpy as np
import pandas as pd

# %% id="9f5c3fcc-147d-4669-8ab2-3ddf16a88b4e"
data_1d = np.array(pd.read_csv('https://raw.githubusercontent.com/rcpaffenroth/dac_raghu/main/data_1d.txt'))
data_2d = np.array(pd.read_csv('https://raw.githubusercontent.com/rcpaffenroth/dac_raghu/main/data_2d.txt'))

# %% [markdown] id="22492944-6bb1-4fec-869c-768de5ae30ef"
# # Some basic plots

# %% id="e4157228-f808-4104-af5e-23b6761aff9f"
import matplotlib.pylab as py

# %% colab={"base_uri": "https://localhost:8080/", "height": 265} id="c4251bd4-e92d-4e0f-b1d0-d36b2b6ec80c" outputId="11d8716c-d2ad-46a9-8218-bf028bf5841f"
py.plot(data_1d.T);

# %% colab={"base_uri": "https://localhost:8080/", "height": 265} id="a3f48dac-07c9-4eb2-9303-a3a4e105160c" outputId="5220d40a-29f2-4177-efb8-834e30643cbc"
py.plot(data_2d.T);

# %% [markdown] id="1880ae29-6269-42a9-870f-fb52bb18cf36"
# # Auto-encoder

# %% [markdown] id="e63b90c0-0999-4ef3-b6c7-0198f463bc1d"
# Based on https://www.kaggle.com/code/sinclairg/pytorch-lightning-tutorial-autoencoders

# %% colab={"base_uri": "https://localhost:8080/"} id="m9GUN_74dOeT" outputId="293a90b5-57fe-4419-c540-cf88cc80170a"
# ! pip install pytorch_lightning

# %% id="7707890d-56e5-4cc8-a0bc-33a1bcd6ba33"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl

# %% id="ebbc9b74-d7dc-4dd6-bb72-88ed57f5b63e"
encoder = nn.Sequential(nn.Linear(100, 64), 
                        nn.ELU(), 
                        nn.Linear(64, 32), 
                        nn.ELU(), 
                        nn.Linear(32, 2))
decoder = nn.Sequential(nn.Linear(2, 32), 
                        nn.ELU(), 
                        nn.Linear(32, 64),
                        nn.ELU(), 
                        nn.Linear(64, 100))


# %% id="3cb147ea-bbe1-4256-9136-434f8f45d695"
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def training_step(self, batch, batch_idx):
        x = batch[0]
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        # Include extra logging here
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        return optimizer


# %% id="6f894137-0ec9-4c4d-9e55-6c5e5a0f9d5e"
ae = LitAutoEncoder(encoder, decoder)

# %% id="cb37907a-0f41-451d-b7ef-744107c1db89"
dataset = torch.utils.data.TensorDataset(torch.Tensor(data_2d))
train_loader = torch.utils.data.DataLoader(dataset)

# %% colab={"base_uri": "https://localhost:8080/", "height": 382, "referenced_widgets": ["915d05c1f046462ebe574b6b306b6110", "df24beefb77f4549b7beed2e23558980", "a5a6545605cb475fb4bd6db5d32b0b24", "5959c15bd28a4059b8952689490ebe2f", "a3075bdd01e047faa3f1246dabaedcc7", "4d5e173f3cdd465b9b40f8b23d270e8a", "4340e450c8c9444784e3ac188b9230df", "c81f9b0a5dee4db396442599cb295097", "faf8ff98c59b4951b62cd8ad4e1d5bfa", "a955f09104b54e37b262198a3807761c", "d4d3328dd8914927b619350b49b6f81c"]} id="7486ca78-5e1c-4f94-afb5-9015303aaa09" outputId="e3247ec9-d372-4fda-c69e-654d985bc905"
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model=ae, train_dataloaders=train_loader)

# %% id="2426e99b-7e1d-451f-923a-458a4ae8c2ab"
X =  torch.Tensor(data_2d)
H = encoder(X).detach()
X_hat = decoder(H).detach()

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="b33a6d09-d3dc-4b0d-9ac6-051f0876c3f8" outputId="0e973c54-ec5f-42f1-e90d-0a39a7f756db"
py.plot(X[0,:])
py.plot(X_hat[0,:])

# %% colab={"base_uri": "https://localhost:8080/", "height": 284} id="5adf1a52-805d-433a-8223-d06ba2a24e34" outputId="98a11c09-30c3-4b88-9f7a-13501be412f5"
py.plot(X[0,:]-X_hat[0,:])

# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="4b49ce86-d1d2-4e3e-8200-ff2a83207d55" outputId="407df450-ce35-43b4-fc90-b08b4a071b60"
py.scatter(H[:,0], H[:,1])

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} id="2d620d61-259c-4ae4-aebb-ea4bf2e3cbce" outputId="52958329-64a6-498f-a386-2c275493cd2b"
h = encoder(X[0,:])
print(h)
x_hat = decoder(h).detach()
py.plot(x_hat)

x_hat = decoder(h+torch.tensor([0.5, 0.5])).detach()
py.plot(x_hat)


# %% colab={"base_uri": "https://localhost:8080/", "height": 282} id="546aaa84-980e-4519-8bde-beee7a2d61aa" outputId="45b83edd-0383-4181-cab9-c1cba9ef585d"
# h = torch.Tensor([2.8186, -2.0229])
h = torch.Tensor([2, -2])
x_hat = decoder(h).detach()
py.plot(x_hat)

# %% id="8ac074be-5d27-480a-83fe-e1161e9d69ef"
