import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

class Encoder(nn.Module):
    """
    Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (3): ReLU()
        (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (5): ReLU()
        (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    """
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        ##################################################################
        # TODO 2.1: Set up the network layers. First create the self.convs.
        # Then create self.fc with output dimension == self.latent_dim
        ##################################################################
        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,kernel_size=(3,3), stride = (1,1), padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32,64,kernel_size=(3,3), stride = (2,2), padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,kernel_size=(3,3), stride = (2,2), padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,256,kernel_size=(3,3), stride = (2,2), padding=(1,1)),
            # torch.nn.ReLU(),
        )
        self.output_dim = input_shape[1]* input_shape[2] * 4
        self.fc = torch.nn.Linear(self.output_dim, self.latent_dim)
        # self.fc = torch.nn.Linear(self.latent_dim, self.output_dim)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, x):
        ##################################################################
        # TODO 2.1: Forward pass through the network, output should be
        # of dimension == self.latent_dim
        ##################################################################
        # pass
        x = self.convs(x)
        x = x.view(x.size(0),-1)
        # x = x.flatten(1)
        x = self.fc(x)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        ##################################################################
        # TODO 2.4: Fill in self.fc, such that output dimension is
        # 2*self.latent_dim
        ##################################################################
        output_dim = input_shape[1]* input_shape[2] * 4
        self.fc = torch.nn.Linear(output_dim, 2*latent_dim)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, x):
        ##################################################################
        # TODO 2.1: Forward pass through the network, should return a
        # tuple of 2 tensors, mu and log_std
        ##################################################################
        x = self.convs(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        mu, log_std = torch.split(x, self.latent_dim, dim=1)
        # log_std = None
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        return mu, log_std


class Decoder(nn.Module):
    """
    Sequential(
        (0): ReLU()
        (1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (2): ReLU()
        (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (4): ReLU()
        (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (6): ReLU()
        (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    """
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        ##################################################################
        # TODO 2.1: Set up the network layers. First, compute
        # self.base_size, then create the self.fc and self.deconvs.
        ##################################################################
        self.base_size = (256,self.output_shape[1]//8,self.output_shape[2]//8)
        # self.base_size = (256,4,4)
        
        self.deconvs = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=(4,4), stride = (2,2), padding = (1,1)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=(4,4), stride = (2,2), padding = (1,1)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=(4,4), stride = (2,2), padding = (1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 3, kernel_size=(3,3), stride = (1,1), padding = (1,1)),
        )
        # self.fc = torch.nn.Linear(self.latent_dim, self.base_size*self.base_size*256)
        # self.fc = torch.nn.Linear(np.prod(self.base_size),latent_dim)
        self.fc = torch.nn.Linear(latent_dim, np.prod(self.base_size))
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, z):
        #TODO 2.1: forward pass through the network, 
        ##################################################################
        # TODO 2.1: Forward pass through the network, first through
        # self.fc, then self.deconvs.
        ##################################################################
        # pass
        z = self.fc(z)
        z = z.reshape(z.size(0),*self.base_size)
        
        z = self.deconvs(z)
        return z
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
    # NOTE: You don't need to implement a forward function for AEModel.
    # For implementing the loss functions in train.py, call model.encoder
    # and model.decoder directly.
