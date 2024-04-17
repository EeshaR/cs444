import torch
import torch.nn as nn


class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        kernel_size, stride, padding, output_channels = 4, 2, 1, 128
        slope_coeff = 0.2

        self.sequence = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),

            nn.LeakyReLU(slope_coeff),
            
            nn.Conv2d(output_channels, output_channels * 2, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channels * 2),

            nn.LeakyReLU(slope_coeff),
            
            nn.Conv2d(output_channels * 2, (output_channels * 2) * 2, kernel_size, stride, padding),
            nn.BatchNorm2d((output_channels * 2) * 2),

            nn.LeakyReLU(slope_coeff),

            nn.Conv2d((output_channels * 2) * 2, ((output_channels * 2) * 2) * 2, kernel_size, stride, padding),
            nn.BatchNorm2d(((output_channels * 2) * 2) * 2),

            nn.LeakyReLU(slope_coeff),

            nn.Conv2d(((output_channels * 2) * 2) * 2, padding, kernel_size, stride - 1, padding - 1),
        )

        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        ##########       END      ##########

        x = self.sequence(x).view(-1)
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        kernel_size, stride, padding, new_output = 4, 2, 1, 1024
        rel_coeff = True

        self.sequence = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim, new_output, kernel_size, stride - 1, padding - 1),
            nn.BatchNorm2d(new_output),

            nn.ReLU(rel_coeff),
            
            nn.ConvTranspose2d(new_output, int(new_output / 2), kernel_size, stride, padding),
            nn.BatchNorm2d(int(new_output / 2)),

            nn.ReLU(rel_coeff),
            
            nn.ConvTranspose2d(int(new_output / 2), int((int(new_output / 2)) / 2), kernel_size, stride, padding),
            nn.BatchNorm2d(int((int(new_output / 2)) / 2)),

            nn.ReLU(rel_coeff),
            
            nn.ConvTranspose2d(int((int(new_output / 2)) / 2), int((int(new_output / 2)) / 4), kernel_size, stride, padding),
            nn.BatchNorm2d(int((int(new_output / 2)) / 4)),

            nn.ReLU(rel_coeff),
            
            nn.ConvTranspose2d(int((int(new_output / 2)) / 4), output_channels, kernel_size, stride, padding),
            nn.Tanh()
        )

        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        ##########       END      ##########

        x = self.sequence(x.view(-1, self.noise_dim, 1, 1))
        return x
