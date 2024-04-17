import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
import torch.nn.functional as F
import torch.nn as nn


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    ##########       END      ##########

    # using use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    lossr = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
    lossf = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))

    loss = (lossr + lossf) / 2
    return loss

    # referenced - 
    # https://stackoverflow.com/questions/68607705/binary-cross-entropy-with-logits-produces-negative-output


def generator_loss(logits_fake):
    """
    Computes the generator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    ##########       END      ##########

    # use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_fake, (torch.ones_like(logits_fake)))
    return loss

    # referenced - 
    # https://stackoverflow.com/questions/74900046/custom-bce-loss-giving-undefined-results
    # https://ai.stackexchange.com/questions/40699/loss-is-negative-dqn-with-bce-loss-function


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    ##########       END      ##########

    loss = 0.5 * ((F.mse_loss(scores_real, torch.ones_like(scores_real))) + (F.mse_loss(scores_fake, torch.zeros_like(scores_fake))))
    return loss

    # referenced - 
    # https://stackoverflow.com/questions/72274562/mseloss-from-pytorch
    # https://stackoverflow.com/questions/78154131/evaluation-metrics-of-mse-mae-and-rmse
    

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = None

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    ##########       END      ##########

    # L = 0.5 × mean((scores_fake)^2)
    loss = 0.5 * torch.mean(torch.pow(scores_fake - 1, 2))
    return loss
