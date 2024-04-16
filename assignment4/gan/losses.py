import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss using binary cross entropy loss.
    """
    true_labels = torch.ones_like(logits_real)
    fake_labels = torch.zeros_like(logits_fake)
    loss_real = bce_loss(logits_real, true_labels)
    loss_fake = bce_loss(logits_fake, fake_labels)
    loss = (loss_real + loss_fake) / 2
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss using binary cross entropy loss.
    """
    true_labels = torch.ones_like(logits_fake)  # We want to fool the discriminator
    loss = bce_loss(logits_fake, true_labels)
    return loss

def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    """
    loss_real = 0.5 * torch.mean((scores_real - 1) ** 2)
    loss_fake = 0.5 * torch.mean(scores_fake ** 2)
    loss = loss_real + loss_fake
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    """
    loss = 0.5 * torch.mean((scores_fake - 1) ** 2)
    return loss
