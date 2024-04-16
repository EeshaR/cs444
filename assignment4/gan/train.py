import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import sample_noise, show_images, deprocess_img, preprocess_img

def train(D, G, D_solver, G_solver, discriminator_loss, generator_loss, 
          show_every=250, batch_size=128, noise_size=100, num_epochs=10, 
          train_loader=None, device=None):
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in train_loader:
            real_images = preprocess_img(x).to(device)

            # Discriminator step
            D_solver.zero_grad()
            logits_real = D(real_images)

            g_fake_seed = sample_noise(batch_size, noise_size, device)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images)

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            D_solver.step()

            # Generator step
            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size, device)
            fake_images = G(g_fake_seed)
            logits_fake = D(fake_images)

            g_error = generator_loss(logits_fake)
            g_error.backward()
            G_solver.step()

            if iter_count % show_every == 0:
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                imgs_numpy = deprocess_img(fake_images.data.cpu())
                show_images(imgs_numpy[0:16])
                plt.show()
            iter_count += 1
