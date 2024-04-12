import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img

def train(
    D,
    G,
    D_solver,
    G_solver,
    discriminator_loss,
    generator_loss,
    show_every=250,
    batch_size=128,
    noise_size=100,
    num_epochs=10,
    train_loader=None,
    device=None,
):
    iter_count = 0
    for epoch in range(num_epochs):
        print("EPOCH: ", (epoch + 1))
        for x, _ in train_loader:
            x = x.to(device)
            _, input_channels, img_size, _ = x.shape

            real_images = preprocess_img(x)  # Assuming preprocess_img normalizes the images

            ####################################
            #      Discriminator step          #
            ####################################
            D_solver.zero_grad()

            noise = sample_noise(batch_size, noise_size).to(device)
            fake_images = G(noise)
            
            logits_real = D(real_images)
            logits_fake = D(fake_images.detach())

            d_error = discriminator_loss(logits_real, logits_fake)
            d_error.backward()
            D_solver.step()

            ####################################
            #        Generator step            #
            ####################################
            G_solver.zero_grad()

            logits_fake = D(fake_images)
            g_error = generator_loss(logits_fake)
            g_error.backward()
            G_solver.step()

            ##########       END      ##########

            # Logging and output visualization
            if iter_count % show_every == 0:
                print(
                    "Iter: {}, D: {:.4}, G:{:.4}".format(
                        iter_count, d_error.item(), g_error.item()
                    )
                )
                disp_fake_images = deprocess_img(fake_images.data)
                imgs_numpy = disp_fake_images.cpu().numpy()
                show_images(imgs_numpy[0:16], color=input_channels != 1)
                plt.show()
            iter_count += 1
