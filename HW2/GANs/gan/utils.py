import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt
import torchvision


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    ##################################################################
    # TODO: 1.2: Generate and save out latent space interpolations.
    # 1. Generate 100 samples of 128-dim vectors. Do so by linearly
    # interpolating for 10 steps across each of the first two
    # dimensions between -1 and 1. Keep the rest of the z vector for
    # the samples to be some fixed value (e.g. 0).
    # 2. Forward the samples through the generator.
    # 3. Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    ##################################################################
    # pass
    # n_samples = 100
    # z_dim = 128

    # latent_vectors = torch.zeros(n_samples, z_dim, device="cuda")

    # for i in range(10):
    #     alpha = i / 9.0
    #     latent_vectors[i*10:(i+1)*10, 0] = (1 - alpha) * (-1) + alpha * 1
    #     latent_vectors[i*10:(i+1)*10, 1] = (1 - alpha) * (-1) + alpha * 1

    # generated_data = gen.forward_given_samples(latent_vectors)
    # generated_data = (generated_data+1)/2
    # torch.torchvision.utils.save_image(generated_data, path)
    
    z = torch.zeros(100,128).cuda()
    
    a = torch.linspace(-1, 1, 10)
    b = torch.linspace(-1, 1, 10)
    xs, ys = torch.meshgrid(a, b)
    
    z[:,0] = xs.reshape(-1).cuda()
    z[:,1] = ys.reshape(-1).cuda()

    generated_data = gen.forward_given_samples(z)
    generated_data = (generated_data + 1)/ 2
    torchvision.utils.save_image(generated_data, path)

    # return generated_data



    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    # parser.add_argument("--disable_amp", action="store_false")
    args = parser.parse_args()
    return args
