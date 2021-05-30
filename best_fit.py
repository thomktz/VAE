# %%
from train import load
from data_treatment import *
import numpy as np
import torch
import torch.optim as optim
import PIL
from tqdm import trange

device = torch.device("cpu")
latent_space = 512

fit_path = "fit_images\\"
treated_path = "D:\\Github\\Data\\new_treated\\"

def load_all(model_number):
    path = f"D:\\Github\misc\\VAE\\models\\number_{model_number}\\"
    eigenvalues = np.load(path + "eigenvalues.npy")
    eigenvectors = np.load(path + "eigenvectors.npy")
    mean = np.load(path + "mean.npy")
    eigenvectorInverses = np.linalg.pinv(eigenvectors)
    e, vae, opt, l1, l2 = load(path + 'checkpoint.pth')
    return eigenvalues, eigenvectors, eigenvectorInverses, mean, vae.to(device)

eigenvalues, eigenvectors, eigenvectorInverses, mean, vae = load_all(1)


def fit(image_name, iter, previous = None):
    #img = PIL.Image.open(image_name+".png")
    #data = torch.Tensor(np.array(img)[:,:,:].transpose(2,0,1)/128-0.5).unsqueeze(0)
    priority_matrix = [ [1/(128+np.sqrt((i-32)**2 + (j-32)**2)) for i in range(128)] for j in range(128) ]
    priority_tensor = torch.Tensor([priority_matrix, priority_matrix, priority_matrix])
    data = process_batch([image_name+".png"])
    if previous is None:
        latent = torch.tensor(mean.copy().reshape((1, latent_space)), requires_grad = True).to(device)
    else:
        latent = previous
    optimizer = optim.Adam([latent], lr = 0.03)
    #print(type(latent))
    t = trange(iter, desc = "Loss")
    for n in t:
            
        optimizer.zero_grad()
        reconstructed_image = vae.decode(latent.float())
        if n<20 or n%50 == 0:
            out = ((np.array(reconstructed_image.squeeze(0).detach().cpu()).transpose(1,2,0)*0.5+0.5)*255).astype(np.uint8)
            matplotlib.image.imsave(fit_path + image_name + str(n) + ".png", out)
        #print(data.shape, reconstructed_image.shape)
        
        loss = (priority_tensor*((data - reconstructed_image)**2)).mean()
        t.set_description(str(loss.item()), refresh=True)
        loss.backward()
        optimizer.step()
    return latent

        
        


# %%
