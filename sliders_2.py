#%%%
from train import load
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import tkinter as tk
import numpy as np
import torch
import time
# %%

def load_all(model_number):
    path = f"D:\\Github\misc\\VAE\\models\\number_{model_number}\\"
    eigenvalues = np.load(path + "eigenvalues.npy")
    eigenvectors = np.load(path + "eigenvectors.npy")
    mean = np.load(path + "mean.npy")
    eigenvectorInverses = np.linalg.pinv(eigenvectors)
    e, vae, opt, l1, l2 = load(path + 'checkpoint.pth')
    return eigenvalues, eigenvectors, eigenvectorInverses, mean, vae

eigenvalues, eigenvectors, eigenvectorInverses, mean, vae = load_all(0)


width, height = 900, 512
nb_sliders = 20
update_time = 0.1
slider_height = 21
latent_space = 512
window = tk.Tk()
functions = [lambda x : slider_moves(x,u) for u in range(nb_sliders)]
window.geometry(f"{width}x{height}")
scales = []
ranges = [[-3, 3] for i in range(nb_sliders)]
last_update = 0


def calculate_image(settings):
    real_settings = mean.copy()
    for i in range(nb_sliders):
        real_settings += settings[i] * eigenvalues[i] * eigenvectors[i]
        real_settings = real_settings.reshape((1, latent_space))

        reconstructed_image = vae.decode(torch.Tensor(real_settings)).squeeze(0)
        #print(reconstructed_image.shape)
        reconstructed_image = np.array(reconstructed_image.detach())

        reconstructed_image = np.swapaxes(reconstructed_image, 0, 2)

        #plt.imshow(reconstructed_image)
    return reconstructed_image
   
def slider_moves(val):
    global last_update, update_time
    t = time.time()
    if t - last_update > update_time:
        values = get_all_values()
        array = calculate_image(values)
        #print(array.shape)
        image=Image.fromarray(((array* 0.5 + 0.5)*255).astype(np.uint8)).transpose(Image.ROTATE_270).resize((512,512))

        img =  ImageTk.PhotoImage(image)
        canvas = tk.Canvas(window,width=512,height=512)
        canvas.pack()
        canvas.place(x=0, y=0)
        canvas.create_image(0,0, anchor=tk.NW, image=img)
        last_update = time.time()
        window.mainloop()

def get_all_values():
    values = np.zeros(nb_sliders)
    for i in range(nb_sliders):
        values[i] = scales[i].get()
    return values


def button_reset():
    global scales
    for i in range(nb_sliders):
        scales[i].set(0)

def button_random():
    global scales
    rd = np.random.randn(nb_sliders)
    for i in range(nb_sliders):
        scales[i].set(rd[i])

for i in range(nb_sliders):
    scales.append(tk.Scale(window, command = slider_moves, showvalue = 0, from_ = ranges[0][1], length = 200, to = ranges[0][0], orient = tk.HORIZONTAL, resolution = 0.00001))
    scales[-1].pack()
    scales[-1].place(x = 550, y = (height-slider_height*(nb_sliders+2))/2 + i*slider_height)
# %%
reset = tk.Button(window, text = "Reset", command = button_reset)
reset.pack()
reset.place(x = 620, y = (height-slider_height*(nb_sliders+2))/2 + (nb_sliders+2)*slider_height)
random = tk.Button(window, text = "Random", command = button_random)
random.pack()
random.place(x = 680, y = (height-slider_height*(nb_sliders+2))/2 + (nb_sliders+2)*slider_height)
print((height-slider_height*(nb_sliders+2))/2 + (nb_sliders+1)*slider_height)

tk.mainloop()
# %%
image_size = 128

def what_slider_does(i, n_steps, maxi):
    print("Slider ", i)
    real_settings = mean.copy()
    settings = np.random.randn(nb_sliders)/5
    settings[i] = -maxi
    out = [settings[i] * eigenvalues[i] * eigenvectors[i]]
    for k in range(n_steps):
        settings[i] += 2* maxi / n_steps 
        real_settings = settings[i] * eigenvalues[i] * eigenvectors[i]
        out.append(real_settings.copy())
    x_rec = vae.decode(torch.Tensor(out))
    resultsample = x_rec * 0.5 + 0.5
    resultsample = resultsample.cpu()
    save_image(resultsample.view(-1, 3, image_size, image_size),'D:/Github/misc/VAE/sliders/slider_' + str(i) + '.png')
# %%
