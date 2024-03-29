## Disclaimer

I intend on making a better version of this later on. The code here is really messy and should only be regarded as a proof on concept

# VAE
Variable Autoencoder with PCA, a GUI and sliders to generate faces.

Trained and created from scratch with Pytorch, sklearn, tkinter.

### Data

The images used to train the model come from the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset)

### Model

Convolutional `encoder`
Deconvolutional `decoder`

`(3,128,128) -> [...] -> (512) -> [...] -> (3,128,128)`

### PCA

Principal Components Analysis is done on the `(#images, latent_space)` sized matrix which contains all 70000 encoded images as `512` sized vectors. 
  

### Results

A screenshot of the GUI, with sliders, `Reset` and `Random` buttons:

![](https://github.com/thomktz/VAE/blob/main/results/sliders_reset.PNG)

This is the `mean` face, i.e. average face. It is set as a zero for all sliders.




A random generated face :

![](https://github.com/thomktz/VAE/blob/main/results/sliders_random.PNG)




And here are a few interesting sliders' effects (reminder: these were made and sorted by the PCA):

`slider_0`, interpretation : Background color (the strongest eigenvalue)
![](https://github.com/thomktz/VAE/blob/main/sliders/slider_0.png)

`slider_1`, interpretation : Face orientation
![](https://github.com/thomktz/VAE/blob/main/sliders/slider_1.png)

`slider_2`, interpretation : Light provenance
![](https://github.com/thomktz/VAE/blob/main/sliders/slider_2.png)

`slider_3`, interpretation : Hair color
![](https://github.com/thomktz/VAE/blob/main/sliders/slider_3.png)

`slider_4` and `slider_5` : interpretation : Sex + light provenance
![](https://github.com/thomktz/VAE/blob/main/sliders/slider_4.png)
![](https://github.com/thomktz/VAE/blob/main/sliders/slider_5.png)

`slider_15` and `slider_19`, interpretation : Smile + chin size/face tilt
![](https://github.com/thomktz/VAE/blob/main/sliders/slider_16.png)
![](https://github.com/thomktz/VAE/blob/main/sliders/slider_19.png)

Other sliders have effects on sex, skin color, smile, face width, and a lot of sliders only affect the background




