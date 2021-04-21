# VAE
Variable Autoencoder with PCA, a GUI and sliders to generate faces
Trained and created from scratch with Pytorch.

### Data

The images used to train the model come from the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset)

### Model

Convolutional `encoder`
Deconvolutional `decoder`

`(3,128,128) -> [...] -> (512) -> [...] -> (3,128,128)`

### PCA

Principal Components Analysis is done on the `(#images, latent_space)` sized matrix which contains all 70000 encoded images as `512` sized vectors. Once sorted, the slider `i` has the following effect :

```python
out += slider[i] + eigenvalues[i] + eigenvectorsInverse[i]
```

### Results

A screenshot of the GUI, with sliders, `Reset` and `Random` buttons:
![](https://github.com/thomktz/VAE/blob/main/sliders_reset.PNG)

A random face :
![](https://github.com/thomktz/VAE/blob/main/sliders_random.PNG)

And here's a few interesting sliders' effects :
