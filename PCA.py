# %% First - Encode all images
from train import *
from sklearn.decomposition import PCA
mean = []

def create_matrix(model_number):
    out = np.zeros((70000, 512))
    epoch, vae, opt, l1, l2 = load(f"D:\\Github\\misc\\VAE\\models\\number_{model_number}\\checkpoint.pth")
    vae.eval()
    for i in range(sub_datasets):
        print(f'Dataset n° : {i + 1}')
        j = 0
        batches = batch_provider(data[i], batch_size, process_batch, 4, 16, True)
        for image_batch in batches:
            with torch.no_grad():
                mu, logvar = vae.encode(image_batch)
                mu = mu.squeeze()
                logvar = logvar.squeeze()
                z = vae.reparameterize(mu, logvar).cpu().detach().numpy()
                n, _ = z.shape
                out[10000*i + batch_size*j : 10000*i + batch_size*j + n, :] = z
                j += 1
    np.save(f"D:\\Github\\misc\\VAE\\models\\number_{model_number}\\to_pca.npy", out)

def create_mean(model_number):
    encoded = np.load(f"D:\\Github\\misc\\VAE\\models\\number_{model_number}\\to_pca.npy")
    print(np.mean(encoded, axis = 0).shape)
    np.save(f"D:\\Github\\misc\\VAE\\models\\number_{model_number}\\mean.npy", np.mean(encoded, axis = 0))

# %%
def make_pca(model_number):
    pca = PCA(n_components=512)
    path = f"D:\\Github\\misc\\VAE\\models\\number_{model_number}\\"
    pca.fit(np.load(path +"to_pca.npy"))
    values = np.sqrt(pca.explained_variance_)
    vectors = pca.components_
    np.save(path + "eigenvalues.npy",values)
    np.save(path + "eigenvectors.npy",vectors)