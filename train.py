# %%
from model import *
from data_treatment import *
from torchvision.utils import save_image
import torch
from pynvml import *
from dlutils import batch_provider

def loss_function(recon_x, x, mu, logvar):
    BCE = torch.mean((recon_x - x)**2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD * 0.1
def print_gpu_memory():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')


def save(epoch, model, optimizer, rec_loss_list, kl_loss_list, path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rec_loss_list': rec_loss_list[:-1],
            'kl_loss_list': kl_loss_list[:-1],
            }, path)

def load(path):
    checkpoint = torch.load(path, map_location=device)
    autoencoder = VAE().to(device)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    rec_loss_list = checkpoint['rec_loss_list']
    kl_loss_list = checkpoint['kl_loss_list']
    return epoch, autoencoder, optimizer, rec_loss_list, kl_loss_list


# %%
learning_rate = 1e-3  #first run @ 1e-3
batch_size = 64
image_every = 1
save_every = 10
num_epochs = 1000
# %%

def train(model_number, new_epochs_number):
    torch.cuda.empty_cache()
    #print_gpu_memory()
    sample1 = torch.randn(128, 512).view(-1, 512, 1, 1)
    print(len(glob.glob(f"D:\\Github\\misc\\VAE\\models\\number_{model_number}\\")))
    if len(glob.glob(f"D:\\Github\\misc\\VAE\\models\\number_{model_number}\\")) == 0:
        print(f"Creating model number {model_number}")
        vae = VAE()
        vae = vae.to(device)
        vae.train()
        vae.weight_init(mean = 0, std = 0.02)
        optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate,betas = (0.5,0.999), weight_decay=1e-5)
        rec_loss_list = []
        kl_loss_list = []
        old_epoch = 0
        epoch=0
        #os.makedirs(f"models\\number_{model_number}", exist_ok=True)
        #save(epoch+old_epoch, vae, optimizer, rec_loss_list, kl_loss_list, f"models\\number_{model_number}\\checkpoint_{epoch+old_epoch}.pth")
    else:
        old_epoch, vae, optimizer, rec_loss_list, kl_loss_list = load(f"D:\\Github\\misc\\VAE\\models\\number_{model_number}\\checkpoint.pth")
        print(f"Succesfully loaded model number {model_number}, continuiing at epoch {old_epoch}")
    print('Training ...')
    start = time.time()
    try:
        vae.train()
        #print_gpu_memory()
        for epoch in range(new_epochs_number):
            kl_loss_list.append(0)
            rec_loss_list.append(0)
            num_batches = 1
            if (epoch+old_epoch + 1) % save_every == 0:
                os.makedirs(f"D:\\Github\\misc\\VAE\\models\\number_{model_number}", exist_ok=True)
                save(epoch+old_epoch, vae, optimizer, rec_loss_list, kl_loss_list, f"D:\\Github\\misc\\VAE\\models\\number_{model_number}\\checkpoint_{epoch+old_epoch}.pth")
            if (epoch+old_epoch + 1) % 5 == 0:
                optimizer.param_groups[0]['lr'] /= 4
                print("learning rate change")
            #print_gpu_memory()
            batches = batch_provider(data[epoch % sub_datasets], batch_size, process_batch, 4, 16, True)
            for image_batch in batches:
                vae.train()
                vae.zero_grad()
                image_batch = image_batch.to(device)
                rec, mu, logvar = vae(image_batch)
                loss_re, loss_kl = loss_function(rec, image_batch, mu, logvar)
                (loss_re + loss_kl).backward()
                optimizer.step()
                rec_loss_list[-1] += loss_re.item()
                kl_loss_list[-1] += loss_kl.item()       
                num_batches += 1
                #print(num_batches)
                if num_batches % print_every == 0:
                    #print(f"Batch no {num_batches}, loss : {loss.item()}")
                    pass
            print(num_batches)
            rec_loss_list[-1] /= 157
            kl_loss_list[-1] /= 157
            print('Epoch [%d / %d] rec loss: %f, RL loss : %f, elapsed time : %d minutes, remaining : %d' % (epoch+1+old_epoch, new_epochs_number+old_epoch, rec_loss_list[-1],kl_loss_list[-1], int((time.time()-start)/60), int((time.time()-start)/60*(1+new_epochs_number-epoch)/(epoch+1))))

            if (epoch + old_epoch) % image_every ==0:
                os.makedirs('D:\\Github\\misc\\VAE\\results_rec', exist_ok=True)
                os.makedirs('D:\\Github\\misc\\VAE\\results_gen', exist_ok=True)
                print("Saving images")
                with torch.no_grad():
                    vae.eval()
                    x_rec, _, _ = vae(image_batch)
                    resultsample = torch.cat([image_batch, x_rec]) * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, image_size, image_size),
                            'D:\\Github\\misc\\VAE\\results_rec\\sample_' + str(epoch+old_epoch) + '.png')
                    x_rec = vae.decode(sample1)
                    resultsample = x_rec * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, image_size, image_size),
                            'D:\\Github\\misc\\VAE\\results_gen\\sample_' + str(epoch+old_epoch) + '.png')


        os.makedirs(f"D:\\Github\\misc\\VAE\\models\\number_{model_number}", exist_ok=True)
        save(epoch+old_epoch, vae, optimizer, rec_loss_list, kl_loss_list, f"D:\\Github\\misc\\VAE\\models\\number_{model_number}\\checkpoint.pth")
    except KeyboardInterrupt:
        print("Training stopped.")
        os.makedirs(f"D:\\Github\\misc\\VAE\\models\\number_{model_number}", exist_ok=True)
        save(epoch+old_epoch, vae, optimizer, rec_loss_list, kl_loss_list, f"D:\\Github\\misc\\VAE\\models\\number_{model_number}\\checkpoint.pth")

#train(0, 100)


# %%
def show_loss(model_number):
    _ , _ , _ , rec_loss_list, kl_loss_list = load(f"D:\\Github\\misc\\VAE\\models\\number_{model_number}\\checkpoint.pth")
    fig = plt.figure()
    plt.plot(rec_loss_list[:-1])
    plt.plot(kl_loss_list[:-1])
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction error')
    plt.show()

#show_loss(0)
# %%
def to_img(x):
    #x = 0.5 * (x + 1)
    #x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualise_output(images, model):
    with torch.no_grad():
        images = images.to(device)
        images = model(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[1:15], 5, 3).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()
    
def reconstruct(model_number):

    old_epoch, autoencoder, optimizer, train_loss_avg = load(f"D:\\Github\\misc\\VAE\\models\\number_{model_number}\\checkpoint.pth")
    autoencoder.eval()
    images, _ = iter(testloader).next()

    # First visualise the original images
    print('Original images')
    show_image(torchvision.utils.make_grid(images[1:15],5,3))
    plt.show()

    # Reconstruct and visualise the images using the autoencoder
    print('Autoencoder reconstruction:')
    visualise_output(images, autoencoder)

#reconstruct(0)
# %%
