# %%
import torch
from torchvision import datasets, transforms
from dlutils import batch_provider
from imageio import imread
import numpy as np

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


image_size = 128
batch_size = 64
sub_datasets = 7
PATH = "D:\\Github\\Data\\images\\ImageFolder\\"
test_PATH = "D:\\Github\\Data\\subset\\ImageFolder\\"

data = [[PATH + f"{i}".zfill(5)+ ".png" for i in range(j*10000, (j + 1) * 10000)] for j in range(7)]


# %%
def process_batch(batch):
    #print(batch[0])
    images = [imread(x) for x in batch]
    #print(images[0])
    data = [x.transpose((2, 0, 1)) for x in images]
    x = torch.from_numpy(np.asarray(data, dtype=np.float32)) / 127.5 - 1.
    x = x.view(-1, 3, image_size, image_size)
    return x
