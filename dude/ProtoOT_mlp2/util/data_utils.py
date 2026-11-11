import torchvision
import numpy as np
import faiss
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
from torchvision.transforms import InterpolationMode



def mine_nearest_neighbors(features, topk=50):
    #print("Computing nearest neighbors...")
    features = features.astype(np.float32)
    n, dim = features.shape[0], features.shape[1]
    index = faiss.IndexFlatIP(dim)
    #index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk + 1)  # Sample itself is included
    #print("Nearest neighbors computed.")
    return indices[:, 1:]



class NeighborsDataset_scan(Dataset):
    def __init__(self, dataset_image, indices_image):
        super(NeighborsDataset_scan, self).__init__()

        self.dataset_image = dataset_image
        self.indices_image = indices_image
        assert self.indices_image.shape[0] == len(self.indices_image)

    def __len__(self):
        return len(self.dataset_image)

    def __getitem__(self, index):
        anchor_image = self.dataset_image.__getitem__(index)
        neighbor_index_image = np.random.choice(self.indices_image[index], 1)[0]
        neighbor_image = self.dataset_image.__getitem__(neighbor_index_image)

        return anchor_image, neighbor_image
