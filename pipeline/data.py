import h5py
import torch
from torch.utils.data import Dataset
import numpy as np 
from torchvision import transforms
import os


class FlipSample():
    """
    Randomly flip image and mask
    """
    def __call__(self, sample):
        img = sample[0]
        mask = sample[1]
        opt = np.random.randint(3)
        if opt == 1:
            img = np.flip(img, axis=0)
            mask = np.flip(mask, axis=0)
        elif opt == 2:
            img = np.flip(img, axis=2)
            mask = np.flip(mask, axis=2)
        return img, mask


class RotSample():
    """
    Rotate image and mask in x-y plane with k (1,2,3) times of 90 degrees
    """
    def __call__(self, sample):
        img = sample[0]
        mask = sample[1]
        k = np.random.randint(4)
        if k:
            img = np.rot90(img, k, axes=(0,1))
            mask = np.rot90(mask, k, axes=(0,1))
        return img, mask


class GaussianNoise():
    """
    Add Gaussian noise 
    """
    def __init__(self, mu=0., std=1.):
        self.mu = mu
        self.std = std
    def __call__(self, sample):
        img = sample[0]
        mask = sample[1]
        opt = np.random.randint(2)
        if opt == 1:
            img = img + np.random.randn(img.shape[0], img.shape[1], img.shape[2])*self.std + self.mu
        return img, mask


class ToTensor():
    """
    Convert image and mask into tensors, image in shape [channel,x,y,z], mask in shape [x,y,z]
    """
    def __call__(self, sample):
        img = sample[0]
        img = np.expand_dims(img, axis=0)
        mask = sample[1]
        img = torch.from_numpy(img.copy())
        mask = torch.from_numpy(mask.copy())
        return img, mask


class GenerateData_Multiclass(Dataset):
    """
    Generate training and validation dataset for labeled multi-class data
    """
    def __init__(self, img_mask_name, unmask_label=None, crop_sz=(64,64,64), num_data=10000, normalize=True, transform=None):
        """
        Args:
        img_mask_name: list of name pairs of image and mask data, each pair is a tuple of (img_name, mask_name)
        unmask_label: label number for unlabeled area, None for fully labeled data
        crop_sz: cropping size 
        num_data: generated sample size
        normalize: if normalize raw image
        transform: data augmentation
        """
        self.img_all = {}
        self.mask_all = {}
        self.num_pairs = len(img_mask_name)
        self.unmask_label = unmask_label
        # for multiple image and mask pairs
        for i in range(self.num_pairs):
            curr_name = img_mask_name[i]
            assert os.path.exists(curr_name[0]) and os.path.exists(curr_name[1]), \
                'Image or mask does not exist!'
            # prepare image in shape [x, y, z]
            img = np.float32(h5py.File(curr_name[0],'r')['raw'][()])
            if normalize:
                img = (img - img.mean()) / img.std()
            self.img_all[str(i)] = img
            # prepare mask in shape [x, y, z]
            mask = np.float32(h5py.File(curr_name[1],'r')['raw'][()])
            self.mask_all[str(i)] = mask
        
        self.crop_sz = crop_sz
        self.num_data = num_data
        self.transform = transform

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        pair_idx = np.random.randint(self.num_pairs)
        img = self.img_all[str(pair_idx)]
        mask = self.mask_all[str(pair_idx)]
        sz = img.shape
        # generate data
        is_accept = False
        while is_accept is not True:
            loc_x = np.random.randint(0, sz[0]-self.crop_sz[0])
            loc_y = np.random.randint(0, sz[1]-self.crop_sz[1])
            loc_z = np.random.randint(0, sz[2]-self.crop_sz[2])
            sample_mask = mask[loc_x:loc_x+self.crop_sz[0], loc_y:loc_y+self.crop_sz[1], loc_z:loc_z+self.crop_sz[2]]
            sample_sz = sample_mask.shape
            if self.unmask_label is not None and sample_mask[sample_sz[0]//2,sample_sz[1]//2,sample_sz[2]//2]==self.unmask_label:
                continue
            else:
                is_accept = True
                sample_img = img[loc_x:loc_x+self.crop_sz[0], loc_y:loc_y+self.crop_sz[1], loc_z:loc_z+self.crop_sz[2]]        
        # data augmentation
        if self.transform is not None:
            sample_img, sample_mask = self.transform([sample_img, sample_mask])        
        return sample_img, sample_mask


if __name__ == "__main__":

    def view_data(save_name, img):
        img = img.numpy()
        with h5py.File(save_name, 'w') as f:
            dset = f.create_dataset('raw', data=img)
        return None

    img_mask_name = [('/groups/flyem/data/dingx/mask/data/trvol-250-1.h5', '/groups/flyem/data/dingx/mask/data/trvol-250-1-mask.h5')]
    Data = GenerateData_Multiclass(img_mask_name, crop_sz=(108,108,108), unmask_label=2, transform=transforms.Compose([FlipSample(), RotSample(), GaussianNoise(), ToTensor()]))

    print(len(Data))

    data = Data.__getitem__(100)
    img = data[0]
    print(img.shape)
    mask = data[1]
    print(mask.shape)

    save_path = '/groups/flyem/data/dingx/mask/data/'
    view_data(save_path+'sample_img.h5', img)
    view_data(save_path+'sample_mask.h5', mask)