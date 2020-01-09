import h5py
import torch
from torch.utils.data import Dataset
import numpy as np 
from torchvision import transforms


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
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)
        return img, mask


class ToTensor():
    """
    Convert image and mask into tensor in shape [channel, x, y, z]
    """
    def __call__(self, sample):
        img = sample[0]
        mask = sample[1]
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        img = torch.from_numpy(img.copy())
        mask = torch.from_numpy(mask.copy())
        return img, mask


class GenerateData(Dataset):
    """
    Generate training and validation dataset
    """
    def __init__(self, img_name, mask_name, crop_sz=(64,64,64), num_data=10000, transform=None):
        """
        Args:
        img_name: name of image data
        mask_name: name of mask (1-positive samples, 0-negative samples, 2-unmasked)
        crop_sz: cropping size 
        num_data: generated sample size
        transform: data augmentation
        """
        self.img = np.float32(h5py.File(img_name,'r')['raw'][()])
        self.img = (self.img - self.img.mean()) / self.img.std()  # normalize image
        img_shape = self.img.shape
        self.mask = np.float32(h5py.File(mask_name, 'r')['raw'][()])
        assert img_shape == self.mask.shape, "Error: Image and mask must be in the same shape!"
        self.crop_sz = crop_sz
        self.num_data = num_data
        self.transform = transform
        idx = np.where(self.mask==1)
        self.pos_idx = list([idx[0][i],idx[1][i],idx[2][i]] for i in range(len(idx[0])) \
            if crop_sz[0]//2<idx[0][i]<img_shape[0]-crop_sz[0]//2 and crop_sz[1]//2<idx[1][i]<img_shape[1]-crop_sz[1]//2 and crop_sz[2]//2<idx[2][i]<img_shape[2]-crop_sz[2]//2)  # location of positive samples
        idx = np.where(self.mask==0)    
        self.neg_idx = list([idx[0][i],idx[1][i],idx[2][i]] for i in range(len(idx[0])) \
            if crop_sz[0]//2<idx[0][i]<img_shape[0]-crop_sz[0]//2 and crop_sz[1]//2<idx[1][i]<img_shape[1]-crop_sz[1]//2 and crop_sz[2]//2<idx[2][i]<img_shape[2]-crop_sz[2]//2)  # location of negative samples
    
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # positive sample
        i = np.random.randint(len(self.pos_idx))
        pos_vxl = self.pos_idx[i]
        pos_img = self.img[pos_vxl[0]-self.crop_sz[0]//2:pos_vxl[0]+self.crop_sz[0]//2, pos_vxl[1]-self.crop_sz[1]//2:pos_vxl[1]+self.crop_sz[1]//2, pos_vxl[2]-self.crop_sz[2]//2:pos_vxl[2]+self.crop_sz[2]//2]
        pos_mask = self.mask[pos_vxl[0]-self.crop_sz[0]//2:pos_vxl[0]+self.crop_sz[0]//2, pos_vxl[1]-self.crop_sz[1]//2:pos_vxl[1]+self.crop_sz[1]//2, pos_vxl[2]-self.crop_sz[2]//2:pos_vxl[2]+self.crop_sz[2]//2]
        # negative sample
        i = np.random.randint(len(self.neg_idx))
        neg_vxl = self.neg_idx[i]
        neg_img = self.img[neg_vxl[0]-self.crop_sz[0]//2:neg_vxl[0]+self.crop_sz[0]//2, neg_vxl[1]-self.crop_sz[1]//2:neg_vxl[1]+self.crop_sz[1]//2, neg_vxl[2]-self.crop_sz[2]//2:neg_vxl[2]+self.crop_sz[2]//2]
        neg_mask = self.mask[neg_vxl[0]-self.crop_sz[0]//2:neg_vxl[0]+self.crop_sz[0]//2, neg_vxl[1]-self.crop_sz[1]//2:neg_vxl[1]+self.crop_sz[1]//2, neg_vxl[2]-self.crop_sz[2]//2:neg_vxl[2]+self.crop_sz[2]//2]
        # data augmentation
        if self.transform is not None:
            pos_img, pos_mask = self.transform([pos_img, pos_mask])
            neg_img, neg_mask = self.transform([neg_img, neg_mask])
        
        return [pos_img, pos_mask], [neg_img, neg_mask]


# NOT efficient, didn't use
class GenerateData_Multi(Dataset):
    """
    Generate training and validation dataset using multiple image and mask pairs
    """
    def __init__(self, img_mask_name, crop_sz=(64,64,64), num_data=10000, transform=None):
        """
        Args:
        img_mask_name: list of name pairs of image and mask data, each pair is a tuple of (img_name, mask_name)
            For mask, 1-positive samples, 0-negative samples, 2-unmasked
        crop_sz: cropping size 
        num_data: generated sample size
        transform: data augmentation 
        """
        self.img_mask_name = img_mask_name
        self.crop_sz = crop_sz
        self.num_data = num_data
        self.transform = transform
    
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        pair_idx = np.random.randint(len(self.img_mask_name))
        pair_name = self.img_mask_name[pair_idx]
        img = np.float32(h5py.File(pair_name[0],'r')['raw'][()])
        img = (img - img.mean()) / img.std()  # normalize image
        img_shape = img.shape
        mask = np.float32(h5py.File(pair_name[1], 'r')['raw'][()])
        assert img_shape == mask.shape, "Error: Image and mask must be in the same shape!"
        idx = np.where(mask==1)
        pos_idx = list([idx[0][i],idx[1][i],idx[2][i]] for i in range(len(idx[0])) \
            if self.crop_sz[0]//2<idx[0][i]<img_shape[0]-self.crop_sz[0]//2 and self.crop_sz[1]//2<idx[1][i]<img_shape[1]-self.crop_sz[1]//2 and self.crop_sz[2]//2<idx[2][i]<img_shape[2]-self.crop_sz[2]//2)  # location of positive samples
        idx = np.where(mask==0)    
        neg_idx = list([idx[0][i],idx[1][i],idx[2][i]] for i in range(len(idx[0])) \
            if self.crop_sz[0]//2<idx[0][i]<img_shape[0]-self.crop_sz[0]//2 and self.crop_sz[1]//2<idx[1][i]<img_shape[1]-self.crop_sz[1]//2 and self.crop_sz[2]//2<idx[2][i]<img_shape[2]-self.crop_sz[2]//2)  # location of negative samples

        # positive sample
        i = np.random.randint(len(pos_idx))
        pos_vxl = pos_idx[i]
        pos_img = img[pos_vxl[0]-self.crop_sz[0]//2:pos_vxl[0]+self.crop_sz[0]//2, pos_vxl[1]-self.crop_sz[1]//2:pos_vxl[1]+self.crop_sz[1]//2, pos_vxl[2]-self.crop_sz[2]//2:pos_vxl[2]+self.crop_sz[2]//2]
        pos_mask = mask[pos_vxl[0]-self.crop_sz[0]//2:pos_vxl[0]+self.crop_sz[0]//2, pos_vxl[1]-self.crop_sz[1]//2:pos_vxl[1]+self.crop_sz[1]//2, pos_vxl[2]-self.crop_sz[2]//2:pos_vxl[2]+self.crop_sz[2]//2]
        # negative sample
        i = np.random.randint(len(neg_idx))
        neg_vxl = neg_idx[i]
        neg_img = img[neg_vxl[0]-self.crop_sz[0]//2:neg_vxl[0]+self.crop_sz[0]//2, neg_vxl[1]-self.crop_sz[1]//2:neg_vxl[1]+self.crop_sz[1]//2, neg_vxl[2]-self.crop_sz[2]//2:neg_vxl[2]+self.crop_sz[2]//2]
        neg_mask = mask[neg_vxl[0]-self.crop_sz[0]//2:neg_vxl[0]+self.crop_sz[0]//2, neg_vxl[1]-self.crop_sz[1]//2:neg_vxl[1]+self.crop_sz[1]//2, neg_vxl[2]-self.crop_sz[2]//2:neg_vxl[2]+self.crop_sz[2]//2]
        # data augmentation
        if self.transform is not None:
            pos_img, pos_mask = self.transform([pos_img, pos_mask])
            neg_img, neg_mask = self.transform([neg_img, neg_mask])
        
        return [pos_img, pos_mask], [neg_img, neg_mask]


if __name__ == "__main__":

    def view_data(save_name, img):
        img = img.numpy()
        img_ch0 = np.zeros(img.shape[1:], dtype=img.dtype)
        img_ch0 = img[0,:,:,:]
        with h5py.File(save_name, 'w') as f:
            dset = f.create_dataset('raw', img_ch0.shape, dtype=img_ch0.dtype)
            dset[:,:,:] = img_ch0
        return None

    # img_name = '/groups/flyem/data/dingx/mask/data/trvol-250-1.h5'
    # mask_name = '/groups/flyem/data/dingx/mask/data/trvol-250-1-mask.h5'
    # Data = GenerateData(img_name, mask_name, crop_sz=(108,108,108), transform=transforms.Compose([FlipSample(), RotSample(), ToTensor()]))
    img_mask_name = [('/groups/flyem/data/dingx/mask/data/trvol-250-1.h5','/groups/flyem/data/dingx/mask/data/trvol-250-1-mask.h5'),\
        ('/groups/flyem/data/dingx/mask/data/tstvol-520-1.h5','/groups/flyem/data/dingx/mask/data/tstvol-520-1-mask.h5')]
    Data = GenerateData_Multi(img_mask_name, crop_sz=(108,108,108), transform=transforms.Compose([FlipSample(), RotSample(), ToTensor()]))

    print(len(Data))
    # print(len(Data.pos_idx))
    # print(len(Data.neg_idx))

    data = Data.__getitem__(100)
    print(len(data))
    pos_img = data[0][0]
    pos_mask = data[0][1]
    neg_img = data[1][0]
    neg_mask = data[1][1]
    print(pos_img.shape)
    print(neg_mask.shape)

    save_path = '/groups/flyem/data/dingx/mask/data/'
    view_data(save_path+'sample_pos_img.h5', pos_img)
    view_data(save_path+'sample_pos_mask.h5', pos_mask)
    view_data(save_path+'sample_neg_img.h5', neg_img)
    view_data(save_path+'sample_neg_mask.h5', neg_mask)