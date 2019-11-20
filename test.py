import h5py
import numpy as np
from model import UNet 
from modules import NeuralNetwork
import torch
import torch.nn as nn 
import torch.optim as optim
import os 
import matplotlib.pyplot as plt 
import json


# model_path = '/groups/flyem/data/dingx/mask/model_depth3_insz64/'
# with open(model_path+'loss.json', 'r') as f:
#     loss = json.load(f)
# train_loss_total = loss['train_loss_total']
# eval_loss_total = loss['eval_loss_total']
# plt.figure()
# plt.plot(range(len(train_loss_total)), train_loss_total, 'k', label='Train')
# plt.plot(range(len(eval_loss_total)), eval_loss_total, 'b', label='Test')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(loc='upper right')
# plt.savefig(model_path+'/loss.pdf')
# plt.show()


# test data
test_data = '/groups/flyem/data/dingx/mask/data/tstvol-520-1.h5'
test_img = np.float32(h5py.File(test_data,'r')['raw'][()])
test_img = (test_img - test_img.mean()) / test_img.std()

# checkpoint
model_path = '/groups/flyem/data/dingx/mask/model_depth3_insz64/'  # '/groups/flyem/data/dingx/mask/model_depth4_insz108/'
ckpt_list = ['model_ckpt_2000.pt', 'model_ckpt_4000.pt', 'model_ckpt_6000.pt', 'model_ckpt_8000.pt', 'model_ckpt_10000.pt']

# Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
depth = 3  # 4
model = UNet(in_channels=1, base_filters=16, out_channels=1, depth=depth)
if torch.cuda.device_count()>1:
    print('---Using {} GPUs---'.format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
model.to(device)
# criterion
criterion = nn.BCELoss(reduction='none')
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=0.00005, nesterov=True)

network = NeuralNetwork(model, criterion, optimizer, device)
input_sz = (108, 108, 108) 
step = (68, 68, 68)  # (20, 20, 20)

# Load checkpoints
for ckpt in ckpt_list:
    print("Test checkpoint {}".format(ckpt))
    result = network.test_model(model_path+ckpt, test_img, input_sz, step)
    result[result>=0.5]=255
    result[result<0.5]=0
    result = np.uint8(result)
    save_name = 'result_{}.h5'.format(os.path.splitext(ckpt)[0])
    with h5py.File(model_path+save_name, 'w') as f:
        dset = f.create_dataset('result', test_img.shape, dtype=result.dtype)
        dset[:,:,:] = result