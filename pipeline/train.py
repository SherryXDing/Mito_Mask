import torch
from torch.utils.data import DataLoader, Subset
from data import *
from model import UNet
from modules import NeuralNetwork
import torch.nn as nn
from torchvision import transforms
import os
import json
import matplotlib.pyplot as plt
import time 


data_path = '/groups/flyem/data/dingx/mask/data/'
img_mask_name = [(data_path+'trvol-250-1.h5', data_path+'trvol-250-1-mask.h5'), \
        (data_path+'tstvol-520-1.h5', data_path+'tstvol-520-1-mask.h5')]

save_path = '/groups/flyem/data/dingx/mask/model_multiclass/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
depth = 3
in_channels = 1
out_channels = 2
base_filters = 16
model = UNet(in_channels=in_channels, base_filters=base_filters, out_channels=out_channels, depth=depth)
if torch.cuda.device_count()>1:
    print('---Using {} GPUs---'.format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
model.to(device)

# criterion
unmask_label=2
if unmask_label is not None:
    criterion = nn.CrossEntropyLoss(ignore_index=unmask_label, reduction='mean')
else:
    criterion = nn.CrossEntropyLoss(reduction='mean')
# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=0.00005, nesterov=True)
network = NeuralNetwork(model, criterion, optimizer, device)
# parameters
crop_sz=(64,64,64)
num_data = 1000
transform = transforms.Compose([FlipSample(), RotSample(), ToTensor()])
batch_sz = 64
total_epoch = 2000
train_loss_total = []
eval_loss_total = []
save_iter = 100
start_time = time.time()

for epoch in range(total_epoch):
    print('......Epoch {}......'.format(epoch))
    since = time.time()
    # generate data for each epoch
    data = GenerateData_Multiclass(img_mask_name, unmask_label=unmask_label, crop_sz=crop_sz, num_data=num_data, transform=transform)
    indices = range(len(data))
    train_data = Subset(data, indices[:-100])
    eval_data = Subset(data, indices[-100:])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_sz, shuffle=True, num_workers=4)
    eval_loader = DataLoader(dataset=eval_data, batch_size=batch_sz, shuffle=False, num_workers=4)

    train_loss = network.train_model(train_loader)
    print('Training loss is {}'.format(train_loss))
    train_loss_total.append(train_loss)
    eval_loss = network.eval_model(eval_loader)
    print('Evaluation loss is {}'.format(eval_loss))
    eval_loss_total.append(eval_loss)

    time_elapsed = time.time() - since
    print('Training for epoch {} takes {:.0f}min {:.0f}sec'.format(epoch, time_elapsed//60, time_elapsed%60))

    if epoch%save_iter == save_iter-1:
        network.save_model(save_path, epoch+1)
        save_dict = {'train_loss_total': train_loss_total, 'eval_loss_total': eval_loss_total}
        with open(save_path+'/loss.json', 'w') as f:
            json.dump(save_dict, f)

total_time = time.time() - start_time
print('Total time elapse: {:.0f}hour {:.0f}min'.format(total_time//3600, total_time%3600//60))

# plot loss
# plt.figure()
# plt.plot(range(len(train_loss_total)), train_loss_total, 'k', label='Train')
# plt.plot(range(len(eval_loss_total)), eval_loss_total, 'b', label='Test')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(loc='upper right')
# plt.savefig(save_path+'/loss.pdf')