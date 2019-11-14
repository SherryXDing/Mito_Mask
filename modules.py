import torch
import torch.nn.functional as F 
import numpy as np 
import os


class NeuralNetwork():
    """
    Network class that wraps model related functions (e.g., training, evaluation, etc)
    """
    def __init__(self, model, criterion, optimizer, device):
        """
        Args:
            model: a deep neural network model (sent to device already)
            criterion: loss function
            optimizer: training optimizer
            device: training device
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device


    def masked_loss(self, out, target):
        """
        calculate masked loss
        """
        if out.shape != target.shape:
            shrink_sz = ((target.shape[2]-out.shape[2])//2, (target.shape[3]-out.shape[3])//2, (target.shape[4]-out.shape[4])//2)
            target = target[:, :, shrink_sz[0]:-shrink_sz[0], shrink_sz[1]:-shrink_sz[1], shrink_sz[2]:-shrink_sz[2]]
        mask = target!=2
        target = mask * target  # target should be numbers between 0 and 1
        loss = self.criterion(out, target)
        loss = loss * mask  # masked loss
        loss = loss.sum() / len(mask[mask])
        return loss


    def train_model(self, data):
        """
        Train the model
        Args:
            data: training dataset generated by DataLoader
        Return batch-wise training loss
        """
        self.model.train()
        training_loss = 0

        for batch, sample in enumerate(data):
            for i in range(len(sample)):
                img = sample[i][0]
                mask = sample[i][1]
                img = img.to(self.device)
                mask = mask.to(self.device)
                # Forward
                out = self.model(img)
                # Calculate loss
                loss = self.masked_loss(out, mask)
                training_loss += loss.item()
                # Zero the parameter gradients
                self.optimizer.zero_grad()                    
                # Backward
                loss.backward()
                # Update weights
                self.optimizer.step()

        batch_loss = training_loss/((batch+1)*2)
        return batch_loss


    def eval_model(self, data):
        """
        Evaluate the model
        Args:
            data: evaluation dataset generated by DataLoader
        Return batch-wise evaluation loss
        """
        self.model.eval()
        eval_loss = 0

        for batch, sample in enumerate(data):
            for i in range(len(sample)):
                with torch.no_grad():  # Disable gradient computation
                    img = sample[i][0]
                    mask = sample[i][1]
                    img = img.to(self.device)
                    mask = mask.to(self.device)
                    out = self.model(img)
                    loss = self.masked_loss(out, mask)
                    eval_loss += loss.item()

        batch_loss = eval_loss/((batch+1)*2)
        return batch_loss


    def save_model(self, path, epoch, entire=False):
        """
        Save the model to disk
        Args:
            path: directory to save the model
            epoch: epoch that model is saved
            entire: if save the entire model rather than just save the state_dict
        """
        if not os.path.exists(path):
            os.mkdir(path)
        if entire:
            torch.save(self.model, path+"/whole_model_epoch_{}.pt".format(epoch))
        else:
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'criterion': self.criterion},
                        path+"/model_ckpt_{}.pt".format(epoch))
    

    def test_model(self, checkpoint, img, input_sz, step):
        """
        Test the model on new data
        Args:
            checkpoint: saved checkpoint
            img: testing data in [x, y, z] (network input is [batch, channel, x, y, z])
            input_sz: network input size in (x,y,z) 
            step: moving step in size (x,y,z)
        """

        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        gap = ((input_sz[0]-step[0])//2, (input_sz[1]-step[1])//2, (input_sz[2]-step[2])//2)

        out = np.zeros(img.shape, dtype=img.dtype)
        for row in range(0, img.shape[0]-input_sz[0], step[0]):
            for col in range(0, img.shape[1]-input_sz[1], step[1]):
                for vol in range(0, img.shape[2]-input_sz[2], step[2]):
                    # Generate 
                    patch_img = np.zeros((1, 1, input_sz[0], input_sz[1], input_sz[2]), dtype=img.dtype)
                    patch_img[0,0,:,:,:] = img[row:row+input_sz[0], col:col+input_sz[1], vol:vol+input_sz[2]]
                    patch_img = torch.from_numpy(patch_img).float()
                    patch_img = patch_img.to(self.device)
                    # Apply model
                    patch_out = self.model(patch_img)
                    patch_out = patch_out.cpu()
                    patch_out = patch_out.detach().numpy()
                    out[row+gap[0]:row+input_sz[0]-gap[0], col+gap[1]:col+input_sz[1]-gap[1], vol+gap[2]:vol+input_sz[2]-gap[2]] = patch_phi[0,0,:,:,:]
                    
        return out