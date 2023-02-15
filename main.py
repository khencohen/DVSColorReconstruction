from LONGDIC import Xception as x
import torch.nn as nn
from LONGDIC import MS_SSIM as SSIN
import torch
import sys
import matplotlib.pyplot as plt
from LONGDIC import data_augmentation
import numpy as np


def gpu_tensor_to_numpy(gpu_tensor):
    '''
    converts a torch tensor that is part of a batch that is stored on the gpu to a np array (stored on cpu obviously)
    :param gpu_tensor: torch tensor stored on gpu that is part of a batch
    :return: np array of RGB frame
    '''
    temp = gpu_tensor.cpu()
    temp = temp.detach().numpy()[0]
    temp = np.swapaxes(temp, 0, 2)
    return temp

def load_i_data(H, W, i, N, cuda_dev="cuda:0"):
    '''
    Loads data using the data_augmentation class and converts it to proper torch form to be used in the NN
    :param W: frame width
    :param H: frame height
    :param i: file index
    :param N: batch size
    :param cuda_dev: cuda device to load tensors on (i.e. "cuda:0")
    :return: tupe of input and corresponding output to the network (in that order)
    '''
    assert i <= data_augmentation.batch_size
    samples_batch, labels_batch = data_augmentation.get_batch(data_augmentation.loc_samples,
                                                              data_augmentation.loc_labels,
                                                              i, H, W, N)
    labels_batch = np.swapaxes(labels_batch, 1, 3)
    labels_batch = np.swapaxes(labels_batch, 2, 3)

    # convert type, load to gpu, normalize.
    frames_in = torch.tensor(samples_batch, dtype=torch.float)
    frames_in = frames_in.to(cuda_dev)
    target = torch.tensor(labels_batch, dtype=torch.float) / 255.0
    target = target.to(cuda_dev)
    return frames_in, target

print('__Python VERSION:', sys.version)
print('_pyTorch VERSION:', torch.version)
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

# configure gpu
if torch.cuda.is_available():
    print("cuda available")
    dev = "cuda:0"
else:
    print("cuda available")
    dev = "cpu"



# vars
device = torch.device(dev)
path = "C:/Users/Public/Desktop/LONGDIC/mem.pth"
learning_rate = 0.01
SSIM_factor, L1_factor = 0.1, 0.9
N = 1
loss_list = []
epoch = 1
training_data_len = 4 # how many training samples do we have in our files

# setup NN, loss, optimizer
SSIM_criterion = SSIN.SSIM()
L1_criterion  = nn.L1Loss()
XNet = x.xception()
XNet.to(device)
XNet = nn.DataParallel(XNet)

# ***check this line agian****
XNet.load_state_dict(torch.load(path))


optimizer = torch.optim.Adam(XNet.parameters(), lr=learning_rate)


# main training loop
for k in range(epoch):
    for i in range(training_data_len):
        # load data
        frames_in, target = load_i_data(128, 96, i+1, N, device)

        # compute output and loss
        out = XNet.forward(frames_in)
        SSIM_gain = SSIM_criterion(out, target)  # 1 is good, 0 is bad (we want to maximize this)
        loss = SSIM_factor * (1 - SSIM_gain) + L1_factor * L1_criterion(out, target)  # the lower the better
        loss_val = loss.item()  # 0 is good, 1 is bad (we want to minimize this)
        loss_list.append(loss_val)

        # calc grad and back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # just to test, show the label, the NN output and the loss
        print("epoch", (k+1), "data seg", (i+1) ,"loss", loss_val)
        if k%100 == 0:
            out_pic = gpu_tensor_to_numpy(out)
            label_pic = gpu_tensor_to_numpy(target)
            plt.imshow(out_pic)
            plt.show()
            plt.figure()
            plt.imshow(label_pic)
            plt.show()

        # free memory
        torch.cuda.empty_cache()

# plot the loss graph
plt.plot(range(len(loss_list)), loss_list)
plt.show()

# save weights
torch.save(XNet.state_dict(), path)


