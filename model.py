import torch.nn as nn
import torch
import torch.nn.functional as F


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling (to be used by Discriminator Only) by:
        -Shifting feature axis of a 3d tensor by a random integer in [-n n]
        -Performing reflection padding where necessary
    """

    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):

        if self.shift_factor == 0:
            return x

        # k_list :  list of batch_size shift factors, random generated uniformly between [-shift_factor, shift_factor]
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        # k_map : dict containing {each shift factor : list of batch indices with that shift factor}
        # e.g. if shift_factor = 1& batch_size = 64, k_map = {-1:[0,2,30,..,52], 0:[1,5,...,60],1:[2,3,4,...,63]}

        k_map = {}

        for sample_idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []

            k_map[k].append(sample_idx)

        shuffled_x = x.clone()

        for k, sample_idxs in k_map.items():

            if k > 0: # Remove the last k values & insert k left-paddings
                shuffled_x[sample_idxs] = F.pad(x[sample_idxs][..., :-k], pad=[k,0], mode='reflect')
            else: # 1. Remove the first k values & 2. Insert k right-paddings
                shuffled_x[sample_idxs] = F.pad(x[sample_idxs][..., abs(k):], pad=[0,abs(k)], mode='reflect')

        assert shuffled_x.shape == x.shape, f"{shuffled_x}, {x.shape}"

        return shuffled_x

class Generator(nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()
        pass
        self.main = list()
        self.audio_length = kwargs['audio_length']
        self.sample_dim = kwargs['sample_dim']
        self.sample_ch = kwargs['sample_ch']
        temp_ch = kwargs['sample_ch']
        # audio_length : 16384
        # sample dim : 1024
        # sample ch : 16
        #(B , 16, 1024)
        #(B , 8, 2048)
        #(B , 4, 4096)
        #(B , 2, 8192)
        #(B , 1, 16384)

        layer_cnt = 0
        while temp_ch != 1:
            layer_cnt += 1
            temp_ch = temp_ch // 2

        for i in range(layer_cnt):
            self.main.append(nn.ConvTranspose1d(self.sample_ch, self.sample_ch // 2, kernel_size=12, stride=2, padding=5))
            self.main.append(nn.BatchNorm1d(self.sample_ch // 2))
            self.main.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            self.sample_ch = self.sample_ch / 2
            self.sample_ch = int(self.sample_ch)

        self.main = nn.Sequential(*self.main)
        self.tanh = nn.Tanh()

    def forward(self, noise):
        return self.tanh(self.main(noise))


class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        self.phase_shuffle = PhaseShuffle(2)
        self.audio_length = kwargs['audio_length']
        self.sample_dim = kwargs['sample_dim']
        self.sample_ch = kwargs['sample_ch']
        self.main = list()
        self.main = nn.Sequential(*self.main)
        #(B , 1, 16384)
        #(B , 2, 8192)
        #(B , 4, 4096)
        #(B , 8, 2048)
        #(B , 16, 1024)
        self.conv1 = nn.Conv1d(1, 2, kernel_size=12, stride=2, padding=5)
        self.conv2 = nn.Conv1d(2, 4, kernel_size=12, stride=2, padding=5)
        self.conv3 = nn.Conv1d(4, 8, kernel_size=12, stride=2, padding=5)
        self.conv4 = nn.Conv1d(8, 16, kernel_size=12, stride=2, padding=5)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.phase_shuffle(x)
        x = self.leaky_relu(self.conv2(x))
        x = self.phase_shuffle(x)
        x = self.leaky_relu(self.conv3(x))
        x = self.phase_shuffle(x)
        x = self.leaky_relu(self.conv4(x))
        x = self.phase_shuffle(x)
        return x