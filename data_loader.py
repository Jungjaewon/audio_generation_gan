import os.path as osp
import glob
import numpy as np
import torch

import librosa

from torch.utils import data
from torchvision import transforms as T


class DataSet(data.Dataset):

    def __init__(self, config, mode, transform):

        self.data_dir = config['TRAINING_CONFIG']['DATA_DIR']
        self.audio_length = config['MODEL_CONFIG']['AUDIO_LENGTH']
        self.data_list = glob.glob(osp.join(self.data_dir, '*.wav'))
        self.mode = mode
        self.transform = transform


    def __getitem__(self, index):
        data_path = self.data_list[index]
        data, sr = librosa.core.load(data_path, sr=16000, mono=False)
        if len(data) < self.audio_length:
            padding_size = self.audio_length - len(data)
            data = np.pad(data, (0, padding_size), mode='edge')
        data = data.reshape(1, data.shape[0])
        #print(np.max(np.abs(data)))
        #print(np.min(np.abs(data)))
        data /= np.max(np.abs(data))
        #print('Normalize')
        #print(np.max(np.abs(data)))
        #print(np.min(np.abs(data)))
        return torch.from_numpy(data)

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)


def get_loader(config, mode):

    transform = list()
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5), std=(0.5)))
    transform = T.Compose(transform)

    dataset = DataSet(config, mode, transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'] if mode == 'train' else 1,
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader
