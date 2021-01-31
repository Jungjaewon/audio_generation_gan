import torch
import librosa
import numpy as np
import soundfile as sf

if __name__ == '__main__':
    pass

    """
    temp_ch = 16
    layer_cnt = 0
    while temp_ch != 1:
        layer_cnt += 1
        temp_ch = temp_ch // 2

    print(layer_cnt)
    """

    """
    L_in = 2048
    stride = 2
    padding = 5
    kernel = 12

    #L_out = (L_in - 1) * stride - 2 * padding + (kernel - 1) + 1
    L_out = ((L_in - 1) - 2 * padding + (kernel - 1) - 1) / stride + 1
    print('L_out : ', L_out)

    tensor = torch.randn(2,1, 1024)
    print(tensor)
    print(len(torch.split(tensor, 2)))

    for t in list(torch.split(tensor, 2, dim=0)[0]):
        print(t)
    """

    audio = np.random.uniform(-1, 1, 16000)
    print(np.shape(audio))
    print(type(audio))
    print(audio.dtype)
    sf.write('noise.wav', audio, 16000)
