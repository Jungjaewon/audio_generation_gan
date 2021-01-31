import os
import time
import datetime
import torch
import glob
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib
import soundfile as sf
import numpy as np

from model import Generator
from model import Discriminator
from sklearn.preprocessing import minmax_scale

matplotlib.use('Agg')


class Solver(object):

    def __init__(self, config, train_data_loader):
        """Initialize configurations."""
        self.train_data_loader = train_data_loader

        self.audio_length = config['MODEL_CONFIG']['AUDIO_LENGTH']
        self.sample_dim   = config['MODEL_CONFIG']['SAMPLE_DIM']
        self.sample_ch    = int(self.audio_length // self.sample_dim)

        self.epoch         = config['TRAINING_CONFIG']['EPOCH']
        self.batch_size    = config['TRAINING_CONFIG']['BATCH_SIZE']
        self.g_lr            = float(config['TRAINING_CONFIG']['G_LR'])
        self.d_lr            = float(config['TRAINING_CONFIG']['D_LR'])
        self.lambda_g_fake = config['TRAINING_CONFIG']['LAMBDA_G_FAKE']
        self.lambda_d_fake = config['TRAINING_CONFIG']['LAMBDA_D_FAKE']
        self.lambda_d_real = config['TRAINING_CONFIG']['LAMBDA_D_REAL']
        self.lambda_d_gp     = config['TRAINING_CONFIG']['LAMBDA_GP']
        self.d_critic      = config['TRAINING_CONFIG']['D_CRITIC']
        self.g_critic      = config['TRAINING_CONFIG']['G_CRITIC']

        self.g_fake_loss_tracker = list()
        self.d_real_loss_tracker = list()
        self.d_fake_loss_tracker = list()
        self.d_gp_loss_tracker = list()

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']

        self.cpu_seed = config['TRAINING_CONFIG']['CPU_SEED']
        self.gpu_seed = config['TRAINING_CONFIG']['GPU_SEED']
        #torch.manual_seed(config['TRAINING_CONFIG']['CPU_SEED'])
        #torch.cuda.manual_seed_all(config['TRAINING_CONFIG']['GPU_SEED'])

        self.label_list = ["air_conditioner", "car_horn", "children_playing",
                           "dog_bark", "drilling", "engine_idling", "gun_shot",
                           "jackhammer", "siren", "street_music"]
        self.data_dir = config['TRAINING_CONFIG']['DATA_DIR']

        #self.criterion = nn.CrossEntropyLoss()

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.use_tensorboard = config['TRAINING_CONFIG']['USE_TENSORBOARD'] == 'True'

        # Directory
        self.train_dir  = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir    = os.path.join(self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir  = os.path.join(self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])

        # Steps
        self.log_step       = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step    = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.save_step      = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start     = config['TRAINING_CONFIG']['SAVE_START']
        self.lr_decay_step  = config['TRAINING_CONFIG']['LR_DECAY_STEP']
        self.lr_update_step  = config['TRAINING_CONFIG']['LR_UPDATE_STEP']

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        k_dict = {'audio_length' : self.audio_length, 'sample_dim' : self.sample_dim, 'sample_ch' : self.sample_ch}
        self.G = Generator(**k_dict).to(self.gpu)
        self.D = Discriminator(**k_dict).to(self.gpu)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, (self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, (self.beta1, self.beta2))

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(os.path.join(self.train_dir,'model_arch.txt'), 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr

        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def restore_model(self):

        ckpt_list = glob.glob(osp.join(self.model_dir, '*.ckpt'))

        if len(ckpt_list) == 0:
            return 0

        ckpt_list = [int(x.split(os.sep)[-1].split('.')[0]) for x in ckpt_list]
        ckpt_list.sort()
        epoch = ckpt_list[-1]
        G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(epoch))
        D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(epoch))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.G.to(self.gpu)
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.D.to(self.gpu)

        return epoch

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.gpu)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def save_g_figures(self):
        x_axis = list(range(1, len(self.g_fake_loss_tracker) + 1))
        plt.figure()
        plt.title("g_loss")
        plt.xlabel('Epoch')
        plt.ylabel('')
        plt.plot(x_axis, self.g_fake_loss_tracker, color='r', label=f"g_fake loss")
        plt.legend()
        plt.legend(loc="lower right")
        plt.savefig(osp.join(self.sample_dir, f'g_fake_loss_graph.png'), dpi=150)
        plt.close()

    def save_d_figures(self):
        x_axis = list(range(1, len(self.d_fake_loss_tracker) + 1))
        plt.figure()
        plt.title("d_loss")
        plt.xlabel('Epoch')
        plt.ylabel('')
        plt.plot(x_axis, self.d_fake_loss_tracker, color='r', label=f"g_fake loss")
        plt.plot(x_axis, self.d_real_loss_tracker, color='r', label=f"g_fake loss")
        plt.plot(x_axis, self.d_gp_loss_tracker, color='r', label=f"g_fake loss")
        plt.legend()
        plt.legend(loc="lower right")
        plt.savefig(osp.join(self.sample_dir, f'd_loss_graph.png'), dpi=150)
        plt.close()

    def averaged_loss(self, loss_list):
        return sum(loss_list) / float(len(loss_list))

    def train(self):
        # Set data loader.
        train_data_loader = self.train_data_loader
        iterations = len(self.train_data_loader)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        train_data_iter = iter(train_data_loader)
        fixed_audios = next(train_data_iter)

        fixed_audios_list = list(torch.split(fixed_audios, self.batch_size, dim=0)[0])
        for idx, fixed_audio in enumerate(fixed_audios_list):
            fixed_audio = fixed_audio.squeeze().cpu().numpy().astype(np.float)
            #fixed_audio = np.clip(fixed_audio, -0.999999, 0.999999)
            path = osp.join(self.sample_dir, f'fixed_batch_{idx}.wav')
            sf.write(path, fixed_audio, 16384)

        fixed_noise = torch.randn(self.batch_size, self.sample_ch, self.sample_dim).to(self.gpu)

        start_epoch = self.restore_model()
        start_time = time.time()

        # =================================================================================== #
        #                             2. Train the model                                      #
        # =================================================================================== #

        print('Start training...')
        for e in range(start_epoch, self.epoch):

            g_fake_loss_list = list()
            d_real_loss_list = list()
            d_fake_loss_list = list()
            d_gp_loss_list = list()
            for i in range(iterations):
                try:
                    real_audios = next(train_data_iter)
                except:
                    train_data_iter = iter(train_data_loader)
                    real_audios = next(train_data_iter)

                real_audios = real_audios.to(self.gpu)
                noise = torch.randn(self.batch_size, self.sample_ch, self.sample_dim).to(self.gpu)
                loss_dict = dict()

                if (i + 1) % self.d_critic == 0:
                    fake_audios = self.G(noise)
                    real_score = self.D(real_audios)
                    fake_score = self.D(fake_audios.detach())
                    d_loss_real = -torch.mean(real_score)
                    d_loss_fake = torch.mean(fake_score)
                    alpha = torch.rand(real_audios.size(0), 1, 1).to(self.gpu)
                    x_hat = (alpha * real_audios.data + (1 - alpha) * fake_audios.data).requires_grad_(True)
                    out_src = self.D(x_hat)
                    d_loss_gp = self.gradient_penalty(out_src, x_hat)
                    d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake + self.lambda_d_gp * d_loss_gp

                    if torch.isnan(d_loss):
                        raise Exception('d_loss is nan at {}'.format(e * iterations + (i + 1)))

                    # Backward and optimize.
                    self.reset_grad()
                    d_loss.backward(retain_graph=True)
                    self.d_optimizer.step()

                    # Logging.
                    loss_dict['D/loss_real'] = self.lambda_d_real * d_loss_real.item()
                    loss_dict['D/loss_fake'] = self.lambda_d_fake * d_loss_fake.item()
                    loss_dict['D/loss_pg'] = self.lambda_d_gp * d_loss_gp.item()

                    d_real_loss_list.append(self.lambda_d_real * d_loss_real.item())
                    d_fake_loss_list.append(self.lambda_d_fake * d_loss_fake.item())
                    d_gp_loss_list.append(self.lambda_d_gp * d_loss_gp.item())

                if (i + 1) % self.g_critic == 0:
                    fake_audios = self.G(noise)
                    fake_score = self.D(fake_audios)
                    g_loss_fake = -torch.mean(fake_score)
                    g_loss = self.lambda_g_fake * g_loss_fake

                    if torch.isnan(g_loss):
                        raise Exception('g_loss is nan at {}'.format(e * iterations + (i + 1)))

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss_dict['G/loss_fake'] = self.lambda_g_fake * g_loss_fake.item()
                    g_fake_loss_list.append(self.lambda_g_fake * g_loss_fake.item())

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e+1, self.epoch, et, i + 1, iterations)
                    for tag, value in loss_dict.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            self.g_fake_loss_tracker.append(self.averaged_loss(g_fake_loss_list))
            self.d_real_loss_tracker.append(self.averaged_loss(d_real_loss_list))
            self.d_fake_loss_tracker.append(self.averaged_loss(d_fake_loss_list))
            self.d_gp_loss_tracker.append(self.averaged_loss(d_gp_loss_list))

            if (e + 1) % self.sample_step == 0:
                with torch.no_grad():
                    fake_audios = self.G(fixed_noise)
                    fake_audios_list = list(torch.split(fake_audios, self.batch_size, dim=0)[0])
                    for idx, fake_audio in enumerate(fake_audios_list):
                        fake_audio = fake_audio.squeeze().cpu().numpy()
                        path = osp.join(self.sample_dir, f'epoch_{e + 1}_batch_{idx}.wav')
                        sf.write(path, fake_audio, 16384)

            # Save model checkpoints.
            if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:
                G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(e + 1))
                D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(e + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))

            if (e + 1) % self.lr_update_step == 0 and self.lr_decay_step > 0:
                pass

        self.save_g_figures()
        self.save_d_figures()
        print('Training is finished')

    def test(self):
        pass

