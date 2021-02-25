# Learnable Frontend (lft) experiment adaptation script
# This script is a collective implementation of learnable frontend
#   techniques, excluding DCT part.
# As most speech researchers are aware of, there are 4 opertors that contain linear
#   transformations that are involved in frontend processing. We
#   attempted to make them learnable one at a time, providing more
#   space for rapid and more explicit DNN adaptation.
#
# 1/ Windowing
# 2/ DFT + |.|^2 i.e. magnitude spectrum
# 3/ Mel filter bank transformation
# 4/ DCT (note: for RESNET experiments we don't need this)
import numpy as np
import torch
import torch.nn as nn

from torchaudio.transforms import SlidingWindowCmn

from python_speech_features import base
from scipy.linalg import dft
from torch_dct import dct

def create_dct_matrix(dim):
    C = np.zeros((dim, dim))
    for k in range(dim):
        for n in range(dim):
            if k == 0:
                C[k,n] = np.sqrt(1/dim)
            else:
                C[k,n] = np.sqrt(2/dim) * np.cos((np.pi*k*(1/2+n))/dim)
    return C


def melbank_loss_reg(loss, model, reg=0.1):
    w = model.ft_model.melbanks
    l1_norm = reg * torch.norm(w**2)
    return loss + l1_norm


def stft_loss_reg(loss, model, reg=0.1):
    def _symmetric_distance(weight):
        normed_w = weight / weight.max()
        symmetric_w = normed_w + normed_w.T
        symmetric_w = symmetric_w / symmetric_w.max()
        return torch.norm(normed_w-symmetric_w) / torch.norm(normed_w)

    real_loss = _symmetric_distance(model.ft_model.real_fft)
    imag_loss = _symmetric_distance(model.ft_model.imag_fft)
    return loss + reg * (real_loss + imag_loss)


def window_loss_reg(loss, model, reg=0.1):
    def _distance_from_cos(weight):
        normed_w = weight / weight.max()
        frame_len = normed_w.shape[0]
        cosine = torch.Tensor(np.cos(2*np.pi*np.arange(0, frame_len)/frame_len)).cuda()
        return torch.norm(normed_w + cosine) / torch.norm(normed_w)
    
    window = model.ft_model.window
    add_loss = _distance_from_cos(window)
    return loss + reg * add_loss


def dct_loss_reg(loss, model, reg=0.1):
    '''soft orthogonality constraint regularizer cited from:
    https://www.isca-speech.org/archive/Odyssey_2020/abstracts/72.html
    '''        
    output_w = model.ft_model.dct.data
    reg_w = torch.matmul(output_w.T, output_w) - torch.ones(output_w.shape)
    ortho_norm = reg * (np.linalg.norm(reg_w) ** 2)
    return loss + ortho_norm


class pcmn(nn.Module):
    '''pcmn implementation as a replacement to cms:
    https://ieeexplore.ieee.org/document/8683674
    '''
    def __init__(self, in_dim, update_beta=False):
        self.in_dim = in_dim        
        # we update alpha here by default, but updating
        #   beta is optional (no by default, since it's
        #   basically scaling factor)
        self.alpha = nn.Parameter(torch.ones((in_dim,)))
        nn.init.xavier_normal_(self.alpha.data)
        self.beta = nn.Parameter(torch.ones(in_dim,))
        if not update_beta:
            self.beta.requires_grad = False
    
    def forward(self, input_x):
        num_utts, rows, cols = input_x.shape
        output_x = torch.zeros(input_x.shape)
        for i in range(input_x):
            vec = input_x[i,:,:]
            norm = torch.mean(vec, axis=0)
            norm_vec = norm.repeat(rows, 1)
            mean_subed = (self.beta.repeat(rows, 1) * vec - 
                          self.alpha.repeat(rows, 1) * norm_vec)
            output_x[i,:,:] = mean_subed
        return output_x


class AdaptiveCMS(nn.Module):
    '''simple full-utterance adaptive CMS,
    making the mean calculation as a weighted approach
    '''
    def __init__(self, feats_dim, chunk_len,
                 update=False, use_cuda=True):
        super(AdaptiveCMS, self).__init__()
        self.feats_dim = feats_dim
        self.use_cuda = use_cuda
        self.eps = 2**-30

        # weight vector for pseudo-mean calc
        self.norm_w = nn.Parameter(torch.ones(chunk_len))
        if not update:
            self.norm_w.requires_grad = False
    
    def forward(self, x):
        '''@x: batch_size X frame_idx X feat_idx
        '''
        assert x.shape[1] == chunk_len
        mean_x = torch.einsum('j,ijk->ijk', self.norm_w, x)
        sum_mean_x = torch.sum(mean_x, axis=2) / chunk_len
        return x.sub(sum_mean_x[:,:, None])



class FullLearnable(nn.Module):
    '''one-in-all with kernel initialization
    '''
    def __init__(self, in_dim, feats_dim, use_cuda=True,
                nfft=512, apply_log=True, 
                update_window=False, update_dft=False,
                update_melbank=False, update_dct=False,
                update_cms=False):
        super(FullLearnable, self).__init__()
        self.in_dim = in_dim
        self.feats_dim = feats_dim
        self.use_cuda = use_cuda
        self.nfft = nfft
        self.apply_log = apply_log

        self.update_window = update_window
        self.update_dft = update_dft
        self.update_melbank = update_melbank
        self.update_dct = update_dct
        self.update_cms = update_cms

        self.win_t = torch.Tensor(np.hamming(self.in_dim))
        self.window = nn.Parameter(self.win_t)
        if not self.update_window:
            self.window.requires_grad = False

        stft_kernel = dft(nfft, scale=None)
        self.real_fft = nn.Parameter(torch.Tensor(stft_kernel.real))
        self.imag_fft = nn.Parameter(torch.Tensor(stft_kernel.imag))
        if not self.update_dft:
            self.real_fft.requires_grad = False
            self.imag_fft.requires_grad = False

        self.melbanks = nn.Parameter(torch.Tensor(
                            base.get_filterbanks(self.feats_dim, self.nfft)))
        if not self.update_melbank:
            self.melbanks.requires_grad = False

        dct_matrix = torch.Tensor(create_dct_matrix(self.feats_dim))
        self.dct = nn.Linear(self.feats_dim, self.feats_dim, bias=False)
        self.dct.weight = nn.Parameter(dct_matrix)

        if self.update_cms:
            self.cmvn = SlidingWindowCmn(cmn_window=300, center=True)
        else:
            self.cmvn = pcmn(self.feats_dim)
    
    def self_log(self, x):
        x[x <= 0.0] = torch.finfo(torch.float32).eps
        x = torch.log(x)
        return x

    @staticmethod
    def loss_reg(loss, model, module='dct', reg=0.1,
                update_window=False, update_dft=False,
                update_melbank=False, update_dct=False):
        '''
        Loss regularization on models, depends on the module
        to be updated, one shall have different schemes
        '''
        if update_melbank:
            loss = melbank_loss_reg(loss, model, reg=0.1)
        if update_dft:
            loss = stft_loss_reg(loss, model, reg=0.1)
        if update_window:
            loss = window_loss_reg(loss, model, reg=0.1)
        if update_dct:
            loss = dct_loss_reg(loss, model, reg=0.1)
        return loss
    
    def melbank_weight_reg(self, eps=1e-4):
        '''update weights by forcing positivity
            in order to not affect the peak value,
            we force negative values to small eps
        '''
        with torch.no_grad():
            self.melbanks[self.melbanks < 0.0] = eps

    def stft_weight_reg(self, eps=1e-4):
        '''weight regularizaiton by
            forcing matrix to be symmetric
        '''
        def _symmetric_positive(weight):
            normed_w = weight / weight.max()
            symmetric_w = normed_w + normed_w.T
            symmetric_w = symmetric_w / symmetric_w.max()
            symmetric_w[symmetric_w < 0.0] = eps
            return symmetric_w
        
        self.real_fft.data = _symmetric_positive(self.real_fft)
        self.imag_fft.data = _symmetric_positive(self.imag_fft)  
    
    def window_weight_reg(self, eps=1e-4):
        '''create symmetric ver. of window
            and force it to be positive
        '''
        win_size = self.window.shape[0]
        if win_size % 2 == 0:
            # even number of samples
            cut_size = int(win_size / 2)
            w = self.window[:cut_size]
            symmetric_w = torch.flip(w, (0,))
            self.window.data = torch.abs(torch.cat((w, symmetric_w), 0))
    
    def dct_weight_reg(self, eps=1e-4):
        '''a more aggressive way to provide orthogonality
        if to simply use QR decomposition on weight square matrix
        and keep the Q part...
        '''
        q, r = torch.qr(self.dct.data)
        self.dct.data = q

    def weight_reg(self, eps=1e-4):
        if self.update_melbank: self.melbank_weight_reg(eps=eps)
        if self.update_dft: self.stft_weight_reg(eps=eps)            
        if self.update_window: self.window_weight_reg(eps=eps)
        if self.update_dct: self.dct_weight_reg(eps=eps)

    def learnt_dft(self, x):
        batch_size, num_frames, frame_len = x.shape
        spec_x = torch.zeros((batch_size, num_frames, int(self.nfft/2+1)))
        if self.use_cuda:
            spec_x = spec_x.cuda()
        for i in range(batch_size):
            tx = x[i,:,:]
            if frame_len < self.nfft:
                zeros = torch.zeros((num_frames, self.nfft-frame_len))
                if self.use_cuda:
                    zeros = zeros.cuda()
                tx = torch.cat([tx, zeros], dim=1)
            else:
                tx = tx[:,:self.nfft]
            real_x = torch.matmul(tx, self.real_fft)[:,:int(self.nfft/2+1)]
            imag_x = torch.matmul(tx, self.imag_fft)[:,:int(self.nfft/2+1)]
            if self.use_cuda:
                real_x, imag_x = real_x.cuda(), imag_x.cuda()
            spec_x[i,:,:] = real_x**2 + imag_x**2
            spec_x[i,:,:] = 1.0 / self.nfft * spec_x[i,:,:]
        return spec_x

    def learnt_dct(self, x):
        out_x = torch.zeros(x.shape)
        if self.use_cuda:
            out_x = out_x.cuda()
        batch_size, _, _ = x.shape
        for i in range(batch_size):
            out_x[i,:,:] = torch.matmul(x[i,:,:], self.log_dct)
        return out_x

    def forward(self, input_x):
        x = input_x
        # windowing
        x = torch.einsum('k,ijk->ijk', self.window, x)
        # dft
        spec_x = self.learnt_dft(x)
        # melbanks
        x = torch.matmul(spec_x, self.melbanks.T)
        if self.apply_log:
            x = self.self_log(x)
        # dct
        if self.update_dct:
            x = self.log_dct(x).transpose(2,1)
        else:
            x = dct(x, norm='ortho')
        x = self.cmvn(x)
        return x


class LFtXvector(nn.Module):
    '''An cascaded ensemble of learnable frontend
    and xvector
    '''
    def __init__(self, ft_model, xvector_model, mode='train'):
        super(LFtXvector, self).__init__()
        self.ft_model = ft_model
        self.xvector_model = xvector_model
        if mode == 'adapt':
            for name, param in self.xvector_model.named_parameters():
                if not 'fc' in name and not 'output' in name:
                    param.requires_grad = False

    def forward(self, input_x):
        x = input_x
        x = self.ft_model(x)
        x = self.xvector_model(x)
        return x
    
    def infer(self, input_x):
        x = input_x
        x = self.ft_model(x)
        x = self.xvector_model.infer(x)
        return x