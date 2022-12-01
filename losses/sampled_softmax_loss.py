# 作者 ：duty
# 创建时间 ：2022/8/4 12:15 下午
# 文件 ：sampled_softmax_loss.py
import numpy as np
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss


# https://zhuanlan.zhihu.com/p/528862933 参考； 相当于在模型输出后又加了一个全连接，该全连接用于sampled softmax的采样计算，同是也在模型训练时共同训练权重
# 如何使用 参考https://github.com/dutyhong/Sampled-Softmax-PyTorch/blob/9545b61f425e3ad3790deb6a3d7b9d9d0a601fa5/main.py#train和evaluate
class LogUniformSampler(object):
    def __init__(self, ntokens):

        self.N = ntokens
        self.prob = [0] * self.N

        self.generate_distribution()

    def generate_distribution(self):
        for i in range(self.N):
            self.prob[i] = (np.log(i+2) - np.log(i+1)) / np.log(self.N + 1)

    def probability(self, idx):
        return self.prob[idx]

    def expected_count(self, num_tries, samples):
        freq = list()
        for sample_idx in samples:
            freq.append(-(np.exp(num_tries * np.log(1-self.prob[sample_idx]))-1))
        return freq

    def accidental_match(self, labels, samples):
        sample_dict = dict()

        for idx in range(len(samples)):
            sample_dict[samples[idx]] = idx

        result = list()
        for idx in range(len(labels)):
            if labels[idx] in sample_dict:
                result.append((idx, sample_dict[labels[idx]]))

        return result

    def sample(self, size, labels):
        log_N = np.log(self.N)

        x = np.random.uniform(low=0.0, high=1.0, size=size)
        value = np.floor(np.exp(x * log_N)).astype(int) - 1
        samples = value.tolist()

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, samples)

        return samples, true_freq, sample_freq

    def sample_unique(self, size, labels):
        # Slow. Not Recommended.
        log_N = np.log(self.N)
        samples = list()

        while (len(samples) < size):
            x = np.random.uniform(low=0.0, high=1.0, size=1)[0]
            value = np.floor(np.exp(x * log_N)).astype(int) - 1
            if value in samples:
                continue
            else:
                samples.append(value)

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, samples)

        return samples, true_freq, sample_freq

class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nsampled, nhid, tied_weight):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled
        # log uniform 采样
        self.sampler = LogUniformSampler(self.ntokens)
        # 每个item的向量表示网络， 网络训练完成需要保存该网络权重，后续预测， 可直接作为item的向量化表示
        self.params = nn.Linear(nhid, ntokens)
        # 初始化权重
        if tied_weight is not None:
            self.params.weight = tied_weight
        else:
            in_, out_ = self.params.weight.size()
            stdv = math.sqrt(3. / (in_ + out_))
            self.params.weight.data.uniform_(-stdv, stdv)

    # 输入inputs为网络最后输出的logits， labels为batch的标签
    def forward(self, inputs, labels):
        if self.training:
            # sample ids according to word distribution - Unique
            sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
            return self.sampled(inputs, labels, sample_values, remove_accidental_match=True)
        else:
            return self.full(inputs)

    def sampled(self, inputs, labels, sample_values, remove_accidental_match=False):

        batch_size, d = inputs.size()
        sample_ids, true_freq, sample_freq = sample_values

        sample_ids = Variable(torch.LongTensor(sample_ids))
        true_freq = Variable(torch.FloatTensor(true_freq))
        sample_freq = Variable(torch.FloatTensor(sample_freq))

        # gather true labels - weights and frequencies
        true_weights = self.params.weight[labels, :]
        true_bias = self.params.bias[labels]

        # gather sample ids - weights and frequencies
        sample_weights = self.params.weight[sample_ids, :]
        sample_bias = self.params.bias[sample_ids]

        # calculate logits
        true_logits = torch.sum(torch.mul(inputs, true_weights), dim=1) + true_bias
        sample_logits = torch.matmul(inputs, torch.t(sample_weights)) + sample_bias
        # remove true labels from sample set
        if remove_accidental_match:
            acc_hits = self.sampler.accidental_match(labels.data.cpu().numpy(), sample_ids.data.cpu().numpy())
            acc_hits = list(zip(*acc_hits))
            sample_logits[acc_hits] = -1e37

        # perform correction
        true_logits = true_logits.sub(torch.log(true_freq))
        sample_logits = sample_logits.sub(torch.log(sample_freq))

        # return logits and new_labels
        logits = torch.cat((torch.unsqueeze(true_logits, dim=1), sample_logits), dim=1)
        new_targets = Variable(torch.zeros(batch_size).long())
        return logits, new_targets

    def full(self, inputs):
        return self.params(inputs)

if __name__=="__main__":
    criterion = nn.CrossEntropyLoss()
    sampled_softmax = SampledSoftmax(ntokens=1000, nsampled=100, nhid=128,tied_weight=None)
    targets = torch.tensor([1,2,2])
    inputs = torch.rand(3,128)
    logits, new_targets = sampled_softmax(inputs, targets)
    print(logits.shape)
    print(new_targets.shape)
    loss_func = CrossEntropyLoss()
    loss = loss_func(logits, new_targets)
    print(loss)
