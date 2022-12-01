# 作者 ：duty
# 创建时间 ：2022/7/18 3:24 下午
# 文件 ：nll_loss.py
from torch import nn
import torch
import torch.nn.functional as F

# 参考https://blog.csdn.net/qq_22210253/article/details/85229988实现
class Loss(nn.Module):
	def __init__(self):
		super().__init__()
		self.gamma = 2
		self.epsilon = 0.1
		self.num_class = 3

	def forward(self, inputs, targets, loss_type="NLL"):
		"""
		将每个label对应的样本输出的logits相加取个均值
		:param input: [B C]
		:param target: [B]
		:return:
		"""
		loss = 0
		if loss_type=="NLL":
			for i, label in enumerate(targets):
				input = inputs[i]
				tmp_loss = input[label]
				loss = loss + tmp_loss
			return -loss/self.num_class
		if loss_type=="CrossEntropy":
			sm_input = torch.softmax(inputs, dim=1)
			log_input = torch.log(sm_input)
			for i, label in enumerate(targets):
				input = log_input[i]
				tmp_loss = input[label]
				loss = loss + tmp_loss
			return -loss/self.num_class
		#https://zhuanlan.zhihu.com/p/49981234
		if loss_type == "Focal":
			sm_input = torch.softmax(inputs, dim=1)
			log_pt = torch.log(sm_input)
			pt = sm_input
			fl_pt = (1-pt)**self.gamma*log_pt
			for i, label in enumerate(targets):
				input = fl_pt[i]
				tmp_loss = input[label]
				loss = loss+tmp_loss
			return -loss/self.num_class
		#https://zhuanlan.zhihu.com/p/433048073
		if loss_type=="LabelSmooth":
			sm_input = torch.softmax(inputs, dim=1)
			log_input = torch.log(sm_input) # [B C]
			for i, input in enumerate(log_input):
				current_label = targets[i]
				for j in range(len(input)):
					if j==current_label:
						tmp_loss = (1-self.epsilon)*input[j]
						loss = tmp_loss+loss
					else:
						tmp_loss = self.epsilon/(self.num_class-1)*input[j]
						loss = loss+tmp_loss
			return -loss/self.num_class


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight,ignore_index=self.ignore_index)
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean',ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
                                                           ignore_index=self.ignore_index)
if __name__=="__main__":
	loss = Loss()
	inputs = torch.tensor([[-0.1342,-2.5835,-0.9810],
							[0.1867,-1.4513,-0.3225],
							[0.6272,-0.1120,0.3048]])
	targets = torch.tensor([0,2,1])
	out = loss(inputs, targets, loss_type="LabelSmooth")

	print(out)
	# focal_loss = FocalLoss()
	# focal_out = focal_loss(inputs, targets)
	# print(focal_out)
	label_smooth_loss = LabelSmoothingCrossEntropy()
	label_smooth_out = label_smooth_loss(inputs, targets)
	print(label_smooth_out)
