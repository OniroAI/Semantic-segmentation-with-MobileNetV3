import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy


def fb_loss(trues, preds, beta, channel_axis):
    smooth = 1e-4
    beta2 = beta*beta
    batch = preds.shape[0]
    classes = preds.shape[channel_axis]
    preds = tf.reshape(preds, [batch, classes, -1])
    trues = tf.reshape(trues, [batch, classes, -1])
    trues_raw = tf.reduce_sum(trues, axis=-1)
    weights = tf.clip_by_value(trues_raw, 0., 1.)
    TP_raw = preds * trues
    TP = tf.reduce_sum(TP_raw, axis=2)
    FP_raw = preds * (1-trues)
    FP = tf.reduce_sum(FP_raw, axis=2)
    FN_raw = (1-preds) * trues
    FN = tf.reduce_sum(FN_raw, axis=2)
    Fb = ((1+beta2) * TP + smooth)/((1+beta2) * TP + beta2 * FN + FP + smooth)
    Fb = Fb * weights
    score = tf.reduce_sum(Fb) / (tf.reduce_sum(weights) + smooth)
    return tf.clip_by_value(score, 0., 1.)

def make_cross_entropy_target(target):
    # target = target.byte()
    b, c, w, h = target.shape
    ce_target = tf.zeros((b, w, h))
    for channel in range(c):
        ce_target = tf.where(target[:, channel, :, :], channel, ce_target)
    return ce_target


class FBLoss:
    def __init__(self, beta=1, channel_axis=-1):
        self.beta = beta
        self.channel_axis = channel_axis

    def __call__(self, target, output):
        return 1 - fb_loss(target, output, self.beta, self.channel_axis)


class FbCombinedLoss:
    def __init__(self, channel_axis=-1, fb_weight=0.5, fb_beta=1, entropy_weight=0.5, use_bce=True, normalize=False):
        self.fb_weight = fb_weight
        self.entropy_weight = entropy_weight
        self.fb_loss = FBLoss(beta=fb_beta, channel_axis=channel_axis)
        self.use_bce = use_bce
        self.normalize = normalize
        if use_bce:
            self.entropy_loss = BinaryCrossentropy()
        else:
            self.entropy_loss = SparseCategoricalCrossentropy()

    def __call__(self, target, output):
        if self.normalize:
            output = F.softmax(output, dim=1)
        if self.fb_weight > 0:
            fb = self.fb_loss(target, output) * self.fb_weight
        else:
            fb = 0
        if self.entropy_weight > 0:
            if self.use_bce is False:
                target = make_cross_entropy_target(target)
            ce = self.entropy_loss(target, output) * self.entropy_weight
        else:
            ce = 0
        return fb + ce


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)):
#             self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list):
#             self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         target = target.long()
#         if input.dim()>2:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1, 1)
#
#         logpt = F.log_softmax(input, dim=1)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = logpt.exp()
#
#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * at
#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()
#
#
# class CrossEntropy(nn.Module):
#     def __init__(self):
#         super(CrossEntropy, self).__init__()
#         self.criterion = nn.CrossEntropyLoss()
#
#     def forward(self, input, target):
#         target = target.long()
#         loss = self.criterion(input, target)
#         return loss
#
#
# def make_cross_entropy_target(target):
#     target = target.byte()
#     b, c, w, h = target.shape
#     ce_target = torch.zeros(b, w, h).type_as(target)
#     for channel in range(c):
#         ce_target.masked_fill_(target[:, channel, :, :], channel)
#     return ce_target.long()