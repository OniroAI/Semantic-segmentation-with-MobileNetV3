import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy,\
    SparseCategoricalCrossentropy


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
    def __init__(self, channel_axis=-1, fb_weight=0.5, fb_beta=1,
                 entropy_weight=0.5, use_bce=True, normalize=False):
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
