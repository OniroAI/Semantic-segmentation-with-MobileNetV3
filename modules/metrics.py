import tensorflow as tf

from modules.loss import fb_loss


class FbSegm():
    """Compute F_beta of segmentation."""

    def __init__(self, beta=1, channel=None, channel_axis=-1):
        self.channel = channel
        self.beta = beta
        self.channel_axis = channel_axis
        self.reset()

    def reset(self):
        self.se = 0
        self.count = 0

    def update(self, target_mask, pred_mask):
        if self.channel is not None:
            target_mask = target_mask[:, self.channel:(self.channel+1), :, :]
            pred_mask = pred_mask[:, self.channel:(self.channel+1), :, :]
        segm_fb = fb_loss(target_mask, pred_mask, self.beta, self.channel_axis)
        self.se += segm_fb * target_mask.shape[0]
        self.count += target_mask.shape[0]

    def compute(self):
        if not self.count:
            raise Exception('Must be at least one example for computation')
        else:
            return (self.se/self.count).numpy()
