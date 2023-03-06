import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ops.histogram_matching import histogram_matching

# from torchvision.utils import save_image


class HistogramLoss(nn.Module):
    def __init__(self):
        super(HistogramLoss, self).__init__()

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

    def forward(self, input_data, target_data, mask_src, mask_tar, source_data=None):
        assert mask_src.shape[0] == 1 and mask_src.shape[1] == 1
        index_tmp = mask_src.squeeze(0).squeeze(0).nonzero() # [[h, w], .. [h, w]] n * 2
        x_A_index = index_tmp[:, 1]
        y_A_index = index_tmp[:, 0]
        index_tmp = mask_tar.squeeze(0).squeeze(0).nonzero()
        x_B_index = index_tmp[:, 1]
        y_B_index = index_tmp[:, 0]

        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar

        if source_data is not None:
            source_data = (self.de_norm(source_data) * 255).squeeze()
            source_masked = source_data * mask_src

            input_match = histogram_matching(
                source_masked, target_masked,
                [y_A_index, x_A_index, y_B_index, x_B_index])

            # save_image(torch.cat([source_data, source_masked, target_data, target_masked, input_data, input_masked, input_match], dim=-1), 'his_after.jpg', normalize=True)
        else:
            input_match = histogram_matching(
                input_masked, target_masked,
                [y_A_index, x_A_index, y_B_index, x_B_index])

            # save_image(torch.cat([target_data, target_masked, input_data, input_masked, input_match], dim=-1), 'his_after.jpg', normalize=True)

        input_match = self.to_var(input_match, requires_grad=False)
        loss = F.l1_loss(input_masked, input_match)
        return loss
