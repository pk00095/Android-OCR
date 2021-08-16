import torch.nn as nn
from .weight_init import xavier_init


class BaseDecoder(nn.Module):
    """Base decoder class for text recognition."""

    def __init__(self, **kwargs):
        super().__init__()

    def init_weights(self):
        pass

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward(self, feat):
        self.train_mode = False

        return self.forward_test(feat)

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super().__init__()

        # Original
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNNDecoder(BaseDecoder):

    def __init__(self,
                 in_channels=None,
                 num_classes=None,
                 rnn_flag=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.rnn_flag = rnn_flag

        if rnn_flag:
            self.decoder = nn.Sequential(
                BidirectionalLSTM(in_channels, 256, 256),
                BidirectionalLSTM(256, 256, num_classes))
        else:
            self.decoder = nn.Conv2d(
                in_channels, num_classes, kernel_size=1, stride=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)

    def forward_train(self, feat):
        assert feat.size(2) == 1, 'feature height must be 1'
        if self.rnn_flag:
            x = feat.squeeze(2)  # [N, C, W]
            x = x.permute(2, 0, 1)  # [W, N, C]
            x = self.decoder(x)  # [W, N, C]
            outputs = x.permute(1, 0, 2).contiguous()
        else:
            x = self.decoder(feat)
            x = x.permute(0, 3, 1, 2).contiguous()
            n, w, c, h = x.size()
            outputs = x.view(n, w, c * h)
        return outputs

    def forward_test(self, feat):
        return self.forward_train(feat)
