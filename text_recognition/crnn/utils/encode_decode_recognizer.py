import math
from typing import List
import torch
import torch.nn.functional as F

from mmdet.models.builder import build_backbone, build_loss
from mmocr.models.builder import (build_convertor, build_decoder,
                                  build_encoder, build_preprocessor)
from .base import BaseRecognizer


class EncodeDecodeRecognizer(BaseRecognizer):
    """Base class for encode-decode recognizer."""

    def __init__(self,
                 preprocessor=None,
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 loss=None,
                 label_convertor=None,
                 train_cfg=None,
                 test_cfg=None,
                 max_seq_len=40,
                 pretrained=None):
        super().__init__()

        # Label convertor (str2tensor, tensor2str)
        assert label_convertor is not None
        # label_convertor.update(max_seq_len=max_seq_len)
        # self.label_convertor = build_convertor(label_convertor)
        self.label_convertor = label_convertor

        # Preprocessor module, e.g., TPS
        self.preprocessor = None
        # if preprocessor is not None:
        #     self.preprocessor = build_preprocessor(preprocessor)

        # Backbone
        assert backbone is not None
        # self.backbone = build_backbone(backbone)
        self.backbone = backbone

        # Encoder module
        self.encoder = None
        # if encoder is not None:
        #     self.encoder = build_encoder(encoder)
        # self.encoder = encoder

        # Decoder module
        assert decoder is not None
        # decoder.update(num_classes=self.label_convertor.num_classes())
        # decoder.update(start_idx=self.label_convertor.start_idx)
        # decoder.update(padding_idx=self.label_convertor.padding_idx)
        # decoder.update(max_seq_len=max_seq_len)
        # self.decoder = build_decoder(decoder)
        self.decoder = decoder

        # Loss
        assert loss is not None
        # loss.update(ignore_index=self.label_convertor.padding_idx)
        # self.loss = build_loss(loss)
        # self.loss = loss

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.max_seq_len = max_seq_len
        self.blank_idx = 0

        #don't init weights
        #self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of recognizer."""
        super().init_weights(pretrained)

        if self.preprocessor is not None:
            self.preprocessor.init_weights()

        self.backbone.init_weights()

        if self.encoder is not None:
            self.encoder.init_weights()

        self.decoder.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone."""
        if self.preprocessor is not None:
            img = self.preprocessor(img)

        x = self.backbone(img)

        return x

    def forward_train(self, img, img_metas):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                contains: 'img_shape', 'filename', and may also contain
                'ori_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        feat = self.extract_feat(img)

        gt_labels = [img_meta['text'] for img_meta in img_metas]

        targets_dict = self.label_convertor.str2tensor(gt_labels)

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat, img_metas)

        out_dec = self.decoder(
            feat, out_enc, targets_dict, img_metas, train_mode=True)

        loss_inputs = (
            out_dec,
            targets_dict,
            img_metas,
        )
        losses = self.loss(*loss_inputs)

        return losses

    def simple_test(self, img):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        feat = self.extract_feat(img)

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat, img_metas)

        out_dec = self.decoder(feat)
        # out_dec = out_dec[0]

        # label_indexes, label_scores = self.label_convertor.tensor2idx(out_dec)
        label_indexes, label_scores = self.tensor2idx(out_dec)
        # label_strings = self.label_convertor.idx2str(label_indexes)

        # flatten batch results
        #results = [dict(index=label_indexes, score=label_scores)]
        #for idx, score in zip(label_indexes, label_scores):
        #    results.append(dict(index=idx, score=score))

        return label_indexes, label_scores

    def merge_aug_results(self, aug_results):
        out_text, out_score = '', -1
        for result in aug_results:
            text = result[0]['text']
            score = sum(result[0]['score']) / max(1, len(text))
            if score > out_score:
                out_text = text
                out_score = score
        out_results = [dict(text=out_text, score=out_score)]
        return out_results

    def aug_test(self, imgs, img_metas):
        """Test function as well as time augmentation.

        Args:
            imgs (list[tensor]): Tensor should have shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): The metadata of images.
        """
        aug_results = []
        for img, img_meta in zip(imgs, img_metas):
            result = self.simple_test(img, img_meta)
            aug_results.append(result)

        return self.merge_aug_results(aug_results)

    def tensor2idx(self, output):

        valid_ratios = [1.0]
        topk = 1

        batch_size = output.size(0)
        output = F.softmax(output, dim=2)
        output = output.cpu().detach()
        batch_topk_value, batch_topk_idx = output.topk(topk, dim=2)
        batch_max_idx = batch_topk_idx[:, :, 0]
        scores_topk, indexes_topk = [], []
        scores, indexes = [], []
        feat_len = output.size(1)

        # for b in range(batch_size):
        b = 0
        valid_ratio = valid_ratios[b]
        decode_len = min(feat_len, math.ceil(feat_len * valid_ratio))

        pred = batch_max_idx[b, :]
        # select_idxs = []
        select_idx = torch.jit.annotate(List[int], [])
        prev_idx = self.blank_idx
        for t in range(decode_len):
            _t = t
            tmp_value = pred[_t].item()
            if tmp_value not in (prev_idx, self.blank_idx):
                select_idx.append(t)
            prev_idx = tmp_value

        # _select_idx = torch.LongTensor(select_idx)
        _select_idx = torch.tensor(select_idx) #.type(torch.int64)
        topk_value = torch.index_select(batch_topk_value[b, :, :], 0,
                                        _select_idx)  # valid_seqlen * topk
        topk_idx = torch.index_select(batch_topk_idx[b, :, :], 0,
                                      _select_idx)
        #print(f"top_idx shape :: {topk_idx.shape}")
        #print(f"top_value shape :: {topk_value.shape}")

        return torch.flatten(topk_idx), torch.flatten(topk_value)
        # topk_idx_list, topk_value_list = topk_idx.numpy().tolist(
        # ), topk_value.numpy().tolist()
        #     indexes_topk.append(topk_idx_list)
        #     scores_topk.append(topk_value_list)
        #     indexes.append([x[0] for x in topk_idx_list])
        #     scores.append([x[0] for x in topk_value_list])

        # if return_topk:
        #     return indexes_topk, scores_topk

        # return indexes, scores

class CRNNNet(EncodeDecodeRecognizer):
    """CTC-loss based recognizer."""
