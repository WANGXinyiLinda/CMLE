# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
from ..lxrt.entry import LXRTEncoder
from ..lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_HAL_LENGTH = 20


class HalModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            opt,
            max_seq_length=MAX_HAL_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        
        # hallucination classification heads
        self.logit_fc = nn.Linear(hid_dim, 2)
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent, input_word_embeddings=None, mask=None, seg=None):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos), 
            input_word_embeddings=input_word_embeddings, input_mask=mask, segment_ids=seg)
        logit = self.logit_fc(x)

        return logit

    def get_word_embeddings(self):
        return self.lxrt_encoder.model.bert.embeddings.word_embeddings

