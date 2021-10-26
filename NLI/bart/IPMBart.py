from transformers import BartConfig, BartForConditionalGeneration
from torch import nn
from torch.nn import CrossEntropyLoss
from ipm_func import *

class IPMBart(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BartConfig, treatment_probs = [0.33]*3, ipm_alpha = 1.0):
        super().__init__(config)
        self.ipm_alpha = ipm_alpha
        self.treatment_probs = treatment_probs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        treatment_labels=None,
    ):
        outputs = super().forward(input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True)
        encode_rep = outputs.encoder_last_hidden_state
        encode_rep = torch.reshape(encode_rep, (encode_rep.size()[0], -1))
        ipm_loss = 0.0
        for l in range(len(self.treatment_probs)):
            imb_dist, imb_mat = wasserstein(encode_rep, treatment_labels, self.treatment_probs[l], l, device=self.device)
            ipm_loss += imb_dist
            # print("ipm loss for ", l, ": ", imb_dist)
        # print("CE loss: ", outputs.loss)
        outputs.loss += self.ipm_alpha * ipm_loss
        # print("total loss: ", outputs.loss)
        return outputs