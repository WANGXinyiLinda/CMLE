from transformers import BartConfig, BartForConditionalGeneration, RobertaForSequenceClassification
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from ipm_func import *

class GumbelBart(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BartConfig, clf_model: RobertaForSequenceClassification, 
            tokenizer, revert, max_gen_len, max_encode_len, hard_sample=False, cf_alpha = 1.0):
        super().__init__(config)
        self.clf_model = clf_model
        self.tokenizer = tokenizer
        self.hard_sample = hard_sample
        self.revert = revert # 'h' or 'p'
        self.max_gen_len = max_gen_len
        self.max_encode_len = max_encode_len
        self.max_clf_len = max_gen_len + max_encode_len - 1
        self.cf_alpha = cf_alpha

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        cf=False,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "cf": cf,
        }

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
        cf=True,
    ):
        if not cf:
            # print("exicuting forward in factual gumbelbart..")
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
                return_dict=return_dict)
        else:
            # print("exicuting forward in counterfactual gumbelbart..")
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
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True)
            
            clf_embed = self.clf_model.get_input_embeddings()
            input_attention_mask = attention_mask

            # sanity check
            # print("sanity check...")
            # temp = labels[:, 2:].clone()
            # temp = torch.cat((torch.zeros_like(labels[:,:1])+self.tokenizer.eos_token_id, temp),1)
            # temp[temp == -100] = self.tokenizer.pad_token_id
            # print("temp: ", temp)
            # clf_outputs = self.clf_model(attention_mask=torch.cat((input_attention_mask, temp!=self.tokenizer.pad_token_id), -1),
            #     input_ids=torch.cat((input_ids, temp), -1),
            #     labels=treatment_labels,
            #     return_dict=True)
            # print("factual cls loss: ", clf_outputs.loss)

            encoder_outputs = BaseModelOutput(
                last_hidden_state=outputs.encoder_last_hidden_state, 
                hidden_states=outputs.encoder_hidden_states, 
                attentions=outputs.encoder_attentions
            )

            randomness = torch.randint(1, 3, (treatment_labels.size()[0],), device=self.device)
            # print(randomness)
            rev_treatment_labels = (treatment_labels + randomness) % 3
            rev_treatment_label_ids = rev_treatment_labels + len(self.tokenizer) - 3
            rev_treatment_label_ids = torch.unsqueeze(rev_treatment_label_ids, 1)
            # print("rev_treatment_label_ids: ", rev_treatment_label_ids)
            rev_decoder_input_ids = torch.cat((rev_treatment_label_ids, 
                torch.zeros_like(rev_treatment_label_ids, device=self.device) + self.tokenizer.bos_token_id), 1)
            # print("rev_decoder_input_ids: ", rev_decoder_input_ids)
            # print("input ids: ", input_ids)
            rev_outputs = super().generate( # greedy search
                input_ids=input_ids,
                encoder_outputs=encoder_outputs,
                decoder_input_ids=rev_decoder_input_ids,
                max_length=self.max_gen_len, 
                num_beams=1,
                num_beam_groups=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                cf=False
                )
            logits = rev_outputs.scores
            rev_output_ids = rev_outputs.sequences # longer by 2
            # print("greedy search output ids: ", rev_output_ids)
            output_attention_mask = rev_output_ids[:,1:] != self.tokenizer.pad_token_id
            # print("output_attention_mask: ", output_attention_mask)
            # print("rev output scores: ", logits)
            # print("num tensors: ", len(logits))
            # print("tensor shape: ", logits[0].size())
            logits = [torch.unsqueeze(x, 1) for x in logits]
            logits = torch.cat(logits, 1)
            # print("rev output scores shape: ", logits.size())
            output_tokens = F.gumbel_softmax(logits, tau=1, hard=self.hard_sample)
            # print("gumbelbart out: ", output_tokens)
            # print("gumbelbart out shape: ", output_tokens.size())
            first_out_token_probs = torch.zeros_like(output_tokens[:,:1,:], device=self.device)
            if self.revert == 'h':
                input_tokens = input_ids
                first_out_token_probs[:,:,self.tokenizer.sep_token_id] = 1
                output_tokens = torch.cat((first_out_token_probs, output_tokens), 1)
            else:
                first_out_token_probs[:,:,self.tokenizer.bos_token_id] = 1
                output_tokens = torch.cat((first_out_token_probs, output_tokens), 1)
                input_tokens = torch.cat((torch.zeros_like(input_ids[:,:1], device=self.device) 
                    + self.tokenizer.sep_token_id, input_ids[:,1:]), 1)
            output_tokens = output_tokens[:,:,:-3]
            output_clf_embed = torch.matmul(output_tokens, clf_embed.weight)
            input_clf_embed = clf_embed(input_tokens)
            # print("out embedding shape: ", output_clf_embed.size())
            if self.revert == 'h':
                inputs_embeds = torch.cat((input_clf_embed, output_clf_embed), 1)
                attention_mask = torch.cat((input_attention_mask, output_attention_mask), 1)
            else:
                inputs_embeds = torch.cat((output_clf_embed, input_clf_embed), 1)
                attention_mask = torch.cat((output_attention_mask, input_attention_mask), 1)
            # # pad to max len
            # inputs_embeds = torch.cat(inputs_embeds, 
            #     torch.zeros((inputs_embeds.size()[0], 
            #         self.max_clf_len-inputs_embeds.size()[1], 
            #         inputs_embeds.size()[2]), device=self.device), 1)
            # attention_mask = torch.cat(attention_mask, 
            #     torch.zeros_like((attention_mask.size()[0], 
            #         self.max_clf_len-attention_mask.size()[1]), device=self.device), 1)
            # print("clf input embedding shape: ", inputs_embeds.size())
            # print("clf input attention mask shape: ", attention_mask.size())

            clf_outputs = self.clf_model(attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=rev_treatment_labels,
                return_dict=True)
            clf_loss = clf_outputs.loss
            # print("counterfactual cls loss: ", clf_outputs.loss)

            # print("factual gen loss: ", outputs.loss)
            outputs.loss += self.cf_alpha * clf_loss
            # print("total loss: ", outputs.loss)
        return outputs