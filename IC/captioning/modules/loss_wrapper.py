import torch
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward
from ..utils.misc import decode_sequence

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.use_hal = self.opt.use_hal
        self.ipm_alpha = self.opt.ipm_alpha
        self.gumbel_alpha = self.opt.gumbel_alpha
        if opt.label_smoothing > 0:
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = losses.LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion()
        self.struc_crit = losses.StructureLosses(opt)
        if self.ipm_alpha > 0:
            self.ipm_crit = losses.CounterFactualIPM(opt)
        if self.gumbel_alpha > 0:
            self.gumbel_crit = losses.GumbelCriterion(opt)

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, feat=None, pos=None, target=None, counterfactual_weight=0.0):
        opt = self.opt
        input_seq = labels[..., :-1]
        out_seq = labels[..., 1:]
        out_mask = masks[..., 1:]
        if opt.use_hal:
            hal_label = labels[..., 0]
            logprob, encoder_out = self.model(fc_feats, att_feats, input_seq, att_masks)
        else:
            logprob = self.model(fc_feats, att_feats, input_seq, att_masks)
        
        out = {}
        loss = 0.0
        if self.ipm_alpha > 0:
            if counterfactual_weight > 0:
                ipm_loss = self.ipm_crit(encoder_out, hal_label)
                loss += counterfactual_weight * ipm_loss
            else:
                ipm_loss = torch.tensor(0.0, device = fc_feats.device)
            out['ipm_loss'] = ipm_loss
        if self.gumbel_alpha > 0:
            if counterfactual_weight > 0:
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    mode='sample',
                    opt={'sample_method': 'greedy',
                        'beam_size': 1})
                print("sequence feed to classifier...")
                print(decode_sequence(self.model.vocab, gen_result.data, self.opt))
                gumbel_loss = self.gumbel_crit(gen_result, sample_logprobs, feat, pos, target)
                loss += counterfactual_weight * gumbel_loss
            else:
                gumbel_loss = torch.tensor(0.0, device = fc_feats.device)
            out['gumbel_loss'] = gumbel_loss
        if struc_flag:
            if opt.structure_loss_weight < 1:
                lm_loss = self.crit(logprob, out_seq, out_mask)
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if opt.structure_loss_weight > 0:
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                            or not 'margin' in opt.structure_loss_type,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts)
            else:
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                              'reward': torch.tensor(0).type_as(fc_feats)}
            loss += (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
        elif not sc_flag:
            lm_loss = self.crit(logprob, out_seq, out_mask)
            loss += lm_loss
            out['lm_loss'] = lm_loss
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks,
                    mode='sample',
                    opt={'sample_method': opt.sc_sample_method,
                         'beam_size': opt.sc_beam_size})
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).to(sample_logprobs)
            loss += self.rl_crit(sample_logprobs, gen_result.data, reward)
            out['reward'] = reward[:,0].mean()
        
        out['loss'] = loss
        print("losses: ", out)
        return out
