# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion 

@register_criterion('ngram_language_loss')
class NgramLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss
        self.attn_weight = args.attn_weight
        self.attn_str = args.attn_str
        self.attn_type = args.attn_type
        self.ngram = args.ngram

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true',
                            help='only comput basic stat')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        logits_list, attn_dict = model(**sample['net_input'], return_all_hiddens=False)
        attn = attn_dict['attn']
        targets = model.get_targets(sample, [logits_list[0]])


        ngram = len(logits_list)
        # [B, ngram, T]
        expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
        for i in range(ngram):
            if i > 0 and self.disable_ngram_loss:
                break

            padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
            if 'target_idx' in sample:
                expend_targets[i,:,:] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
            else:
                expend_targets[i,:,:] = targets
        targets = expend_targets

        logits = torch.cat(logits_list, dim=0) #.view(ngram, *logits_list[0].size())

        lprobs = F.log_softmax(
                    logits.view(-1, logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                )

        loss = F.nll_loss(
               lprobs,
               targets.view(-1),
               reduction='sum',
               ignore_index=self.padding_idx,
               )
        # print('______________________________')
        if True and 'word_align' in sample:
            if self.attn_str=='SMA':
                attn = attn[0]
            elif self.attn_str=='SA':
                attn = attn.mean(0)
            attn = attn.transpose(0, 1).chunk(1 + self.ngram, 0)[1:]
            # attn.transpose(0, 1).chunk(1 + self.ngram, 0)
            # sample_id = 0
            # src_words = [sample['src_dict'][a] for a in sample['net_input']['src_tokens'][sample_id].tolist()]
            # target_words = [sample['src_dict'][a] for a in sample['target'][sample_id].tolist()]
            # aligned_words = [target_words[b] for b in src_words.argmax(-1).tolist()]
            align_target = sample['word_align']
            for copy_attn in attn:
                copy_attn = copy_attn .transpose(0,1)
                if align_target.max().item() == 0:
                    x = 1
                    align_loss = 0
                else:
                    if self.attn_type == 'SCE':
                        word_loss = (copy_attn * (align_target > 0).float()).sum(-1)
                        idf_weight = (align_target * (align_target > 0).float()).sum(-1)/(align_target > 0).float().sum(-1)
                        t_loss = -(word_loss[word_loss > 0]  + 1e-20).log()* idf_weight[word_loss > 0]
                        align_loss = (t_loss).sum()
                    elif self.attn_type == 'CE':
                        idf_weight = torch.masked_select(align_target, (align_target > 0).byte())
                        t_loss = -(torch.masked_select(copy_attn, (align_target>0).byte()) + 1e-20).log()
                        scale = torch.masked_select((align_target>0).sum(-1).unsqueeze(-1).expand(align_target.size()),
                                                    (align_target>0).byte())
                        scale[scale == 0] = 1
                        align_loss = (t_loss * idf_weight / scale.float()).sum()  # /len(t_loss)


                word_align_loss = align_loss
                # print('___________self.attention_weight')
                loss += word_align_loss * self.attn_weight
                # print('_____________________________loss' + str(loss))
                # print('_____________________________align_loss' + str(word_align_loss))

        if self.eps > 0.:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()

            eps_i = self.eps / lprobs.size(-1)
            loss = (1. - self.eps) * loss + eps_i * smooth_loss

        sample_size = targets.ne(self.padding_idx).int().sum().item()

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
