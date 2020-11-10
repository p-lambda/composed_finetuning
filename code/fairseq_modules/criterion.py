from torch import nn
import math
import numpy as np
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss

from fairseq.data import encoders
import torch


def get_lens(tokens, pad_idx):
    return torch.sum(tokens != pad_idx, dim=1)

def pad(tokens, pad_idx, eos_idx):
    tokens = tokens.clone()
    lens = []
    for i in range(tokens.shape[0]):
        eos = torch.nonzero(tokens[i] == eos_idx, as_tuple=False)
        if len(eos) > 0:
            tokens[i][eos[0]+1:] = pad_idx
            lens.append(eos[0])
        else:
            lens.append(tokens.shape[1])
    return tokens, torch.tensor(lens).cuda()

def pad_before(tokens, pad_idx, eos_idx):
    tokens = [tok_list[tok_list != pad_idx] for tok_list in tokens]
    lens = [len(tok_list) for tok_list in tokens]
    max_len = np.amax(lens)
    new_tokens = []
    for tok_list in tokens:
        new_tokens.append(torch.cat([(pad_idx * torch.ones(max_len - len(tok_list))).long().cuda(), tok_list, torch.tensor([eos_idx]).long().cuda()]))
    return torch.stack(new_tokens)


def pad_after(tokens, pad_idx, eos_idx):
    tokens = [tok_list[tok_list != pad_idx] for tok_list in tokens]
    lens = [len(tok_list) for tok_list in tokens]
    max_len = np.amax(lens)
    new_tokens = []
    for tok_list in tokens:
        new_tokens.append(torch.cat([tok_list, torch.tensor([eos_idx]).long().cuda(), (pad_idx * torch.ones(max_len - len(tok_list))).long().cuda()]))
    return torch.stack(new_tokens)


def pad_to_len(tokens, desired_length, pad_idx):
    if tokens.shape[1] >= desired_length:
        return tokens
    pad = (torch.ones((tokens.shape[0], desired_length - tokens.shape[1])) * pad_idx).long().cuda()
    return torch.cat([tokens, pad], dim=1)


@register_criterion('reinforce_criterion')
class ReinforceCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, ce_loss_lambda):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.task = task
        self.ce_loss_lambda = ce_loss_lambda

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--ce-loss-lambda', default=0., type=float, metavar='D',
                            help='lambda for ce loss')
        # fmt: on

    def get_reinforce_loss(self, model, sample, target):
        # this function implicitly makes the model go to eval mode
        yhat_obj, yhat, p_yhat, yhat_prev = model._get_sample(**sample['net_input'], training=True)
        # fix padding
        yhat_pad = pad_before(yhat, self.padding_idx, model.generator.eos)

        yhat_lens = get_lens(yhat_pad, self.padding_idx)
        net_output_2 = model(yhat_pad, yhat_lens, sample['net_input']['prev_output_tokens'], output_mode='second')
        lprobs_2 = model.get_normalized_probs(net_output_2, log_probs=True)
        lprobs_2 = lprobs_2.view(-1, lprobs_2.size(-1))
        loss_2_smoothing = 0.0
        loss_2, nll_loss_2 = label_smoothed_nll_loss(
            lprobs_2, target.view(-1, 1), loss_2_smoothing, ignore_index=self.padding_idx, reduce=False,
        )
        # replace the nll_loss from above with the stacked one
        nll_loss_2 = nll_loss_2.sum()

        loss_2 = loss_2.view(target.shape).sum(dim=1)
        probs = p_yhat.log_prob(yhat.view(-1)).contiguous()
        probs = probs.view(-1, yhat.shape[1]).sum(dim=1)
        # clamp the probs to prevent divergence
        probs = probs.clamp(min=-50)
        # the loss is roughly a order of magnitude larger than CE loss, we normalize here
        reinforce_loss = (probs * loss_2).sum() / 10.0

        stacked_pred = lprobs_2.max(1)[1]
        stacked_acc = (stacked_pred == target.view(-1)).sum().float()
        return reinforce_loss, stacked_acc, loss_2, nll_loss_2

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'], output_mode='first')
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target.view(-1, 1), self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        loss *= self.ce_loss_lambda

        num_samples = 1
        reinforce_loss_outs = [
            self.get_reinforce_loss(model, sample, target) for i in range(num_samples)]
        reinforce_loss = torch.mean(torch.stack([r[0] for r in reinforce_loss_outs]))
        stacked_acc = torch.mean(torch.stack([r[1] for r in reinforce_loss_outs]))
        loss_2 = torch.mean(torch.stack([r[2] for r in reinforce_loss_outs]))
        nll_loss_2 = torch.mean(torch.stack([r[3] for r in reinforce_loss_outs]))

        loss += reinforce_loss

        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'reinforce_loss': reinforce_loss.sum().data,
            'loss_2': loss_2.sum().data,
            'nll_loss_2': nll_loss_2.data,
            'stacked_acc': stacked_acc.data,
            'ntokens': sample['ntokens'],
            'target_ntokens': float(len(target.view(-1))),
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        loss_2_sum = utils.item(sum(log.get('loss_2', 0) for log in logging_outputs))
        reinforce_sum = utils.item(sum(log.get('reinforce_loss', 0) for log in logging_outputs))
        stacked_acc_sum = utils.item(sum(log.get('stacked_acc', 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        nll_loss_2_sum = utils.item(sum(log.get('nll_loss_2', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        target_ntokens = utils.item(sum(log.get('target_ntokens', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('loss_2', loss_2_sum / target_ntokens / math.log(2), target_ntokens, round=5)
        metrics.log_scalar('reinforce_loss', reinforce_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('stacked_acc', stacked_acc_sum / target_ntokens, target_ntokens, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('nll_loss_2', nll_loss_2_sum / target_ntokens / math.log(2), target_ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion('denoise_criterion')
class DenoiseCLSCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def diff_token_check(self, src, tgt):
        max_len = max(src.shape[1], tgt.shape[1])
        src = pad_to_len(src, max_len, self.padding_idx)
        tgt = pad_to_len(tgt, max_len, self.padding_idx)
        return ((src - tgt).sum(1) != 0).unsqueeze(1)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_tokens = sample['net_input']['src_tokens']
        src_lengths = sample['net_input']['src_lengths']
        prev_output_tokens = sample['net_input']['prev_output_tokens']
        cls_targets = self.diff_token_check(src_tokens, sample['target']).float()

        encoder_out = model.encoder(src_tokens,
                                    src_lengths=src_lengths,
                                    return_all_hiddens=True)

        sample_size = sample['target'].size(0)
        # classify whether it is correct
        cls_out = model.classifier(encoder_out.encoder_out[-1])
        # compute targets
        loss = torch.nn.BCEWithLogitsLoss(reduction='sum')(cls_out, cls_targets)

        pred = (cls_out > 0).float()
        acc = (pred == cls_targets).sum().float()

        logging_output = {
            'loss': loss.data,
            'acc': acc.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        acc_sum = sum(log.get('acc', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('acc', acc_sum / sample_size, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
