import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any

# pylint:disable=no-member


class SequenceUnlikelihoodLoss(nn.Module):
    def __init__(
        self,
        padding_idx=50257,
        rank_alpha=0.2,
        candidate_type="prev_context",
        debug=False,
    ):
        super().__init__()
        # tokenizer.encoder["[PAD]"]
        self.rank_alpha = rank_alpha
        self.debug = debug
        self.candidate_type = candidate_type
        self.padding_idx = padding_idx

    def forward(self, logits, targets):
        # ============ unlikelihood loss ============ #
        # -- mle loss
        # shape : (batch * sequence_length, num_classes)
        target = targets
        logits_flat = logits.view(-1, logits.size(-1))
        # shape : (batch * sequence_length, num_classes)
        lprobs = F.log_softmax(logits_flat, dim=-1)
        # lprobs = lprobs.view(-1, lprobs.size(-1))

        # shape : (batch * max_len, 1)
        target = target.view(-1).long()

        true_token_lprobs = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="none",
        )
        mle_loss = true_token_lprobs.sum()

        with torch.no_grad():
            if self.candidate_type == "prev_context":
                ctx_cands = target.unsqueeze(0).expand(
                    target.size(0), target.size(0)
                )  # torch.Size([64, 64])
                ctx_cands_ = ctx_cands.tril(-1) + self.padding_idx
                ctx_cands_ = ctx_cands_ * ctx_cands_.triu()
                ctx_cands = ctx_cands.tril(-1) + ctx_cands_

                # Don't include the target for that timestep as a negative target.
                ctx_cands = ctx_cands.masked_fill(
                    ctx_cands == target.unsqueeze(1), self.padding_idx
                )
                ctx_cands = ctx_cands.masked_fill(
                    ctx_cands == (self.padding_idx ** 2), self.padding_idx
                )
                negative_targets = torch.zeros_like(lprobs).scatter_(1, ctx_cands, 1)
            else:
                raise NotImplementedError("candidate type %s" % self.candidate_type)

        one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-5)

        custom_loss = -torch.log(one_minus_probs) * negative_targets
        custom_loss = custom_loss.sum()

        loss = mle_loss + self.rank_alpha * custom_loss

        # =================
        return loss
