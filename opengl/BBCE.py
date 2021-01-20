import typing

import torch


def bootstrapped_cross_entropy(
    inputs,
    targets,
    iteration,
    p: float,
    warmup: typing.Union[typing.Callable[[float, int], float], int] = -1,
    weight=None,
    ignore_index=-100,
    reduction: typing.Callable[[torch.Tensor], torch.Tensor] = torch.mean,
):
    if not 0 < p < 1:
        raise ValueError("p should be in [0, 1] range, got: {}".format(p))

    if isinstance(warmup, int):
        this_p = 1.0 if iteration < warmup else p
    elif callable(warmup):
        this_p = warmup(p, iteration)
    else:
        raise ValueError(
            "warmup should be int or callable, got {}".format(type(warmup))
        )

    # Shortcut
    if this_p == 1.0:
        return torch.nn.functional.cross_entropy(
            inputs, targets, weight, ignore_index=ignore_index, reduction=reduction
        )

    raw_loss = torch.nn.functional.cross_entropy(
        inputs, targets, weight=weight, ignore_index=ignore_index, reduction="none"
    ).view(-1)
    num_pixels = raw_loss.numel()

    loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
    return reduction(loss)


class BoostStrappedCrossEntropy(torch.nn.modules.loss._WeightedLoss):
    def __init__(
        self,
        p: float,
        warmup: typing.Union[typing.Callable[[float, int], float], int] = -1,
        weight=None,
        ignore_index=-100,
        reduction: typing.Callable[[torch.Tensor], torch.Tensor] = torch.mean,
    ):
        self.p = p
        self.warmup = warmup
        self.ignore_index = ignore_index
        self._current_iteration = -1

        super().__init__(weight, size_average=None, reduce=None, reduction=reduction)

    def forward(self, inputs, targets):
        self._current_iteration += 1
        return bootstrapped_cross_entropy(
            inputs,
            targets,
            self._current_iteration,
            self.p,
            self.warmup,
            self.weight,
            self.ignore_index,
            self.reduction,
        )


class BootstrappedCE(torch.nn.Module):
    def __init__(self, start_warm=1000, end_warm=5000, top_p=0.25):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return torch.nn.functional.binary_cross_entropy(input, target), 1.0

        raw_loss = torch.nn.functional.binary_cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * self.top_p), sorted=False)
        return loss.mean(), self.top_p
