"""不修改训练模型或 checkpoint 的推理专用 Conv–BN 融合。"""

from __future__ import annotations

from copy import deepcopy

from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def fuse_sequential_conv_bn_eval(model: nn.Module) -> tuple[nn.Module, int]:
    """复制 eval 模型，只融合 Sequential 中执行顺序明确的相邻 Conv/BN。

    GroupNorm、LayerNorm、无固定统计量的 BatchNorm 和自定义 forward 分支
    都不猜测融合；训练模型及磁盘 checkpoint 保持原样。
    """

    if any(module.training for module in model.modules()):
        raise ValueError("Conv–BN inference fusion requires an eval() model.")
    inference_model = deepcopy(model)

    def fuse_children(parent: nn.Module) -> int:
        count = sum(fuse_children(child) for child in parent.children())
        if not isinstance(parent, nn.Sequential):
            return count
        children = list(parent._modules.items())
        for (conv_name, conv), (bn_name, bn) in zip(children, children[1:]):
            if (
                isinstance(conv, nn.Conv2d)
                and isinstance(bn, nn.BatchNorm2d)
                and bn.running_mean is not None
                and bn.running_var is not None
            ):
                parent._modules[conv_name] = fuse_conv_bn_eval(conv, bn)
                parent._modules[bn_name] = nn.Identity().eval()
                count += 1
        return count

    count = fuse_children(inference_model)
    return inference_model.eval(), count
