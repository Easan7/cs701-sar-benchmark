"""LoRA and MoE-LoRA adapters for the attention layers of a frozen backbone."""
from torch import nn


class LoRA(nn.Module):
    """y = W x + (alpha / r) * B A x, where W is the wrapped layer (frozen by add_adapters)."""

    def __init__(self, base: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)  # A
        self.up = nn.Linear(rank, base.out_features, bias=False)  # B
        nn.init.zeros_(self.up.weight)  # B = 0: training starts from exactly the pretrained layer
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * self.up(self.down(x))


class MoELoRA(nn.Module):
    """y = W x + (alpha / r) * sum_e g_e(x) B_e A_e x, with gates g(x) = softmax(router[task](x)) per token.

    The experts' A_e are stacked in `down` and their B_e side by side in `up`, so gating each
    expert's r hidden units and applying `up` sums the experts.
    """

    def __init__(self, base: nn.Linear, experts=4, rank=4, alpha=8, tasks=2):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, experts * rank, bias=False)  # A_1 ... A_E
        self.up = nn.Linear(experts * rank, base.out_features, bias=False)  # B_1 ... B_E
        nn.init.zeros_(self.up.weight)
        self.routers = nn.ModuleList(nn.Linear(base.in_features, experts, bias=False) for _ in range(tasks))
        self.rank = rank
        self.scale = alpha / rank
        self.task = 0  # which router to use; see set_task

    def forward(self, x):
        gates = self.routers[self.task](x).softmax(dim=-1)
        gates = gates.repeat_interleave(self.rank, dim=-1)  # expert e's gate on each of its r hidden units
        return self.base(x) + self.scale * self.up(self.down(x) * gates)


def add_adapters(backbone, kind, rank=16, alpha=32, unfreeze_blocks=2):
    """Adapt attention, or fully tune the selected final blocks while adapting earlier blocks."""
    if rank < 1 or alpha < 1 or unfreeze_blocks < 1:
        raise ValueError('LoRA rank, alpha, and number of unfrozen blocks must be positive')
    if kind == 'moelora' and rank < 4:
        raise ValueError('MoE-LoRA rank must be at least 4')
    if kind in {'frozen', 'partial'}:
        backbone.requires_grad_(False)
        if kind == 'partial':
            blocks = backbone.blocks
            for block in blocks[-unfreeze_blocks:]:
                block.requires_grad_(True)
            if hasattr(backbone, 'vit'):
                backbone.vit.norm.requires_grad_(True)
        return
    adapter = {'lora': LoRA, 'moelora': MoELoRA, 'hybrid': LoRA}[kind]
    backbone.requires_grad_(False)
    blocks = backbone.blocks
    for blk in blocks[:-unfreeze_blocks] if kind == 'hybrid' else blocks:
        if kind == 'moelora':
            blk.attn.qkv = MoELoRA(blk.attn.qkv, rank=max(1, rank // 4), alpha=alpha // 4)
            blk.attn.proj = MoELoRA(blk.attn.proj, rank=max(1, rank // 4), alpha=alpha // 4)
        else:
            blk.attn.qkv = adapter(blk.attn.qkv, rank=rank, alpha=alpha)
            blk.attn.proj = adapter(blk.attn.proj, rank=rank, alpha=alpha)
    if kind == 'hybrid':
        for blk in blocks[-unfreeze_blocks:]:
            blk.requires_grad_(True)
        # The final token normalization also needs to track the changed representations.
        if hasattr(backbone, 'vit'):
            backbone.vit.norm.requires_grad_(True)


def configure_adaptation(backbone, kind, rank=16, alpha=32, unfreeze_blocks=2):
    """Apply a named adaptation strategy to transformer or hierarchical CNN backbones."""
    if unfreeze_blocks < 1:
        raise ValueError('unfreeze_blocks must be positive')
    if hasattr(backbone, 'feature_channels'):
        backbone.requires_grad_(False)
        if kind == 'full':
            backbone.requires_grad_(True)
        elif kind == 'partial':
            stages = backbone.blocks
            for stage in list(stages)[-unfreeze_blocks:]:
                stage.requires_grad_(True)
        elif kind != 'frozen':
            raise ValueError(f'{kind} adapters are transformer-only; use full, partial, or frozen for this backbone')
        return
    if kind == 'full':
        return
    if kind not in {'lora', 'moelora', 'hybrid', 'partial', 'frozen'}:
        raise ValueError(f'Unsupported adaptation: {kind}')
    add_adapters(backbone, kind, rank=rank, alpha=alpha, unfreeze_blocks=unfreeze_blocks)


def set_task(module, task):
    """Route every MoELoRA inside `module` through the router of `task` (0 = classification, 1 = detection)."""
    for layer in module.modules():
        if isinstance(layer, MoELoRA):
            layer.task = task
