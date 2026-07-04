
"""CIFAR-100 single-A100 speedrun baseline (experimental knobs).

This is an ADDITIVE copy of train_cifar100_resnet_muon.py. Every new
C100_* knob defaults to the frozen baseline behavior, so running this
script with NO new env vars set reproduces the baseline byte-for-byte
(same architecture, fp16, cosine schedule, LS 0.05, plain heavy-ball
Muon, Newton-Schulz steps 5, reflect-pad 4).

BENCHMARK CONTRACT
- Validation is frozen. Do not optimize, tune, branch, adapt, augment, or compile
  against the validation path as a benchmark improvement. Validation is an
  untimed pass/fail gate only.
- Compilation, data staging, warmup, logging, and measurement boundaries are
  benchmark infrastructure. Do not optimize them for record claims. They may
  only be changed to fix correctness/portability bugs while preserving semantics.
- The only admissible optimization surfaces are model architecture, optimizer,
  and training hyperparameters inside the timed training loop.
"""

import math
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


torch.backends.cudnn.benchmark = True
# Precision knob: fp16 (baseline) casts model+data to float16; bf16 casts to
# bfloat16 instead. MEAN/STD carry the same dtype so normalize() stays in-dtype.
# Loss is always computed via logits.float() regardless of this setting.
_PRECISION = os.getenv("C100_PRECISION", "fp16")
_DTYPE = torch.bfloat16 if _PRECISION == "bf16" else torch.float16
MEAN = torch.tensor((0.5071, 0.4867, 0.4408), dtype=_DTYPE, device="cuda").view(1, 3, 1, 1)
STD = torch.tensor((0.2675, 0.2565, 0.2761), dtype=_DTYPE, device="cuda").view(1, 3, 1, 1)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def load_split(name):
    data = torch.load(Path("cifar100") / f"{name}.pt", map_location="cuda", weights_only=True)
    images = data["images"].to(_DTYPE).div_(255.0).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    labels = data["labels"].long()
    return images, labels


@torch.no_grad()
def normalize(x):
    return (x - MEAN) / STD


@torch.no_grad()
def random_crop_flip(x, pad=4):
    b, c, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    ys = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    xs = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    yy = torch.arange(h, device=x.device).view(1, 1, h, 1) + ys.view(b, 1, 1, 1)
    xx = torch.arange(w, device=x.device).view(1, 1, 1, w) + xs.view(b, 1, 1, 1)
    bb = torch.arange(b, device=x.device).view(b, 1, 1, 1)
    cc = torch.arange(c, device=x.device).view(1, c, 1, 1)
    out = x[bb, cc, yy, xx]
    flip = (torch.rand(b, device=x.device) < 0.5).view(b, 1, 1, 1)
    return torch.where(flip, out.flip(-1), out).contiguous(memory_format=torch.channels_last)


# Module-level ResNet-D flag (knob C100_RESNET_D). Read once at import; gates the
# downsampling skip-path construction inside Block. Default "0" -> exact baseline
# stride-2 1x1 conv skip.
_RESNET_D = os.getenv("C100_RESNET_D", "0") == "1"


class GhostBatchNorm(nn.BatchNorm2d):
    """Ghost BatchNorm (knob C100_GHOST_BN).

    TRAINING mode: normalizes over virtual sub-batches ("ghost" batches) of size
    `ghost_size` -- the incoming batch is split into chunks of that many samples
    and each chunk is normalized by its own batch statistics. Running stats are
    updated per chunk with the momentum divided by the number of chunks, so the
    aggregate per-forward running-stat update stays comparable to a single
    standard-BN forward. EVAL mode: byte-for-byte a standard nn.BatchNorm2d
    forward (uses the running stats), so the frozen validation path is unchanged.

    Subclasses nn.BatchNorm2d on purpose so isinstance(m, nn.BatchNorm2d) still
    matches (the BN-bias LR-split knob relies on that) and so the affine params,
    running buffers, and reset_parameters() are inherited unchanged.
    """

    def __init__(self, num_features, ghost_size, **kwargs):
        super().__init__(num_features, **kwargs)
        self.ghost_size = int(ghost_size)

    def forward(self, x):
        if not self.training:
            # Eval path: identical to standard BatchNorm2d (running stats).
            return super().forward(x)
        n = x.shape[0]
        g = min(self.ghost_size, n) if self.ghost_size > 0 else n
        chunks = torch.split(x, g, dim=0)  # last chunk may be smaller (remainder)
        n_splits = len(chunks)
        # Divide momentum by the split count so per-forward running-stat drift
        # stays comparable to standard BN. (None momentum -> cumulative average.)
        mom = self.momentum / n_splits if self.momentum is not None else None
        outs = [
            F.batch_norm(
                chunk, self.running_mean, self.running_var,
                self.weight, self.bias, True, mom, self.eps,
            )
            for chunk in chunks
        ]
        return torch.cat(outs, dim=0)


def convert_to_ghost_bn(module, ghost_size):
    """Recursively swap every nn.BatchNorm2d for a GhostBatchNorm carrying the
    same configuration. Called only when C100_GHOST_BN > 0."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d) and not isinstance(child, GhostBatchNorm):
            ghost = GhostBatchNorm(
                child.num_features, ghost_size,
                eps=child.eps, momentum=child.momentum,
                affine=child.affine, track_running_stats=child.track_running_stats,
            )
            setattr(module, name, ghost)
        else:
            convert_to_ghost_bn(child, ghost_size)


class Block(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels_out)
        if channels_in == channels_out and stride == 1:
            self.skip = nn.Identity()
        elif _RESNET_D:
            # ResNet-D downsampling skip (knob C100_RESNET_D): AvgPool for the
            # spatial downsample, then a stride-1 1x1 projection + BN. The main
            # path conv1 keeps its own stride. Off -> the baseline stride-2 1x1
            # conv skip in the else branch.
            skip_layers = []
            if stride != 1:
                skip_layers.append(nn.AvgPool2d(2, 2, ceil_mode=True))
            skip_layers.append(nn.Conv2d(channels_in, channels_out, 1, stride=1, bias=False))
            skip_layers.append(nn.BatchNorm2d(channels_out))
            self.skip = nn.Sequential(*skip_layers)
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels_out),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class SimpleResNet(nn.Module):
    def __init__(self, widths=(64, 128, 256), blocks=(2, 2, 2), num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        layers = []
        channels = widths[0]
        for stage, (width, n_blocks) in enumerate(zip(widths, blocks)):
            for i in range(n_blocks):
                stride = 2 if stage > 0 and i == 0 else 1
                layers.append(Block(channels, width, stride))
                channels = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(channels, num_classes, bias=False)
        self.to(memory_format=torch.channels_last)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


@torch.no_grad()
def zeropower_newton_schulz(g, steps=5):
    shape = g.shape
    x = g.reshape(g.shape[0], -1).float()
    if x.norm() == 0:
        return torch.zeros_like(g)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx = x @ x.T
        x = a * x + (b * xx + c * xx @ xx) @ x
    if transposed:
        x = x.T
    return x.reshape(shape).to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, nesterov=False, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        # nesterov / ns_steps are optimizer-wide knobs (not per-group). With the
        # baseline defaults (nesterov=False, ns_steps=5) the step below is
        # byte-identical to the frozen baseline.
        self.nesterov = nesterov
        self.ns_steps = ns_steps

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd:
                    p.mul_(1 - lr * wd)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(p.grad)
                if self.nesterov:
                    # Orthogonalize grad + momentum*buf (buffer already updated).
                    matrix = p.grad.add(buf, alpha=momentum)
                else:
                    # Plain heavy-ball: orthogonalize the momentum buffer itself.
                    matrix = buf
                update = zeropower_newton_schulz(matrix, steps=self.ns_steps)
                fan_out = update.shape[0]
                fan_in = max(1, update.numel() // fan_out)
                scale = math.sqrt(max(1.0, fan_out / fan_in))
                p.add_(update, alpha=-lr * scale)


def batches(images, labels, batch_size):
    order = torch.randperm(len(images), device=images.device)
    usable = len(order) // batch_size * batch_size
    order = order[:usable]
    for i in range(0, usable, batch_size):
        idx = order[i:i + batch_size]
        yield images[idx], labels[idx]


def schedule_multiplier(step, total_steps, schedule, warmup_frac, cooldown_frac):
    """LR-multiplier shape. With schedule='cosine' this returns exactly the
    baseline value 0.5*(1+cos(pi*progress))."""
    progress = step / total_steps
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "wsd":
        warmup_steps = warmup_frac * total_steps
        cooldown_steps = cooldown_frac * total_steps
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        cooldown_start = total_steps - cooldown_steps
        if cooldown_steps > 0 and step >= cooldown_start:
            return max(0.0, (total_steps - step) / cooldown_steps)
        return 1.0
    if schedule == "onecycle":
        warmup_steps = warmup_frac * total_steps
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        remain = total_steps - warmup_steps
        p = (step - warmup_steps) / remain if remain > 0 else 1.0
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def muon_momentum_at(step, total_steps, momentum, mom_warmup, mom_start):
    """Effective Muon momentum at a given step. With mom_warmup=0 (baseline)
    this is the constant `momentum`."""
    if mom_warmup > 0:
        warmup_steps = mom_warmup * total_steps
        frac = min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
        return mom_start + (momentum - mom_start) * frac
    return momentum


# Frozen validation gate. This function is outside the timed score and is not
# an optimization surface: no TTA, TTT, ensembling, confidence branches, BN
# adaptation, validation-label feedback, or benchmark-specific compilation games.
@torch.no_grad()
def evaluate(model, images, labels, batch_size=1000):
    model.eval()
    total = 0
    correct = 0
    for i in range(0, len(images), batch_size):
        x = normalize(images[i:i + batch_size])
        y = labels[i:i + batch_size]
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


def reset_model(model):
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def train_once(run_name, seed, model, train_images, train_labels, test_images, test_labels, epochs, batch_size, target):
    seed_all(seed)
    reset_model(model)
    muon_lr = float(os.getenv("C100_MUON_LR", "0.035"))
    bias_lr = float(os.getenv("C100_BIAS_LR", "0.02"))
    muon_momentum = float(os.getenv("C100_MUON_MOMENTUM", "0.95"))
    mom_warmup = float(os.getenv("C100_MUON_MOM_WARMUP", "0.0"))
    mom_start = float(os.getenv("C100_MUON_MOM_START", "0.85"))
    nesterov = os.getenv("C100_MUON_NESTEROV", "0") != "0"
    ns_steps = int(os.getenv("C100_NS_STEPS", "5"))
    label_smoothing = float(os.getenv("C100_LABEL_SMOOTHING", "0.05"))
    schedule = os.getenv("C100_SCHEDULE", "cosine")
    warmup_frac = float(os.getenv("C100_WARMUP_FRAC", "0.0"))
    cooldown_frac = float(os.getenv("C100_COOLDOWN_FRAC", "0.0"))
    crop_pad = int(os.getenv("C100_CROP_PAD", "4"))
    # --- Knob C100_HEAD_OPT / C100_HEAD_LR: which optimizer owns head.weight.
    head_opt = os.getenv("C100_HEAD_OPT", "muon")
    head_lr = float(os.getenv("C100_HEAD_LR", "0.001"))
    # --- Knob C100_BN_SHIFT_LR_MULT: separate LR factor for BN bias (beta) only.
    bn_shift_lr_mult = float(os.getenv("C100_BN_SHIFT_LR_MULT", "1.0"))
    # --- Knob C100_DIRAC_INIT: partial-identity init of body 3x3 convs.
    dirac_init = os.getenv("C100_DIRAC_INIT", "0") == "1"
    # --- Knob C100_EMA_DECAY: EMA of all params+buffers, evaluated after timer.
    ema_decay = float(os.getenv("C100_EMA_DECAY", "0.0"))
    ema_enabled = ema_decay > 0.0
    # --- Knob C100_LOOKAHEAD_K / _ALPHA: Lookahead over the Muon+SGD inner loop.
    lookahead_k = int(os.getenv("C100_LOOKAHEAD_K", "0"))
    lookahead_alpha = float(os.getenv("C100_LOOKAHEAD_ALPHA", "0.5"))
    lookahead_enabled = lookahead_k > 0
    # (EMA and Lookahead are meant to be mutually exclusive; if both are set, both
    # simply run -- the Lookahead sync happens first, then EMA reads the params.)
    # --- Knob C100_RESIZE_PX / _FRAC: progressive input resizing (train only).
    resize_px = int(os.getenv("C100_RESIZE_PX", "0"))
    resize_frac = float(os.getenv("C100_RESIZE_FRAC", "0.0"))
    resize_enabled = resize_px > 0 and resize_frac > 0.0

    # Partial-identity (Dirac) init of the two body 3x3 convs per Block. Applied
    # AFTER reset_model (so it overrides the default init) and BEFORE the timer.
    # Uses no RNG, so with the knob off the RNG stream is untouched.
    if dirac_init:
        for m in model.modules():
            if isinstance(m, Block):
                torch.nn.init.dirac_(m.conv1.weight)
                torch.nn.init.dirac_(m.conv2.weight)

    # Build parameter groups. head.weight is identified by identity (never shape).
    head_param = model.head.weight
    head_in_muon = head_opt not in ("sgd", "adam")
    # BN bias (beta) params get their own SGD group ONLY when the shift mult is
    # non-trivial; with mult == 1.0 we keep the single baseline group, so behavior
    # is byte-for-byte identical (splitting into two equal-LR/equal-momentum SGD
    # groups is numerically identical anyway, since SGD updates each param
    # independently with per-param momentum state).
    bn_bias_ids = set()
    if bn_shift_lr_mult != 1.0:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) and m.bias is not None:
                bn_bias_ids.add(id(m.bias))
    muon_params = []
    sgd_other_params = []
    bn_bias_params = []
    for p in model.parameters():
        if p is head_param and not head_in_muon:
            if head_opt == "sgd":
                sgd_other_params.append(p)
            # head_opt == "adam" -> optimized by its own Adam group (below).
            continue
        if p.ndim >= 2:
            muon_params.append(p)
        elif id(p) in bn_bias_ids:
            bn_bias_params.append(p)
        else:
            sgd_other_params.append(p)

    muon = Muon(muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=2e-4, nesterov=nesterov, ns_steps=ns_steps)
    sgd_groups = [{"params": sgd_other_params}]
    if bn_bias_params:
        sgd_groups.append({"params": bn_bias_params, "lr": bias_lr * bn_shift_lr_mult})
    sgd = torch.optim.SGD(sgd_groups, lr=bias_lr, momentum=0.9, nesterov=True)
    sgd_base_lrs = [g["lr"] for g in sgd.param_groups]
    head_adam = torch.optim.Adam([head_param], lr=head_lr) if head_opt == "adam" else None

    steps_per_epoch = len(train_images) // batch_size
    total_steps = max(1, int(math.ceil(epochs * steps_per_epoch)))
    resize_until = resize_frac * total_steps
    step = 0

    # EMA / Lookahead shadow state is captured here -- AFTER reset_model + Dirac
    # init, BEFORE the timer -- so it is pure setup, not timed work.
    if ema_enabled:
        ema_targets = list(model.parameters()) + list(model.buffers())
        ema_is_float = [t.is_floating_point() for t in ema_targets]
        # fp32 shadow for float tensors so the (1-decay) increment does not
        # underflow in bf16/fp16 over a short (~hundreds of steps) run; non-float
        # buffers (num_batches_tracked) are cloned as-is.
        ema_shadow = [t.detach().float() if f else t.detach().clone() for t, f in zip(ema_targets, ema_is_float)]
    if lookahead_enabled:
        lookahead_params = list(model.parameters())
        lookahead_slow = [p.detach().clone() for p in lookahead_params]

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    # Timed training begins here. Only architecture, optimizer, and training
    # hyperparameters inside this region are valid speedrun optimization surfaces.
    starter.record()
    model.train()
    while step < total_steps:
        for x, y in batches(train_images, train_labels, batch_size):
            x = normalize(random_crop_flip(x, pad=crop_pad))
            # Knob C100_RESIZE_*: bilinearly downscale the (already cropped +
            # normalized) TRAIN batch for the first resize_frac of steps. Inside
            # the timed region. Eval is never resized, so the frozen 32x32 val
            # path stays legal.
            if resize_enabled and step < resize_until:
                x = F.interpolate(x, size=(resize_px, resize_px), mode="bilinear", align_corners=False)
                x = x.contiguous(memory_format=torch.channels_last)
            logits = model(x)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=label_smoothing)
            loss.backward()
            lr_mult = schedule_multiplier(step, total_steps, schedule, warmup_frac, cooldown_frac)
            mom = muon_momentum_at(step, total_steps, muon_momentum, mom_warmup, mom_start)
            muon.param_groups[0]["lr"] = muon_lr * lr_mult
            muon.param_groups[0]["momentum"] = mom
            # Scale every SGD group (baseline group 0, plus optional BN-bias group)
            # by the scheduled multiplier off its own base LR.
            for gi, g in enumerate(sgd.param_groups):
                g["lr"] = sgd_base_lrs[gi] * lr_mult
            if head_adam is not None:
                head_adam.param_groups[0]["lr"] = head_lr * lr_mult
            muon.step(); sgd.step()
            if head_adam is not None:
                head_adam.step()
            muon.zero_grad(set_to_none=True); sgd.zero_grad(set_to_none=True)
            if head_adam is not None:
                head_adam.zero_grad(set_to_none=True)
            step += 1
            # Lookahead slow-weight sync (inside the timed region). Every k steps:
            # slow += alpha*(fast - slow); fast <- slow. Params only, not buffers.
            if lookahead_enabled and step % lookahead_k == 0:
                with torch.no_grad():
                    for slow_t, fast_t in zip(lookahead_slow, lookahead_params):
                        slow_t.add_(fast_t.detach() - slow_t, alpha=lookahead_alpha)
                        fast_t.copy_(slow_t)
            # EMA update (inside the timed region), after any Lookahead sync so the
            # shadow tracks the post-sync params. All params AND buffers tracked;
            # non-float buffers (num_batches_tracked) are copied, not averaged.
            if ema_enabled:
                with torch.no_grad():
                    for shadow_t, cur_t, is_f in zip(ema_shadow, ema_targets, ema_is_float):
                        if is_f:
                            shadow_t.mul_(ema_decay).add_(cur_t.detach().float(), alpha=1.0 - ema_decay)
                        else:
                            shadow_t.copy_(cur_t)
            if step >= total_steps:
                break
    # Timed training ends here. Validation remains an untimed correctness gate.
    ender.record(); torch.cuda.synchronize()
    time_seconds = starter.elapsed_time(ender) * 1e-3
    # Knob C100_EMA_DECAY (eval side, AFTER the timer): swap the EMA shadow into
    # the model so the reported accuracies are the EMA model's -- preregistered
    # and unconditional, never a val-gated choice. Raw weights are saved first and
    # restored after eval so nothing downstream breaks.
    if ema_enabled:
        with torch.no_grad():
            raw_state = [t.detach().clone() for t in ema_targets]
            for shadow_t, cur_t in zip(ema_shadow, ema_targets):
                cur_t.copy_(shadow_t)
    val_acc = evaluate(model, test_images, test_labels)
    train_acc = evaluate(model, train_images[:10000], train_labels[:10000])
    if ema_enabled:
        with torch.no_grad():
            for raw_t, cur_t in zip(raw_state, ema_targets):
                cur_t.copy_(raw_t)
    hit = float(val_acc >= target)
    print(f"|  {str(run_name).rjust(6)}  |   eval  |     {train_acc:0.4f}  |   {val_acc:0.4f}  |       {hit:0.4f}  |      {time_seconds:0.4f}  |", flush=True)
    return val_acc, time_seconds


def main():
    runs = int(os.getenv("C100_RUNS", "30"))
    epochs = float(os.getenv("C100_EPOCHS", "16"))
    batch_size = int(os.getenv("C100_BATCH", "1024"))
    target = float(os.getenv("C100_TARGET", "0.70"))
    seed_base = int(os.getenv("C100_SEED_BASE", "880000"))
    sleep_cycles = int(os.getenv("C100_SLEEP_CYCLES", "1000000000"))
    widths = tuple(int(w) for w in os.getenv("C100_WIDTHS", "64,128,256").split(","))
    blocks = tuple(int(x) for x in os.getenv("C100_BLOCKS", "2,2,2").split(","))
    train_images, train_labels = load_split("train")
    test_images, test_labels = load_split("test")
    compile_enabled = os.getenv("C100_COMPILE", "1") != "0"
    compile_mode = os.getenv("C100_COMPILE_MODE", "default")
    ghost_bn = int(os.getenv("C100_GHOST_BN", "0"))
    model = SimpleResNet(widths=widths, blocks=blocks)
    # Knob C100_GHOST_BN: swap standard BN for GhostBatchNorm before moving to
    # device/dtype and before compile. ghost_bn == 0 -> untouched standard BN.
    if ghost_bn > 0:
        convert_to_ghost_bn(model, ghost_bn)
    model = model.cuda().to(_DTYPE).to(memory_format=torch.channels_last)
    # Compile is infrastructure, not a record surface. It is paid in warmup and
    # must not be tuned as a benchmark trick; use it only to make the fixed
    # training implementation run normally on the target stack.
    if compile_enabled:
        if compile_mode in ("", "default", "none"):
            model.compile()
        else:
            model.compile(mode=compile_mode)
    print(f"config model=simple_resnet_muon runs={runs} epochs={epochs} batch={batch_size} target={target} compile={int(compile_enabled)} compile_mode={compile_mode if compile_enabled else off} no_tta=1")
    # Extra self-documenting line (NOT greppable by analyze_cifar100.py). Records
    # every experimental knob so a run log fully identifies its recipe.
    print(
        f"exp_config precision={_PRECISION}"
        f" ls={float(os.getenv('C100_LABEL_SMOOTHING', '0.05'))}"
        f" schedule={os.getenv('C100_SCHEDULE', 'cosine')}"
        f" warmup={float(os.getenv('C100_WARMUP_FRAC', '0.0'))}"
        f" cooldown={float(os.getenv('C100_COOLDOWN_FRAC', '0.0'))}"
        f" nesterov={int(os.getenv('C100_MUON_NESTEROV', '0') != '0')}"
        f" ns_steps={int(os.getenv('C100_NS_STEPS', '5'))}"
        f" muon_mom={float(os.getenv('C100_MUON_MOMENTUM', '0.95'))}"
        f" mom_warmup={float(os.getenv('C100_MUON_MOM_WARMUP', '0.0'))}"
        f" crop_pad={int(os.getenv('C100_CROP_PAD', '4'))}"
        f" head_opt={os.getenv('C100_HEAD_OPT', 'muon')}"
        f" head_lr={float(os.getenv('C100_HEAD_LR', '0.001'))}"
        f" bn_shift_lr_mult={float(os.getenv('C100_BN_SHIFT_LR_MULT', '1.0'))}"
        f" ema_decay={float(os.getenv('C100_EMA_DECAY', '0.0'))}"
        f" lookahead_k={int(os.getenv('C100_LOOKAHEAD_K', '0'))}"
        f" lookahead_alpha={float(os.getenv('C100_LOOKAHEAD_ALPHA', '0.5'))}"
        f" ghost_bn={int(os.getenv('C100_GHOST_BN', '0'))}"
        f" dirac_init={int(os.getenv('C100_DIRAC_INIT', '0') == '1')}"
        f" resnet_d={int(os.getenv('C100_RESNET_D', '0') == '1')}"
        f" resize_px={int(os.getenv('C100_RESIZE_PX', '0'))}"
        f" resize_frac={float(os.getenv('C100_RESIZE_FRAC', '0.0'))}"
        f" widths={','.join(str(w) for w in widths)}"
        f" blocks={','.join(str(x) for x in blocks)}"
    )
    print("---------------------------------------------------------------------------------")
    print("|  run     |  epoch  |  train_acc  |  val_acc  |  target_hit   |  time_seconds  |")
    print("---------------------------------------------------------------------------------")
    train_once("warmup", seed_base - 1, model, train_images, train_labels, test_images, test_labels, min(1.0, epochs), batch_size, target)
    vals, times = [], []
    for run in range(runs):
        torch.cuda.empty_cache(); torch.cuda.synchronize()
        if sleep_cycles > 0:
            torch.cuda._sleep(sleep_cycles)
        val, sec = train_once(run + 1, seed_base + run, model, train_images, train_labels, test_images, test_labels, epochs, batch_size, target)
        vals.append(val); times.append(sec)
        print(f"Mean val accuracy after {run + 1} runs: {sum(vals) / len(vals):.6f} | Mean time: {sum(times) / len(times):.6f}s", end="\r", flush=True)
    print()
    v = torch.tensor(vals); t = torch.tensor(times)
    print("Val accuracies: Mean: %.6f    Std: %.6f    Min: %.6f    Max: %.6f" % (v.mean(), v.std(unbiased=False), v.min(), v.max()))
    print("Times (s):      Mean: %.6f    Std: %.6f    Min: %.6f    Max: %.6f" % (t.mean(), t.std(unbiased=False), t.min(), t.max()))
    print("Target %.4f hit count: %d/%d" % (target, int((v >= target).sum().item()), runs))


if __name__ == "__main__":
    main()
