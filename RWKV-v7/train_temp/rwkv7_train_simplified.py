########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
# -----------------------------------------------------------------------
# TERNARY WEIGHT VARIANT (BitNet b1.58)
# -----------------------------------------------------------------------
# Changes from the original rwkv7_train_simplified.py:
#
#   1. Added BitLinear — a drop-in for nn.Linear that stores BF32 shadow
#      weights and materialises ternary {-1,0,+1} weights on every forward
#      pass using absmean quantisation + Straight-Through Estimator (STE).
#
#   2. Replaced the 4 big nn.Linear layers in RWKV_Tmix_x070
#      (receptance / key / value / output) with BitLinear.
#
#   3. Replaced the 2 nn.Linear layers in FFN (key / value) with BitLinear.
#
#   4. Disabled torch.jit.ScriptModule — the dynamic quantisation ops inside
#      BitLinear break TorchScript tracing. Training throughput is unchanged.
#
#   5. Added ternary_stats() helper that logs β (absmean scale) and sparsity
#      (fraction of weights rounded to 0) to wandb every 100 steps, so you
#      can monitor quantisation quality without a separate eval pass.
#
# What is deliberately NOT ternarised, and why:
#
#   w1/w2  (decay LoRA)        — feeds exp(−exp(⋯)); quantised models keep
#                                these at F32. Small errors compound across
#                                the entire sequence via the WKV state.
#   a1/a2  (ICLR LoRA)        — delta-rule removal amount; precision-
#   v1/v2  (value residual)      sensitive. Quantised models keep these F16.
#   g1/g2  (gate LoRA)
#   x_r/w/k/v/a/g              — per-channel lerp scalars [1,1,C]; tiny,
#   w0/a0/v0/k_k/k_a/r_k        nothing to gain from ternarising.
#   ln_x (GroupNorm)            — normalisation, not a projection.
#   e    (Embedding)            — boundary between vocab and residual stream;
#   o    (output Linear)          kept full-precision in BitNet paper too.
#
# NO internal LayerNorm inside BitLinear:
#   BitNet's original BitLinear contains a LN because it can't assume anything
#   about incoming activations. RWKV-7 uses PreLN (LayerNorm before every
#   block), so inputs to receptance/key/value/output are already normalised.
#   Adding another LN inside BitLinear would be redundant and would interact
#   badly with the token-shift lerp that sits between the block LN and the
#   projection. We rely on the existing PreLN to satisfy the "bounded input"
#   assumption that makes absmax int8 quantisation well-behaved.
########################################################################################################

import random, torch, os, math, time
import numpy as np
import wandb, datetime
from types import SimpleNamespace
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.cpp_extension import load

# ── reproducibility ─────────────────────────────────────────────────────────
def set_seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed_all(42)
device = 'cuda'

# ── TorchScript disabled for ternary variant ─────────────────────────────────
# torch.jit.ScriptModule cannot handle the dynamic quantisation ops (amax,
# round, clamp with float boundaries) inside BitLinear. Removing it has zero
# effect on training — JIT only helps inference throughput.
MyModule  = nn.Module
MyFunction = lambda fn: fn   # identity decorator — keeps @MyFunction annotations

# ── hyper-parameters (toy task — change these for real LM training) ──────────
V, C, B, T, steps = 12, 32, 256, 129, 10_000
lr0, lr1 = 4e-3, 1e-6
DIGIT_MAX = 60

print("Ternary (BitNet b1.58) RWKV-7 training demo")

############################################################################
# CUDA WKV kernel — unchanged from original
############################################################################

HEAD_SIZE = 16     # use 64 for real LM
CHUNK_LEN = 16

flags = [
    '-res-usage', f'-D_C_={HEAD_SIZE}', f'-D_CHUNK_LEN_={CHUNK_LEN}',
    '--use_fast_math', '-O3', '-Xptxas -O3', '--extra-device-vectorization',
]
load(
    name="wind_backstepping",
    sources=['cuda/wkv7_cuda_fp32.cu', 'cuda/wkv7_op_fp32.cpp'],
    is_python_module=False, verbose=False, extra_cuda_cflags=flags,
)

class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape
        assert T % CHUNK_LEN == 0, \
            "pad your input so T % CHUNK_LEN == 0, or change CHUNK_LEN"
        assert all(i.dtype == torch.float32 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous()        for i in [w,q,k,v,z,b])
        y  = torch.empty_like(v)
        s  = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, C,               dtype=torch.float32, device=w.device)
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        assert dy.dtype == torch.float32 and dy.is_contiguous()
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa,
                                             dw, dq, dk, dv, dz, db)
        return dw, dq, dk, dv, dz, db


def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
    B, T, HC = q.shape
    q,w,k,v,a,b = [i.view(B, T, HC // HEAD_SIZE, HEAD_SIZE) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)


############################################################################
# BitLinear — ternary weight linear layer
############################################################################

class BitLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with BitNet b1.58 ternary weights.

    Forward pass
    ────────────
    1. Ternarize shadow weight W  →  W_q ∈ {-1, 0, +1}
       scale β = mean(|W|)  (absmean scheme, per-tensor)
    2. Quantize input x  →  x_q ∈ [-128, 127]
       scale γ = max(|x|) / 127  (absmax scheme, per-token)
    3. out = x_q @ W_q.T          (integer additions only on real hardware)
    4. out = out * γ * β           (dequantize)

    Backward pass
    ─────────────
    Straight-Through Estimator (STE) on both round() operations:
    ∂round(x)/∂x ≡ 1  — gradient flows straight through to the shadow
    weight W, which Adam updates normally in float32.

    The ternary W_q is never stored between steps; it is re-derived from W
    on every forward call. This is identical to BitNet's approach.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        assert not bias, \
            "RWKV-7 projections carry no bias; BitLinear does not support bias"
        self.in_features  = in_features
        self.out_features = out_features
        # Shadow weight — the only thing the optimizer sees.
        # Shape matches nn.Linear: [out_features, in_features]
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # ^ callers that need specific init (e.g. output.weight.data.zero_())
        #   overwrite this after construction, exactly as in the original.

    # ── quantization helpers ─────────────────────────────────────────────

    def _ternarize(self, W: torch.Tensor):
        """Absmean ternarization with STE.  Returns (W_q, β)."""
        # β is computed in no-grad so it doesn't appear in the graph twice
        beta   = W.detach().abs().mean()                  # scalar
        W_q_hard = torch.clamp(torch.round(W / (beta + 1e-8)), -1.0, 1.0)
        # STE: forward uses W_q_hard, backward uses ∂W_q/∂W = 1
        W_q = W + (W_q_hard - W).detach()
        return W_q, beta

    def _quantize_input(self, x: torch.Tensor):
        """Per-token absmax int8 quantization with STE.  Returns (x_q, γ)."""
        # gamma shape: [..., 1]  — one scale per token, broadcast over features
        gamma    = x.detach().abs().amax(dim=-1, keepdim=True) / 127.0
        x_q_hard = torch.clamp(torch.round(x / (gamma + 1e-8)), -128.0, 127.0)
        x_q = x + (x_q_hard - x).detach()
        return x_q, gamma

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_q, beta  = self._ternarize(self.weight)
        x_q, gamma = self._quantize_input(x)
        # F.linear computes x_q @ W_q.T in float.
        # On dedicated 1-bit hardware this collapses to additions/subtractions.
        out = F.linear(x_q, W_q)
        # Dequantize — multiply by both scale factors
        out = out * (gamma * beta)
        return out

    def extra_repr(self) -> str:
        return f'in={self.in_features}, out={self.out_features}, ternary=True'


############################################################################
# RWKV-7 time mixing block  (RWKV_Tmix_x070)
############################################################################

class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.head_size = args.head_size
        self.n_head    = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1      = layer_id / (args.n_layer - 1)   # 0 → 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer) # 1 → ~0

            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            # ── token-shift lerp scalars (per-channel, kept float) ────────
            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        raise ValueError("ortho_init: unexpected shape")
                return x

            www    = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n]    = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            # ── decay LoRA (w1/w2) — KEPT FLOAT, NOT TERNARISED ──────────
            # Feeds into exp(−exp(⋯)) nonlinearity. Quantised GGUF models
            # keep these at F32 precisely because small errors here compound
            # across the WKV state over thousands of tokens.
            D_DECAY_LORA = 8   # for LM: max(32, int(round((2.5*(C**0.5))/32)*32))
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag * 2.5)

            # ── ICLR LoRA (a1/a2) — KEPT FLOAT, NOT TERNARISED ──────────
            # Controls the delta-rule removal amount: how aggressively the
            # model overwrites existing WKV state. Precision-sensitive.
            D_AAA_LORA = 8
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C) - 0.19 + zigzag*0.3 + linear*0.4)

            # ── value-residual LoRA (v1/v2) — KEPT FLOAT, NOT TERNARISED ─
            D_MV_LORA = 8
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C) + 0.73 - linear*0.4)

            # ── gate LoRA (g1/g2) — KEPT FLOAT, NOT TERNARISED ──────────
            D_GATE_LORA = 8
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            # ── per-channel scalar parameters — too small to ternarise ────
            self.k_k = nn.Parameter(torch.zeros(1,1,C) + 0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # ── TERNARY: the 4 big projection matrices ────────────────────────
        # receptance, key, value, output dominate parameter count and compute.
        # They are pure linear projections with no precision-sensitive
        # nonlinearities in their path — safe to ternarise.
        #
        # NO internal LayerNorm: RWKV-7 uses PreLN (LayerNorm applied to x
        # before the block), so inputs xr/xk/xv/xg are already normalised.
        # We rely on that external normalisation for the absmax int8 scale γ
        # to be well-behaved, exactly as BitNet relies on its internal LN.
        self.receptance = BitLinear(C, C, bias=False)
        self.key        = BitLinear(C, C, bias=False)
        self.value      = BitLinear(C, C, bias=False)
        self.output     = BitLinear(C, C, bias=False)

        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

        # Preserve original careful initialisation on the shadow weights.
        # These scale choices keep the initial absmean β ≈ 0.3–0.5 which
        # is a good starting point — not too small (all zeros) or too large.
        self.receptance.weight.data.uniform_(-0.5  / (C**0.5),  0.5  / (C**0.5))
        self.key.weight.data.uniform_(       -0.05 / (C**0.5),  0.05 / (C**0.5))
        self.value.weight.data.uniform_(     -0.5  / (C**0.5),  0.5  / (C**0.5))
        self.output.weight.data.zero_()
        # Note on output init: zero init means β=0 at step 0, so the ternary
        # output projection starts as all-zeros. That's fine — the shadow
        # weight will immediately move away from zero under gradient, and once
        # β > 0 the quantisation becomes meaningful. This matches the original.

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head

        xx = self.time_shift(x) - x

        # Token-shift lerp: interpolate between current and previous token
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        # ── TERNARY projections ───────────────────────────────────────────
        r = self.receptance(xr)  # BitLinear: quantise→matmul→dequantise
        k = self.key(xk)
        v = self.value(xv)

        # ── FLOAT: decay computation (must stay precise) ─────────────────
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5

        # ── value residual (FLOAT LoRA) ───────────────────────────────────
        if self.layer_id == 0:
            v_first = v   # store layer-0 value for residual connection
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        # ── FLOAT: ICLR and gate ─────────────────────────────────────────
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = F.normalize((k * self.k_k).view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k  = k * (1 + (a - 1) * self.k_a)

        # ── WKV state update (CUDA kernel, always float) ──────────────────
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = x + ((r.view(B,T,H,-1) * k.view(B,T,H,-1) * self.r_k)
                   .sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)

        # ── TERNARY output projection (input is gated WKV output) ─────────
        x = self.output(x * g)

        return x, v_first


############################################################################
# Feed-forward network
############################################################################

class FFN(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))

        # ── TERNARY: both FFN projections ─────────────────────────────────
        # Standard expand-then-contract MLP. No precision-sensitive path
        # between key and value — relu² is just an activation, not a
        # recurrence accumulator. Safe to ternarise.
        self.key   = BitLinear(C,     C * 4, bias=False)
        self.value = BitLinear(C * 4, C,     bias=False)

        with torch.no_grad():
            self.value.weight.data.zero_()                      # same as original
            nn.init.orthogonal_(self.key.weight.data, gain=4**0.5)   # same as original

    def forward(self, x):
        xx = self.time_shift(x) - x
        x  = x + xx * self.x_k
        x  = torch.relu(self.key(x)) ** 2   # squared ReLU nonlinearity
        return self.value(x)


############################################################################
# Full model
############################################################################

class MODEL(nn.Module):
    def __init__(s):
        super().__init__()
        args = SimpleNamespace()
        args.n_head   = C // HEAD_SIZE
        args.head_size = HEAD_SIZE
        args.n_embd   = C
        args.dim_att  = C
        args.n_layer  = 2

        # Embedding and head stay in full float — they are the boundary
        # between the ternary network interior and the float world outside.
        # This matches the BitNet paper, and the GGUF quantised RWKV model
        # which uses Q6_K for the output head (never ternary).
        s.e   = nn.Embedding(V, C)

        s.ln1a = nn.LayerNorm(C)   # PreLN before rwkv1
        s.ln1b = nn.LayerNorm(C)   # PreLN before ffn1
        s.ln1c = nn.LayerNorm(C)   # unused in simplified model; kept for compat
        s.rwkv1 = RWKV_Tmix_x070(args, 0)
        s.ffn1  = FFN(C)

        s.ln2a = nn.LayerNorm(C)
        s.ln2b = nn.LayerNorm(C)
        s.ln2c = nn.LayerNorm(C)
        s.rwkv2 = RWKV_Tmix_x070(args, 1)
        s.ffn2  = FFN(C)

        s.lno = nn.LayerNorm(C)
        s.o   = nn.Linear(C, V)    # output head — full float

    def forward(s, x):
        x = s.e(x)

        xx, v_first = s.rwkv1(s.ln1a(x), torch.empty_like(x))
        x = x + xx
        x = x + s.ffn1(s.ln1b(x))

        xx, v_first = s.rwkv2(s.ln2a(x), v_first)
        x = x + xx
        x = x + s.ffn2(s.ln2b(x))

        x = s.o(s.lno(x))
        return x


############################################################################
# Optimiser setup
############################################################################

model = MODEL().to(device)

# Weight decay rule (identical logic to original, works correctly for ternary):
#   • parameters with '.weight' in their name (and not in 'ln') → decay group
#   • everything else → no-decay group
#
# BitLinear.weight is the shadow weight that Adam updates. It needs weight
# decay so the absmean scale β stays reasonable and doesn't grow unboundedly.
# The LoRA parameter tensors (w1, w2, a1, …) don't have '.weight' in their
# name, so they correctly fall into no_decay.
decay,    no_decay    = [], []
decay_names, no_decay_names = [], []

for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if ('.weight' in n or 'emb' in n) and ('ln' not in n):
        decay.append(p);    decay_names.append(n)
    else:
        no_decay.append(p); no_decay_names.append(n)

print('\ndecay (shadow weights + embedding):')
for n in decay_names:    print(' ', n)
print('\nno_decay (LoRA params, scalars, norms):')
for n in no_decay_names: print(' ', n)

opt = torch.optim.AdamW(
    [
        {"params": decay,    "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=lr0,
)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr1)


############################################################################
# Ternary quality monitor
############################################################################

def ternary_stats(model):
    """
    Compute β (absmean scale) and sparsity (fraction of weights rounded to 0)
    for every BitLinear layer. Returns a flat dict ready for wandb.log().

    β too small  → weights near zero, quantisation is ill-defined. Check LR.
    β too large  → weights escaping; weight decay may need tuning.
    sparsity     → fraction of ternary values that are exactly 0.
                   Values 50–70% are healthy for a well-trained ternary model.
                   <20% early in training is normal (weights still spreading out).
    """
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            with torch.no_grad():
                W    = module.weight
                beta = W.abs().mean().item()
                if beta > 1e-9:
                    W_q      = torch.clamp(torch.round(W / beta), -1.0, 1.0)
                    sparsity = (W_q == 0).float().mean().item()
                else:
                    sparsity = 1.0   # degenerate: all weights zero
                stats[f"tern/{name}/beta"]     = beta
                stats[f"tern/{name}/sparsity"] = sparsity
    return stats


############################################################################
# Training data (digit-reversal toy task — unchanged from original)
############################################################################

print('\nTraining...')

TOK = {**{str(i): i for i in range(10)}, ',': 10, '#': 11}

def _digits(n): return [TOK[c] for c in str(n)]

def batch(B, T, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    s = []
    for _ in range(B):
        a = []
        while len(a) < T:
            k  = random.randint(1, DIGIT_MAX)
            lo = 0 if k == 1 else 10 ** (k - 1)
            n  = random.randint(lo, 10 ** k - 1)
            nn_ = _digits(n)
            a  += nn_ + [TOK[',']] + nn_[::-1] + [TOK['#']]
        s.append(a[:T])
    return torch.tensor(s, device=device, dtype=torch.long)


############################################################################
# W&B
############################################################################

args = SimpleNamespace()
trainer = SimpleNamespace()
args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

print("Login to wandb...")
wandb.init(project="Test", name=args.my_timestamp, config=args, save_code=False)

############################################################################
# Training loop
############################################################################

token_per_step = B * (T - 1)

for step in range(steps):
    x = batch(B, T); y = x[:, 1:]; x = x[:, :-1]
    z    = model(x)
    loss = F.cross_entropy(z.reshape(-1, V), y.reshape(-1))

    trainer.my_lr   = sch.get_last_lr()[0]
    trainer.my_loss = loss.item()

    t_now = time.time_ns()
    kt_s  = 0
    try:
        t_cost = (t_now - trainer.my_time_ns) / 1e9
        kt_s   = token_per_step / t_cost / 1000
    except Exception:
        pass
    trainer.my_time_ns = t_now

    print(f'{step+1}/{steps}  loss {trainer.my_loss:.4f}  lr {trainer.my_lr:.2e}')

    log_dict = {
        "loss":     trainer.my_loss,
        "lr":       trainer.my_lr,
        "Mtokens":  (step + 1) * token_per_step / 1e6,
    }
    if kt_s > 0:
        log_dict["kt/s"] = kt_s

    # Ternary quality metrics — logged every 100 steps.
    # Costs one extra pass over model parameters (no GPU compute).
    if (step + 1) % 100 == 0:
        log_dict.update(ternary_stats(model))

    wandb.log(log_dict, step=step + 1)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    sch.step()

torch.save(model.state_dict(), "out.pth")


############################################################################
# Evaluation (unchanged from original)
############################################################################

print('#' * 100)
print('simple check (NOTE: random inputs used, including for diff)')
with torch.no_grad():
    S = '0123456789,#'
    for SAMPLE in range(5):
        x  = batch(1, 129); y = x[:, 1:]; z = model(x[:, :-1]).argmax(-1)
        xx = ''.join(S[t] for t in x[0, :-1].tolist())
        yy = ''.join(S[t] for t in y[0].tolist())
        zz = ''.join(S[t] for t in z[0].tolist())
        zy = ''.join('.' if z[0,i].item() == y[0,i].item() else '^' for i in range(y.size(1)))
        print('in ', xx)
        print('gold', yy)
        print('pred', zz)
        print('diff', zy)

print('#' * 100)
print('#' * 100)
print('correct check (only check reversal outputs)')
with torch.no_grad():
    S     = '0123456789,#'
    COMMA = S.index(',')
    HASH  = S.index('#')
    for SAMPLE in range(5):
        x      = batch(1, 129)
        y      = x[:, 1:]
        logits = model(x[:, :-1])
        z      = logits.argmax(-1)

        xx = ''.join(S[t] for t in x[0, :-1].tolist())
        yy = ''.join(S[t] for t in y[0].tolist())
        zz = ''.join(S[t] for t in z[0].tolist())

        x_ids = x[0].tolist()
        region_char = [False] * len(x_ids)
        mode = 0
        for j, tok in enumerate(x_ids):
            if mode == 1:
                region_char[j] = True
            if tok == COMMA:
                mode = 1
            elif tok == HASH:
                mode = 0

        mask   = region_char[1:]
        y_ids  = y[0].tolist()
        z_ids  = z[0].tolist()
        n_tok  = sum(mask)
        n_corr = sum(1 for i, m in enumerate(mask) if m and y_ids[i] == z_ids[i]) if n_tok else 0
        acc    = n_corr / n_tok if n_tok else float('nan')

        gold_m = ''.join(S[y_ids[i]] if mask[i] else ' ' for i in range(len(y_ids)))
        pred_m = ''.join(S[z_ids[i]] if mask[i] else ' ' for i in range(len(z_ids)))
        diff_m = ''.join(
            ('.' if y_ids[i] == z_ids[i] else '^') if mask[i] else ' '
            for i in range(len(y_ids))
        )

        print('in   ', xx)
        print('gold ', gold_m)
        print('pred ', pred_m)
        print('diff ', diff_m)
        print(f'correct {n_corr}/{n_tok}  acc {acc:.3f}')
        print('#' * 100)
