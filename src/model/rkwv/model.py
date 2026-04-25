##############################################################################
# The RWKV File Model
##############################################################################
import os, math, torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from .miss import MiSSLinear

##############################################################################
# 0. Global settings and loading the WKV6 CUDA kernel
##############################################################################
HEAD_SIZE = 64
HEAD_DIVISOR = float(os.environ.get("RWKV_HEAD_SIZE_DIVISOR", 8))
CTXLEN = 2048

flags = [
    "-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
    "--extra-device-vectorization",
    f"-D_N_={HEAD_SIZE}", f"-D_T_={CTXLEN}"
]
wkv6 = load(
    name="wkv6",
    sources=[f"../src/model/cuda/wkv6_op.cpp", f"../src/model/cuda/wkv6_cuda.cu"],
    verbose=False, extra_cuda_cflags=flags, is_python_module=False
)


def apply_miss_to_model(model, miss_shard_size=16):

    for block in model.blocks:
        att = block.att   # RWKV_Tmix_x060
        ffn = block.ffn   # RWKV_CMix_x060

        att.receptance = MiSSLinear(att.receptance, shard_size=miss_shard_size)
        att.key        = MiSSLinear(att.key,        shard_size=miss_shard_size)
        att.value      = MiSSLinear(att.value,      shard_size=miss_shard_size)
        att.output     = MiSSLinear(att.output,     shard_size=miss_shard_size)
        att.gate       = MiSSLinear(att.gate,       shard_size=miss_shard_size)
        ffn.key        = MiSSLinear(ffn.key,        shard_size=miss_shard_size)
        ffn.value      = MiSSLinear(ffn.value,      shard_size=miss_shard_size)
        ffn.receptance = MiSSLinear(ffn.receptance, shard_size=miss_shard_size)

    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'shard' in name:
            param.requires_grad = True

class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert all(t.dtype == torch.bfloat16 for t in (r, k, v, w, u))
            assert HEAD_SIZE == C // H
            ctx.B, ctx.T, ctx.C, ctx.H = B, T, C, H
            ctx.save_for_backward(r, k, v, w, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16)
            torch.ops.wkv6.forward(B, T, C, H, r, k, v, w, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B, T, C, H = ctx.B, ctx.T, ctx.C, ctx.H
            r, k, v, w, u = ctx.saved_tensors
            gr = torch.empty_like(r)
            gk = torch.empty_like(k)
            gv = torch.empty_like(v)
            gw = torch.empty_like(w)
            gu = torch.empty((B, C), device=gy.device, dtype=torch.bfloat16)
            torch.ops.wkv6.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C // H)
            return (None, None, None, None, gr, gk, gv, gw, gu)


def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)


##############################################################################
# 1. Time-Mixing Block (attention) RWKV_Tmix_x060
##############################################################################
class RWKV_Tmix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.n_head = args.dim_att // args.head_size_a
        self.head_size = args.head_size_a
        C = args.n_embd
        H = self.n_head

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)
            ddd = torch.linspace(0, 1, C).unsqueeze(0).unsqueeze(0)

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        D_MIX_LORA = 32
        self.time_maa_w1 = nn.Parameter(torch.zeros(C, D_MIX_LORA * 5))
        self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, C).uniform_(-0.01, 0.01))

        decay_speed = torch.ones(args.dim_att)
        for n in range(args.dim_att):
            decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
        self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, args.dim_att))

        D_DECAY_LORA = 64
        self.time_decay_w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
        self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

        tmp = torch.zeros(args.dim_att)
        for n in range(args.dim_att):
            zigzag = ((n + 1) % 3 - 1) * 0.1
            tmp[n] = ratio_0_to_1 * (1 - n / (args.dim_att - 1)) + zigzag
        self.time_faaaa = nn.Parameter(tmp.reshape(H, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.receptance = nn.Linear(C, args.dim_att, bias=False)
        self.key        = nn.Linear(C, args.dim_att, bias=False)
        self.value      = nn.Linear(C, args.dim_att, bias=False)
        self.output     = nn.Linear(args.dim_att, C, bias=False)
        self.gate       = nn.Linear(C, args.dim_att, bias=False)

        if args.use_miss:   # аналогично
            self.receptance = MiSSLinear(self.receptance, shard_size=16)
            self.key        = MiSSLinear(self.key,        shard_size=16)
            self.value      = MiSSLinear(self.value,      shard_size=16)
            self.output     = MiSSLinear(self.output,     shard_size=16)
            self.gate       = MiSSLinear(self.gate,       shard_size=16)

        self.ln_x = nn.GroupNorm(H, args.dim_att, eps=1e-5 * (HEAD_DIVISOR ** 2))

    # --------------------------- GPT‑forward ---------------------------
    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = self.output(x * g)
        return x

    # --------------------------- RNN‑forward ---------------------------
    def init_state(self, batch_size, device):
        C = self.key.weight.shape[1]
        H = self.n_head
        S = self.head_size
        state = torch.zeros(batch_size, C + H * S * S, device=device, dtype=torch.bfloat16)
        return state

    def forward_one_step(self, x, state):
        B, T, C = x.shape
        assert T == 1
        H = self.n_head
        S = self.head_size

        sx = state[:, :C]  # (B, C)
        s_mat = state[:, C:].view(B, H, S, S)  # (B, H, S, S)

        # ------------------------- time shift --------------------------
        xx = sx - x.view(B, C)  # (B, C)

        # -------------------- dynamic coefficients ---------------------
        xxx = x.view(B, C) + xx * self.time_maa_x.view(1, C)  # (B, C)
        xxx = torch.tanh(xxx @ self.time_maa_w1)  # (B, 5*D_MIX_LORA)
        xxx = xxx.view(B, 5, -1).transpose(0, 1)  # (5, B, D_MIX_LORA)
        xxx = torch.bmm(xxx, self.time_maa_w2)  # (5, B, C)
        mw, mk, mv, mr, mg = xxx.unbind(0)  # each (B, C)

        xw = x.view(B, C) + xx * (self.time_maa_w.view(1, C) + mw)
        xk = x.view(B, C) + xx * (self.time_maa_k.view(1, C) + mk)
        xv = x.view(B, C) + xx * (self.time_maa_v.view(1, C) + mv)
        xr = x.view(B, C) + xx * (self.time_maa_r.view(1, C) + mr)
        xg = x.view(B, C) + xx * (self.time_maa_g.view(1, C) + mg)

        # ---------- projection ----------
        r = self.receptance(xr).view(B, H, 1, S)  # (B, H, 1, S)
        k = self.key(xk).view(B, H, S, 1)  # (B, H, S, 1)
        v = self.value(xv).view(B, H, 1, S)  # (B, H, 1, S)
        g = F.silu(self.gate(xg))  # (B, dim_att)

        # ---------- dynamic attenuation ----------
        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2  # (B, dim_att)
        w = w.view(B, H, S, 1)
        w = torch.exp(-torch.exp(w.float())).to(dtype=w.dtype)  # actual attenuation coefficient

        # ---------- recurrent step ----------
        u = self.time_faaaa.view(1, H, S, 1)  # (1, H, S, 1)
        a = k @ v  # (B, H, S, S)
        x_out = r @ (u * a + s_mat)  # (B, H, 1, S)
        s_new = a + w * s_mat

        x_out = x_out.view(B, H * S)  # (B, dim_att)

        # ---------- group norm ----------
        x_out = self.ln_x(x_out)  # правильно используем модуль
        x_out = self.output(x_out * g)

        # ---------- save state ----------
        new_state = torch.cat([x.view(B, C), s_new.view(B, H * S * S)], dim=1)
        return x_out, new_state


##############################################################################
# 2. Channel Mixing (FFN) RWKV_CMix_x060
##############################################################################
class RWKV_CMix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)
            ddd = torch.linspace(0, 1, args.n_embd).reshape(1, 1, -1)
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key        = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value      = nn.Linear(args.dim_ffn, args.n_embd, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)

        if args.use_miss:
            self.key        = MiSSLinear(self.key,        shard_size=16)
            self.value      = MiSSLinear(self.value,      shard_size=16)
            self.receptance = MiSSLinear(self.receptance, shard_size=16)

        # save n_embd to create the correct state
        self.n_embd = args.n_embd

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r
        k = torch.relu(self.key(xk)) ** 2
        return torch.sigmoid(self.receptance(xr)) * self.value(k)

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.n_embd, device=device, dtype=torch.bfloat16)

    def forward_one_step(self, x, state):
        B, T, C = x.shape
        assert T == 1
        xx = state - x.view(B, C)
        xk = x.view(B, C) + xx * self.time_maa_k
        xr = x.view(B, C) + xx * self.time_maa_r
        k = torch.relu(self.key(xk)) ** 2
        out = torch.sigmoid(self.receptance(xr)) * self.value(k)
        new_state = x.view(B, C)
        return out, new_state


##############################################################################
# 3. Block (Attention + FFN)
##############################################################################
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        #if layer_id == 0:
        #    self.ln0 = nn.LayerNorm(args.n_embd)
        self.att = RWKV_Tmix_x060(args, layer_id)
        self.ffn = RWKV_CMix_x060(args, layer_id)

    def forward(self, x):
        #if self.layer_id == 0:
        #    x = self.ln0(x)
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    def init_state(self, batch_size, device):
        return (self.att.init_state(batch_size, device),
                self.ffn.init_state(batch_size, device))

    def forward_one_step(self, x, state):
        att_state, ffn_state = state
        #if self.layer_id == 0:
        #    x = self.ln0(x)
        x1 = self.ln1(x)
        a_out, new_att_state = self.att.forward_one_step(x1, att_state)
        x = x + a_out
        x2 = self.ln2(x)
        f_out, new_ffn_state = self.ffn.forward_one_step(x2, ffn_state)
        x = x + f_out
        return x, (new_att_state, new_ffn_state)


import tqdm


##############################################################################
# 4. Full model RWKV‑6
##############################################################################
class FileRWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.dim_att = args.n_embd
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for n, p in self.named_parameters():
            shape = p.shape
            if "ln_" in n or ".ln" in n or "time_" in n or '_mask' in n or '.mask.' in n or n.endswith(
                    '_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias'):
                if 'ln_x.weight' in n:
                    layer_scale = (1 + int(n.split('.')[1])) / self.args.n_layer
                    nn.init.constant_(p, layer_scale ** 0.7)
                else:
                    pass
            elif n == "emb.weight":
                nn.init.uniform_(p, -1e-4, 1e-4)
            elif n == "head.weight":
                gain = 0.5 * math.sqrt(
                    self.args.vocab_size / self.args.n_embd) if self.args.vocab_size > self.args.n_embd else 0.5
                nn.init.orthogonal_(p, gain=gain)
            elif n.endswith('.weight'):
                scale = 1.0
                zero_params = [".att.output.", ".ffn.value.", ".ffn.receptance."]
                if any(k in n for k in zero_params):
                    scale = 0.0
                elif ".att.key." in n or ".att.gate." in n:
                    scale = 0.1
                if scale == 0:
                    nn.init.zeros_(p)
                else:
                    nn.init.orthogonal_(p, gain=scale)
            else:
                pass

    def forward(self, idx):
        x = self.emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        x = self.head(x)
        return x

    # ---------- RNN‑mode ----------
    def init_rnn_state(self, batch_size):
        states = []
        for block in self.blocks:
            states.append(block.init_state(batch_size, self.emb.weight.device))
        return states

    def forward_one_step(self, token, state):
        x = self.emb(token)
        new_states = []
        for block, s in zip(self.blocks, state):
            x, new_s = block.forward_one_step(x, s)
            new_states.append(new_s)
        x = self.ln_out(x)
        logits = self.head(x)
        return logits, new_states

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()
        state = self.init_rnn_state(prompt.shape[0])
        # префиллинг
        for i in range(prompt.size(1)):
            token = prompt[:, i:i + 1]
            _, state = self.forward_one_step(token, state)

        generated = []
        next_token = prompt[:, -1:]
        for _ in tqdm.tqdm(range(max_new_tokens)):
            logits, state = self.forward_one_step(next_token, state)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token.item())
        return generated


# ##############################################################################
# # 5. Пример использования
# ##############################################################################
# if __name__ == "__main__":
#     os.environ["RWKV_HEAD_SIZE_A"] = "64"
#     os.environ["RWKV_CTXLEN"] = "1024"
#     os.environ["RWKV_FLOAT_MODE"] = "bf16"
#
#
#     class Args:
#         vocab_size = 65558
#         ctx_len = 1024
#         n_layer = 12
#         n_embd = 512
#         head_size_a = 64
#         head_size_divisor = 8
#
#
#     args = Args()
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = RWKV(args).to(device).bfloat16()
#     print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}\n")
#
#     # Обучение (GPT‑режим)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#
#     B, T = 2, 512
#     inp = torch.randint(0, args.vocab_size, (B, T)).to(device)
#     tgt = torch.randint(0, args.vocab_size, (B, T)).to(device)
#     for step in range(100):
#         optimizer.zero_grad()
#
#         logits = model(inp)
#         loss = F.cross_entropy(logits.view(-1, args.vocab_size), tgt.view(-1))
#         loss.backward()
#         optimizer.step()
#         if step % 2 == 0:
#             print(f"step {step}, loss = {loss.item():.4f}")
#
#     # Генерация (RNN‑режим)
#     prompt = torch.tensor([[0, 1, 2, 3, 4]], device=device, dtype=torch.long)
#     generated = model.generate(prompt, max_new_tokens=2048, temperature=0.8, top_k=10)
#     print("Сгенерировано:", generated)