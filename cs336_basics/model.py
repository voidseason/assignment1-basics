import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

class Linear(nn.Module):
    def __init__(self,
    in_features:int,
    out_features:int,
    device:torch.device|None = None,
    dtype:torch.dtype|None = None
    ):
        super.__init__()
        self.W = nn.Parameter(
                torch.empty(in_features, out_features, device=device, dtype=dtype)
            )
        # 按照3.3.1 Parameter Initialization的要求来初始化
        std = math.sqrt(2.0 / (in_features + out_features))
        trunc_normal_(self.W,mean=0.0,std=std,a=-3*std,b=3*std)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return x @ self.W

class RMSNorm(nn.Module):
    def __init__(self,
    d_model:int,
    eps:float = 1e-5,
    device:torch.device|None = None,
    dtype:torch.dtype|None = None
    ):
        super().__init__()
        self.eps = eps
        # 按照3.3.1 Parameter Initialization的要求来初始化
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self,x:torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight
        return result.to(in_dtype)

class SwiGLUFFN(nn.Module):
    def __init__(self,d_model:int,d_ff:int = None):
        super().__init()
        if d_ff is None:
            d_ff= int(8 / 3 * d_model)
            d_ff = math.ceil(d_ff/64)*64
        self.w1 = Linear(d_model,d_ff)
        self.w2 = Linear(d_ff,d_model)
        self.w3 = Linear(d_mode,d_ff)
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        value = self.w3(x)

        # SiLU激活：
        silu_gate = gate * torch.sigmoid(gate)

        hidden = silu_gate * value
        return self.w2(hidden)


class RoPE(nn.module):
    def __init__(self,theta:float,d_k:int,max_seq_len :int,device:torch.device|None = None):
        """
        初始化 RoPE，预计算所有位置的 cos/sin 缓存
        Args:
            theta: 基础频率常数 Θ，通常取 10000.0
            d_k: 单个注意力头的特征维度
            max_seq_len: 支持的最大序列长度
            device: 缓存存放的设备
        """
        super().__init__()
        # 生成每组旋转频率
        freqs = 1.0 / (theta ** (torch.arange(0,d_k,2,device=device,dtype=torch.float32) / d_k))
        # 生成 0~max_seq_len-1 的位置索引
        positions = torch.arange(max_seq_len,device=device,dtype=torch.float32)
        # 外积得到每个位置、每个频率组的旋转角度：形状 (max_seq_len, d_k//2),论文中的 𝜃𝑖,k
        freqs_cis = torch.outer(positions,freqs)
        # 预计算 cos 和 sin，注册为 buffer（非可训练参数，随模型移动设备）
        self.register_buffer("cos_cached", freqs_cis.cos())
        self.register_buffer("sin_cached", freqs_cis.sin())

    def forward(self,x:torch.Tensor,token_position:torch.Tensor)->torch.Tensor:
        """
        对输入张量施加 RoPE 旋转
        Args:
            x: 输入张量，形状 (..., seq_len, d_k)，支持任意数量的前置 batch 维度
            token_positions: 每个 token 的绝对位置索引，形状 (..., seq_len)，和 x 的前置维度对齐
        Returns:
            旋转后的张量，形状和 x 完全一致
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        cos = self.cos_cached[token_position]
        sin = self.sin_cached[token_position]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        x_rot = torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2)
        return x_rot.to(in_dtype)


def softmax(self,x:torch.Tensor,dim:int)->torch.Tensor:
    x_max = x.max(dim=dim,keepdim=True).values
    # 广播减法
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    sum_exp = exp_x.sum(dim=dim,keepdim=True)
    # 广播除法
    return exp_x / sum_exp


def scaled_dot_product_atttention(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,mask:torch.Tensor):
    """
    缩放点积注意力
    Args:
        q: 查询张量，形状 (batch_size, ..., seq_len_q, d_k)，支持任意前置batch维度
        k: 键张量，形状 (batch_size, ..., seq_len_k, d_k)
        v: 值张量，形状 (batch_size, ..., seq_len_k, d_v)
        mask: 可选布尔掩码，形状 (seq_len_q, seq_len_k)
              值为 True 的位置保留（概率和为1），值为 False 的位置概率置0
    Returns:
        输出张量，形状 (batch_size, ..., seq_len_q, d_v)
    """
    d_k = q.size(-1)  # 查询/键的特征维度

    # 1. 计算 Q·K^T 注意力分数
    scores = torch.einsum( q, k,"... s d, ... t d -> ... s t")
    # 2. 缩放
    scale = math.sqrt(d_k)
    scores = score / scale
    # 3.应用掩码
    if mask is not None:
        # mask为False，取反后填充-inf
        scores = scores.masked_fill(~mask, float("-inf"))
    
    attn_weights = softmax(scores, dim=-1)
    output = torch.einsum(attn_weights,v,"... s t,... t d -> s d")

    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048
    ):
        super().__init__()
        assert d_model % num_heads ==0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # 三个输入投影层: W_Q,W_K,W_V
        self.q_proj = Linear(d_model,d_model)
        self.k_proj = Linear(d_model,d_model)
        self.v_proj = Linear(d_model,d_model)
        # 输出投影层
        self.out_proj = Linear(d_model,d_model)
        # RoPE模块：只作用于Q和K，头维度自动作为batch维度
        self.rope = RoPE(
            theta=rope_theta,
            d_k=self.head_dim,
            max_seq_len=max_seq_len
        )
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        """
        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
        Returns:
            输出张量，形状 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影得到 Q、K、V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # 2.拆分成多头：总维度拆成 num_heads × head_dim
        q = torch.rearrange(q,"batch_size,seq_len,d_model -> batch_size,seq_len,num_heads,head_dim")
        k = torch.rearrange(q,"batch_size,seq_len,d_model -> batch_size,seq_len,num_heads,head_dim")
        v = torch.rearrange(q,"batch_size,seq_len,d_model -> batch_size,seq_len,num_heads,head_dim")
        # 3. 仅对 Q 和 K 施加 RoPE 旋转
        position = torch.arange(seq_len, device=x.device)
        q = self.rope(q,position)
        k = self.rope(k,position)
        # 4. 构造因果掩码
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        # 5. 调用缩放点积注意力
        # 每个头独立计算，batch维度自动并行
        attn_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        attn_output = torch.rearrange(attn_output, "batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)")
        output = self.out_proj(attn_output)
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model:int,
        num_heads:int,
        d_ff:int
    ):
        """
        Pre-Norm 结构的 Transformer 块
        Args:
            d_model: 输入隐藏层总维度
            num_heads: 注意力头数量
            d_ff: 前馈网络中间层维度
        """
        super().__init__()
        # 第一层:因果多头注意力 + Pre-Norm
        self.norm1 = RMSNorm(d_model)
        self.multihead_attn = MultiHeadSelfAttention(d_model,num_heads)
        # 第二层：SwiGLU 前馈网络 + Pre-Norm
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model,d_ff)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        """
        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
        Returns:
            输出张量，形状和输入完全一致
        """
        # 第一层：
        x = x + self.multihead_attn(norm1(x))
        # 第二层:
        x = x+ self.ffn(self.norm2(x))

        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model:int,
        num_heads:int,
        d_ff:int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
    ):
        """
        完整的因果 Transformer 语言模型
        Args:
            vocab_size: 词表大小，决定词嵌入和输出头的维度
            context_length: 最大上下文长度，用于 RoPE 缓存大小
            num_layers: Transformer Block 的堆叠层数
            d_model: 隐藏层总维度
            num_heads: 注意力头数量
            d_ff: 前馈网络中间层维度
            rope_theta: RoPE 基础频率常数
        """
        self.vocab_size = vocab_size
        self.context_length = context_length

        # 1. 词嵌入层
        # 作业初始化规则：均值0，方差1，截断在[-3, 3]
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        trunc_normal_(self.token_embedding.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        # 2. 堆叠多层 Transformer Block
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
            )
            for _ in range(num_layers)
        ])

        # 3. 最终归一化层:所有 Block 计算完后，必须再加一层 RMSNorm，再接输出头
        self.final_norm = RMSNorm(d_model)

        # 4. LM 输出头：映射到词表大小的未归一化 logits
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: 输入 token 索引，形状 (batch_size, seq_len)
        Returns:
            logits: 未归一化的词表分数，形状 (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # 1.token 索引 -> 词嵌入向量
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)

        # 2.依次经过每一层 Transformer Block
        for layer in self.layers:
            x = layer(x)

        # 3.最终归一化
        x = self.final_norm(x)

        # 4.输出词表 logits（不要加 softmax，作业要求未归一化）
        logits = self.lm_head(x)
        return logits
    
