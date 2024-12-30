import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from decoding import decode as decode_function
from decoding import detect_language as detect_language_function
from transcribe import transcribe as transcribe_function

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False


@dataclass
class ModelDimensions:
    """
    模型维度配置类，用于定义模型的各个维度参数。

    参数:
        n_mels (int): 梅尔滤波器组（Mel filterbank）的数量，用于音频特征提取。
        n_audio_ctx (int): 音频上下文的长度，影响音频编码器的输入序列长度。
        n_audio_state (int): 音频编码器的状态维度，通常与嵌入维度相同。
        n_audio_head (int): 音频编码器中多头注意力机制的头数。
        n_audio_layer (int): 音频编码器的层数。
        n_vocab (int): 词汇表的大小，即模型可以处理的唯一标记的数量。
        n_text_ctx (int): 文本上下文的长度，影响文本编码器的输入序列长度。
        n_text_state (int): 文本编码器的状态维度，通常与嵌入维度相同。
        n_text_head (int): 文本编码器中多头注意力机制的头数。
        n_text_layer (int): 文本编码器的层数。
    """
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    """
    自定义的 LayerNorm 层，支持将输入转换为 float 类型进行归一化处理，然后转换回原始数据类型。

    这是为了处理可能存在的半精度（fp16）输入，同时保持计算精度。
    """
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过层归一化处理后的张量，保持原始数据类型。
        """
        # 将输入张量转换为 float 类型进行归一化处理
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    """
    自定义的线性层，支持将权重和偏置转换为输入张量的数据类型。

    这是为了处理可能存在的半精度（fp16）输入，同时保持计算精度。
    """
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 经过线性变换后的张量。
        """
        # 将权重转换为输入张量的数据类型
        # 将偏置转换为输入张量的数据类型（如果存在）
        return F.linear(
            x,
            self.weight.to(x.dtype), 
            None if self.bias is None else self.bias.to(x.dtype), 
        )


class Conv1d(nn.Conv1d):
    """
    自定义的 1D 卷积层，支持将权重和偏置转换为输入张量的数据类型。

    这是为了处理可能存在的半精度（fp16）输入，同时保持计算精度。
    """
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。
            weight (torch.Tensor): 卷积核权重。
            bias (Optional[torch.Tensor]): 偏置。

        返回:
            torch.Tensor: 经过卷积后的张量。
        """
        # 将权重和偏置转换为输入张量的数据类型
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    """
    生成正弦波，用于位置编码。

    参数:
        length (int): 序列的长度。
        channels (int): 通道的数量，必须是偶数。
        max_timescale (int, 可选): 最大时间尺度，默认为 10000。

    返回:
        torch.Tensor: 正弦波张量，形状为 (length, channels)。
    """
    assert channels % 2 == 0
    # 计算对数时间尺度增量
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    # 计算逆时间尺度
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    # 计算缩放后的时间
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    # 生成正弦和余弦波，并连接起来
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


@contextmanager
def disable_sdpa():
    """
    上下文管理器，用于临时禁用自注意力机制中的自注意力路径（Self-Attention Path, SDPA）。

    这在某些情况下可能需要，例如在调试或特定训练阶段。

    使用方法:
        with disable_sdpa():
            # 在此代码块内，SDPA 被禁用
            ...
    """
    # 保存当前 SDPA 的使用状态
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制（Multi-Head Self-Attention）模块。

    多头注意力机制通过并行计算多个注意力头，从而捕获输入序列中不同位置之间的关系。

    属性:
        use_sdpa (bool): 是否使用 PyTorch 的缩放点积注意力（SDPA）实现。默认为 True。
    """
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        """
        初始化多头自注意力模块。

        参数:
            n_state (int): 输入和输出的状态维度。
            n_head (int): 注意力头的数量。
        """
        super().__init__()
        # 注意力头的数量
        self.n_head = n_head

        # 线性层，用于计算查询（query）、键（key）和值（value）
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        # 线性层，用于最终的输出
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 查询输入。
            xa (Optional[torch.Tensor]): 交叉注意力中的键和值输入。如果为 None，则执行自注意力。
            mask (Optional[torch.Tensor]): 注意力掩码。
            kv_cache (Optional[dict]): 键值缓存，用于存储键和值。

        返回:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 输出张量和注意力权重。
        """
        # 计算查询（query）
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # 如果没有缓存，或者没有交叉注意力输入，或者键不在缓存中，则计算键（key）和值（value）
            # 对于自注意力，键和值来自输入 x；对于交叉注意力，键和值来自输入 xa
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # 如果有缓存，并且有交叉注意力输入，则从缓存中获取键和值
            # 这用于加速推理过程，避免重复计算键和值
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        # 执行多头注意力机制
        wv, qk = self.qkv_attention(q, k, v, mask)
        # 通过最终的线性层输出
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        计算多头注意力。

        参数:
            q (torch.Tensor): 查询张量，形状为 (batch_size, n_ctx, n_state)。
            k (torch.Tensor): 键张量，形状为 (batch_size, n_ctx, n_state)。
            v (torch.Tensor): 值张量，形状为 (batch_size, n_ctx, n_state)。
            mask (Optional[torch.Tensor]): 注意力掩码。

        返回:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 输出张量和注意力权重。
        """
        # 获取批量大小、序列长度和状态维度
        n_batch, n_ctx, n_state = q.shape
        # 计算缩放因子
        scale = (n_state // self.n_head) ** -0.25
        # 重塑查询、键和值，以适应多头注意力
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            # 如果 SDPA 可用并且启用了 SDPA，则使用缩放点积注意力
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            # 重塑输出张量
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            # 否则，手动计算注意力权重
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            # 计算 softmax 注意力权重
            w = F.softmax(qk, dim=-1).to(q.dtype)
            # 计算最终输出
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            # 分离注意力权重以防止梯度传播
            qk = qk.detach()

        return out, qk


class ResidualAttentionBlock(nn.Module):
    """
    残差注意力块（Residual Attention Block），结合了多头自注意力和前馈神经网络，并通过残差连接和层归一化进行增强。
    """
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        """
        初始化残差注意力块。

        参数:
            n_state (int): 输入和输出的状态维度。
            n_head (int): 多头注意力机制中的头数。
            cross_attention (bool, 可选): 是否使用交叉注意力。默认为 False。
        """
        super().__init__()

        # 多头自注意力机制
        self.attn = MultiHeadAttention(n_state, n_head)
        # 层归一化层，用于注意力机制输入
        self.attn_ln = LayerNorm(n_state)

        # 如果使用交叉注意力，则初始化交叉注意力机制和相应的层归一化层
        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        # 前馈神经网络，由线性层、GELU 激活函数和另一个线性层组成
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        # 层归一化层，用于前馈神经网络输入
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。
            xa (Optional[torch.Tensor]): 交叉注意力输入。
            mask (Optional[torch.Tensor]): 注意力掩码。
            kv_cache (Optional[dict]): 键值缓存。

        返回:
            torch.Tensor: 输出张量。
        """
        # 自注意力 + 层归一化
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        # 如果使用交叉注意力，则添加交叉注意力 + 层归一化
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        # 前馈神经网络 + 层归一化
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    """
    音频编码器模块，用于将音频的梅尔频谱图转换为高维特征表示。

    音频编码器由卷积层和残差注意力块组成，通过卷积层提取局部特征，并通过注意力机制捕捉全局依赖关系。
    """

    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        """
        初始化音频编码器。

        参数:
            n_mels (int): 梅尔频谱图的通道数。
            n_ctx (int): 音频上下文的长度，影响输入序列的长度。
            n_state (int): 模型的隐藏状态维度。
            n_head (int): 多头注意力机制中的头数。
            n_layer (int): 残差注意力块的层数。
        """
        super().__init__()
        # 第一个卷积层，将梅尔频谱图从 n_mels 通道映射到 n_state 通道
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        # 第二个卷积层，进一步提取特征，并进行下采样
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        # 注册一个位置嵌入张量，使用正弦波生成
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        # 初始化多个残差注意力块
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        # 层归一化层，用于最终的输出
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入的梅尔频谱图，形状为 (batch_size, n_mels, n_ctx)。

        返回:
            torch.Tensor: 编码后的音频特征，形状为 (batch_size, n_ctx, n_state)。
        """
        # 应用第一个卷积层，并使用 GELU 激活函数
        x = F.gelu(self.conv1(x))
        # 应用第二个卷积层，并使用 GELU 激活函数
        x = F.gelu(self.conv2(x))
        # 转置张量形状，从 (batch_size, n_state, n_ctx) 变为 (batch_size, n_ctx, n_state)
        x = x.permute(0, 2, 1)

        # 确保输入形状与位置嵌入形状一致
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        # 添加位置嵌入，并转换为输入张量的数据类型
        x = (x + self.positional_embedding).to(x.dtype)

        # 依次通过多个残差注意力块
        for block in self.blocks:
            x = block(x)

        # 应用层归一化层
        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    """
    文本解码器模块，用于将音频编码器的输出作为条件，生成文本标记。

    文本解码器由嵌入层、残差注意力块和线性层组成，通过交叉注意力机制将音频特征融入文本生成过程。
    """
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        """
        初始化文本解码器。

        参数:
            n_vocab (int): 词汇表的大小。
            n_ctx (int): 文本上下文的长度，影响输入序列的长度。
            n_state (int): 模型的隐藏状态维度。
            n_head (int): 多头注意力机制中的头数。
            n_layer (int): 残差注意力块的层数。
        """
        super().__init__()

        # 词嵌入层，将词汇表中的每个词映射到隐藏状态维度
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        # 位置嵌入，可学习的参数，用于编码每个词的位置信息
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        # 初始化多个残差注意力块，并启用交叉注意力机制
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        # 层归一化层，用于最终的输出
        self.ln = LayerNorm(n_state)

        # 构建注意力掩码，实现因果注意力
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        前向传播函数。

        参数:
            x (torch.LongTensor): 输入的文本标记，形状为 (batch_size, <= n_ctx)。
            xa (torch.Tensor): 编码后的音频特征，形状为 (batch_size, n_audio_ctx, n_audio_state)。
            kv_cache (Optional[dict]): 键值缓存，用于存储键和值。

        返回:
            torch.Tensor: 输出的文本标记的对数概率，形状为 (batch_size, n_vocab)。
        """
        # 获取当前键值缓存中的偏移量
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        # 计算文本标记的嵌入，并添加位置嵌入
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        # 将嵌入后的张量转换为与音频特征相同的类型
        x = x.to(xa.dtype)

        # 依次通过多个残差注意力块
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        # 应用层归一化层
        x = self.ln(x)
        # 计算最终的输出对数概率
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class Whisper(nn.Module):
    """
    Whisper 模型类，实现了语音识别和翻译功能。

    Whisper 模型由音频编码器和文本解码器组成，能够将音频信号转换为文本，并支持多语言翻译。
    """
    def __init__(self, dims: ModelDimensions):
        """
        初始化 Whisper 模型。

        参数:
            dims (ModelDimensions): 模型维度配置，包含各个部分的维度参数。
        """
        super().__init__()
        # 保存模型维度配置
        self.dims = dims

        # 初始化音频编码器
        self.encoder = AudioEncoder(
            self.dims.n_mels,                # 梅尔频谱图的通道数
            self.dims.n_audio_ctx,           # 音频上下文的长度
            self.dims.n_audio_state,         # 音频编码器的状态维度
            self.dims.n_audio_head,          # 音频编码器多头注意力机制的头数
            self.dims.n_audio_layer,         # 音频编码器的层数
        )

        # 初始化文本解码器
        self.decoder = TextDecoder(
            self.dims.n_vocab,               # 词汇表的大小
            self.dims.n_text_ctx,            # 文本上下文的长度
            self.dims.n_text_state,          # 文本解码器的状态维度
            self.dims.n_text_head,           # 文本解码器多头注意力机制的头数
            self.dims.n_text_layer,          # 文本解码器的层数
        )
        # 默认情况下，使用解码器层的下半部分进行时间对齐；
        # 如果需要使用特定的注意力头，请使用下面的 `set_alignment_heads()` 方法。
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        ) # 创建一个布尔张量，形状为 (n_text_layer, n_text_head)

        # 将下半部分的注意力头设为 True
        all_heads[self.dims.n_text_layer // 2 :] = True
        # 注册对齐注意力头的稀疏缓冲区
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        """
        设置用于时间对齐的注意力头。

        参数:
            dump (bytes): 包含对齐注意力头信息的压缩和编码后的字节数据。
        """
        # 解码并解压字节数据
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        # 将解码后的数据重塑为 (n_text_layer, n_text_head) 的布尔张量
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        # 注册对齐注意力头的稀疏缓冲区
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        """
        将梅尔频谱图嵌入为音频特征。

        参数:
            mel (torch.Tensor): 输入的梅尔频谱图，形状为 (batch_size, n_mels, n_audio_ctx)。

        返回:
            torch.Tensor: 嵌入后的音频特征，形状为 (batch_size, n_audio_ctx, n_audio_state)。
        """
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        """
        计算文本标记的对数概率。

        参数:
            tokens (torch.Tensor): 输入的文本标记，形状为 (batch_size, <= n_text_ctx)。
            audio_features (torch.Tensor): 嵌入后的音频特征，形状为 (batch_size, n_audio_ctx, n_audio_state)。

        返回:
            torch.Tensor: 对数概率，形状为 (batch_size, n_vocab)。
        """
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播函数，计算文本标记的对数概率。

        参数:
            mel (torch.Tensor): 输入的梅尔频谱图。
            tokens (torch.Tensor): 输入的文本标记。

        返回:
            Dict[str, torch.Tensor]: 包含对数概率的字典。
        """
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        """
        获取模型所在的设备。

        返回:
            torch.device: 模型所在的设备。
        """
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        """
        判断模型是否为多语言模型。

        返回:
            bool: 如果词汇表大小大于或等于 51865，则为多语言模型。
        """
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        """
        获取模型支持的语言数量。

        返回:
            int: 语言数量。
        """
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        安装键值缓存钩子，用于存储和重用计算中的键和值。

        参数:
            cache (Optional[dict]): 初始缓存字典。

        返回:
            Tuple[Dict[nn.Module, torch.Tensor], List[nn.Module]]: 包含缓存和钩子的元组。
        """
        # 初始化缓存字典
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # 如果模块不在缓存中，或者输出序列长度大于文本上下文长度，则保存输出
                cache[module] = output
            else:
                # 否则，将输出连接到现有缓存中
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                # 如果层是多头注意力机制，则安装钩子
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        # 对解码器应用钩子安装函数
        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
