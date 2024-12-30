import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union
import numpy as np

import torch
import torch.nn.functional as F

from utils import exact_div


# 硬编码的音频超参数
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    打开一个音频文件并读取为单声道波形，根据需要重新采样。

    参数:
        file (str): 要打开的音频文件路径。
        sr (int, 可选): 如果需要，重新采样到目标采样率。默认为 SAMPLE_RATE（16kHz）。

    返回:
        np.ndarray: 包含音频波形的 NumPy 数组，数据类型为 float32。
    """
    # 构建 ffmpeg 命令，用于解码音频并下混到单声道，同时根据需要重新采样
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        # 执行 ffmpeg 命令，捕获标准输出和标准错误
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    # 将标准输出字节数据转换为 NumPy 数组，数据类型为 int16
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    """
    对音频数组进行填充或裁剪，使其长度符合编码器预期的 N_SAMPLES。

    参数:
        array: 输入的音频数组，可以是 NumPy 数组或 PyTorch 张量。
        length (int, 可选): 目标长度，默认为 N_SAMPLES（480000）。
        axis (int, 可选): 沿着哪个轴进行填充或裁剪，默认为最后一个轴 (-1)。

    返回:
        填充或裁剪后的数组，保持输入类型（NumPy 数组或 PyTorch 张量）。
    """
    if torch.is_tensor(array):
        # 如果输入是 PyTorch 张量
        if array.shape[axis] > length:
            # 如果张量长度大于目标长度，则裁剪
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            # 如果张量长度小于目标长度，则填充
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            # 计算填充宽度，并进行填充
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        # 如果输入是 NumPy 数组
        if array.shape[axis] > length:
            # 如果数组长度大于目标长度，则裁剪
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            # 如果数组长度小于目标长度，则填充
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            # 计算填充宽度，并进行填充
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    加载用于将 STFT 转换为梅尔频谱图的梅尔滤波器组矩阵。

    该函数通过缓存机制避免重复加载滤波器组文件，实现了与 librosa 的解耦。
    滤波器组文件是通过以下命令生成的：

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )

    参数:
        device: 设备类型（例如 "cpu" 或 "cuda"）。
        n_mels (int): 梅尔滤波器的数量，目前只支持 80 和 128。

    返回:
        torch.Tensor: 梅尔滤波器组矩阵。
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    # 构建滤波器组文件的路径
    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        # 从文件中加载指定数量的梅尔滤波器组
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    计算音频的对数梅尔频谱图。

    参数:
        audio (Union[str, np.ndarray, torch.Tensor]): 输入音频，可以是文件路径、NumPy 数组或 PyTorch 张量。
            如果是文件路径，则加载音频文件；如果是 NumPy 数组或张量，则直接使用。
            音频应采用 16 kHz 采样率。

        n_mels (int): 梅尔滤波器的数量，目前只支持 80 和 128。

        padding (int): 在音频右侧填充的零样本数。

        device (Optional[Union[str, torch.device]]): 如果给定，音频张量将在进行 STFT 之前移动到该设备。

    返回:
        torch.Tensor: 包含梅尔频谱图的张量，形状为 (n_mels, n_frames)。
    """
    if not torch.is_tensor(audio):
        # 如果输入不是张量，则根据类型进行处理
        if isinstance(audio, str):
            # 如果是文件路径，则加载音频
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        # 如果指定了设备，则将音频移动到该设备
        audio = audio.to(device)
    if padding > 0:
        # 如果需要填充，则在音频右侧填充零
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    # 计算短时傅里叶变换（STFT）
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    # 计算频谱幅度的平方
    magnitudes = stft[..., :-1].abs() ** 2

    # 加载梅尔滤波器组
    filters = mel_filters(audio.device, n_mels)
    # 将滤波器组应用于幅度谱，计算梅尔频谱图
    mel_spec = filters @ magnitudes

    # 对梅尔频谱图进行对数转换，并进行裁剪以避免对数计算中的下溢
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # 将对数频谱裁剪到最大值以下 8 分贝
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    # 对对数频谱进行归一化，使其范围在 [-1.0, 1.0] 之间
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
