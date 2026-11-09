import torch
import torch.nn as nn
import numpy as np

def get_batch(x:np.ndarray,
              batch_size:int,
              context_length:int,
              device:torch.device|None = None,
            ):
            """
            Args:
                x:输入的token IDs序列
                batch_size:每批次包含的样本数量
                context_length:上下文大小
                device:张量放置的设备
            Retruns:
                inputs: 输入序列张量，形状 (batch_size, context_length)
                targets: 目标序列张量，形状 (batch_size, context_length)
            """
            # 计算合法起始索引的最大值（包含）
            # 输入取 [i, i+context_length)，目标取 [i+1, i+1+context_length)
            # 需保证目标不越界：i + 1 + context_length <= len(x)
            max_start_idx = len(x) - context_length - 1

            # 随机采样 batch_size 个起始索引（有放回采样，符合训练数据加载惯例）
            start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)

            # 向量化生成所有样本的索引，避免循环，提升效率
            # 输出形状：(batch_size, context_length)
            seq_offset = np.arange(context_length)
            input_idx = start_indices[:, None] + seq_offset[None, :]
            target_idx = input_idx + 1  # 目标序列整体右移一位

            # 索引取值并转为 PyTorch 长整型张量，直接放到指定设备
            inputs = torch.tensor(x[input_idx], dtype=torch.long, device=device)
            targets = torch.tensor(x[target_idx], dtype=torch.long, device=device)

            return inputs, targets


            

