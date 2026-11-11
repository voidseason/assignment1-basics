import numpy as np
import torch


def softmax(self,x:torch.Tensor,dim:int)->torch.Tensor:
    x_max = x.max(dim=dim,keepdim=True).values
    # 广播减法
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    sum_exp = exp_x.sum(dim=dim,keepdim=True)
    # 广播除法
    return exp_x / sum_exp

# decode the prompt
def decode(x:np.ndarray,max_context_length:int,top_p:float,eos_id: int,model,temperature:float=1.0,device:torch.device=torch.device("cpu")):
    """
    自回归解码，同时支持 top_k + top_p 采样
    Args:
        x: 初始prompt token ids, np.ndarray shape [seq_len]
        max_context_length: 生成后的最大总长度（包含prompt）
        top_k: top-k，设0表示不启用topk
        top_p: nucleus top-p，1.0表示不启用topp
        model: 推理模型
    Return:
        完整序列 np.ndarray
    """
    # 转成torch tensor, batch=1
    seq = torch.from_numpy(x).long().unsqueeze(0).to(device)
    while seq.size(1) < max_context_length:
        logits = model(seq)  # [1,vocab_size]
        logits = logits / temperature
        probs = softmax(logits,dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cum_probs > top_p
            # 关键：第一个超过top-p的token保留，后面全部删掉
            remove_mask[..., 1:] = remove_mask[..., :-1] & remove_mask[..., 1:]
            sorted_probs[remove_mask] = 0.0

            # 写回原概率分布
            probs.zero_()
            probs.scatter_(-1, sorted_indices, sorted_probs)

        # 重归一化
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # EOS 判断
        if next_token_id.item() == eos_id:
            seq = torch.cat([seq, next_token_id], dim=-1)
            break

        seq = torch.cat([seq, next_token_id], dim=-1)

    return seq.squeeze(0).cpu().numpy()



