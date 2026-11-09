import torch
import torch.nn as nn
import math

def cross_entropy(logits:torch.Tensor,targets:torch.Tensor):
    """
    交叉熵损失函数:
    Args:
        logits:预测的未归一化的分数，形状:(...,vocab_size),前面九二一是任意数量的batch类维度，
               比如(batch),(batch,seq_len)
        targets:真实标签，形状:(...)，和logits去掉最后一维的形状完全一致
    Return:
        标量，所有位置交叉损失的平均值
    """
    # 1.取最大值
    max_val = x.max(dim=-1,keepdim=True).values
    logits_stable = logits - max_val
    # 2.计算log_sum_exp值
    logits_exp = logits_stable.exp()
    logits_sum_exp = logits_exp.sum(dim=-1,keepdim=True)
    log_sum_exp = logits_sum_exp.log()
    # 3.得到log_softmax
    log_softmax = logits_stable - log_sum_exp
    # 4.取出正确标签对应的 log_softmax 的值
    # 用unsqueeze函数升维
    targets_unsqueezed = targets.unsqueeze(-1)
    correct_log_probs = log_softmax.gather(dim=-1, index=targets_unsqueezed).squeeze(-1)

    # 5. 取负得到逐点损失，再全局求平均
    loss = -correct_log_probs.mean()
    return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self,
    params,
    lr:float = 1e-3,
    betas : tuple = (0.9,0.999),
    eps:float = 1e-8,
    weight_decay:float = 0.0
    ):
        super().__init__(params,defaults={
            "lr":lr,
            "beta1":beta[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay
        })
    """
    AdamW 优化器（解耦权重衰减的 Adam）
    Args:
        params: 模型可训练参数
        lr: 基础学习率 α
        betas: 一阶矩、二阶矩的指数衰减系数 (β1, β2)
        eps: 数值稳定极小值 ε
        weight_decay: 权重衰减系数 λ
    """


    @torch.no_grad()
    def step(self,closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # 遍历所有参数组
        for gruop in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    # 一阶矩 m、二阶矩 v 初始全0，和参数同形状同设备
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    # 初始步数为0
                    state["t"] = 0
                
                m = state["m"]
                v = state["v"]
                t = state["t"]

                # 步数+1：t从1开始计数
                t += 1

                # 2. 计算偏差修正后的有效学习率（对应算法第7行）
                # α_t = α * sqrt(1 - β2^t) / (1 - β1^t)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                lr_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # 3. 解耦权重衰减（对应算法第8行）
                # θ = θ - α * λ * θ
                # 形如mul_这种带下划线的接口，都是在原张量上进行的操作，节省了内存，mul_表示乘
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # 4. 更新一阶矩 m（对应算法第9行）
                # m = β1*m + (1-β1)*g
                # add_表示加
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 5. 更新二阶矩 v（对应算法第10行）
                # v = β2*v + (1-β2)*g²
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 6. 最终参数更新（对应算法第11行）
                # θ = θ - α_t * m / (sqrt(v) + ε)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)

                # 保存更新后的步数
                state["t"] = t

        return loss


def lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int
    ) -> float:
    """
    带 Warm-up 的余弦退火学习率调度
    Args:
        t: 当前迭代步数
        alpha_max: 最大学习率（峰值学习率）
        alpha_min: 最小学习率（退火结束后的保持值）
        T_w: Warm-up 总步数
        T_c: 余弦退火结束的总步数
    Returns:
        当前步数对应的学习率 alpha_t
    """
    # 阶段1：Warm-up 线性增长（t < T_w）
    if t < T_w:
        return (t / T_w) * alpha_max

    # 阶段2：余弦退火（T_w ≤ t ≤ T_c）
    elif t <= T_c:
        # 计算退火进度：0 ~ 1，对应从峰值降到最低值
        progress = (t - T_w) / (T_c - T_w)
        # 余弦值从 cos(0)=1 平滑过渡到 cos(π)=-1
        cos_value = math.cos(progress * math.pi)
        return alpha_min + 0.5 * (1 + cos_value) * (alpha_max - alpha_min)

    # 阶段3：退火结束，保持最低学习率（t > T_c）
    else:
        return alpha_min



def gradient_clipping(parameters, max_norm: float, eps: float = 1e-6) -> float:
    """
    全局L2范数梯度裁剪
    Args:
        parameters: 模型可训练参数的可迭代对象
        max_norm: 允许的最大全局梯度L2范数 M
        eps: 数值稳定极小值，默认 1e-6
    Returns:
        裁剪前的全局梯度L2范数
    """
    # 1. 收集所有存在梯度的参数
    grad_list = [p.grad for p in parameters if p.grad is not None]
    if not grad_list:
        return 0.0

    # 2. 计算全局梯度的L2范数：所有梯度元素平方和开根号
    total_sq = 0.0
    for grad in grad_list:
        total_sq += grad.pow(2).sum().item()
    total_norm = math.sqrt(total_sq)

    # 3. 计算缩放系数
    if total_norm <= max_norm:
        # 范数未超阈值，不裁剪，系数为1
        clip_factor = 1.0
    else:
        # 超阈值则按比例缩放，分母加eps防止除零
        clip_factor = max_norm / (total_norm + eps)

    # 4. 原地缩放所有梯度
    for grad in grad_list:
        grad.mul_(clip_factor)

    return total_norm




