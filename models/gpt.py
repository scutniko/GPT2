"""
GPT模型定义
支持可插拔的注意力机制和位置编码
"""

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import GPTConfig
from modules.attentions import BaseAttention
from modules.position_encodings import (
    LearnedPositionEncoding,
    ALiBi,
    RoPE,
    SinusoidalPositionalEncoding,
)
from modules.block import Block


class GPT(nn.Module):
    """
    GPT模型
    支持多种注意力机制和位置编码的组合
    """
    
    def __init__(self, config, attention_class=None, position_encoding_class=None,
                 mlp_class=None, norm_class=None):
        """
        Args:
            config: GPTConfig 配置对象
            attention_class: 注意力机制类（如BaseAttention, MQAAttention等）
            position_encoding_class: 位置编码类（如LearnedPositionEncoding, RoPE等）
            mlp_class: MLP类（如MLP, ReLUMLP, SwiGLUMLP等）
            norm_class: 归一化层类（如LayerNorm, RMSNorm等）
        """
        super().__init__()
        self.config = config
        
        # 如果没有指定attention_class，使用默认的BaseAttention
        if attention_class is None:
            attention_class = BaseAttention
        self.attention_class = attention_class
        
        # 如果没有指定position_encoding_class，使用默认的LearnedPositionEncoding
        if position_encoding_class is None:
            position_encoding_class = LearnedPositionEncoding
        self.position_encoding_class = position_encoding_class
        
        # 如果没有指定mlp_class，使用默认的MLP
        if mlp_class is None:
            from modules.mlp import MLP as DefaultMLP
            mlp_class = DefaultMLP
        self.mlp_class = mlp_class
        
        # 如果没有指定norm_class，使用默认的LayerNorm
        if norm_class is None:
            norm_class = nn.LayerNorm
        self.norm_class = norm_class
        
        # 判断位置编码类型
        self.use_alibi = position_encoding_class == ALiBi
        self.use_rope = position_encoding_class == RoPE
        self.use_learned_pe = position_encoding_class == LearnedPositionEncoding
        self.use_sine_pe = position_encoding_class == SinusoidalPositionalEncoding
        
        # 构建transformer
        self.transformer = nn.ModuleDict(dict(
            # token embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
        ))
        
        # 根据位置编码类型添加相应的模块
        if self.use_learned_pe:
            # 可学习的位置编码
            self.transformer['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        elif self.use_sine_pe:
            # 正弦位置编码
            self.pos_encoder = SinusoidalPositionalEncoding(config.n_embd, config.block_size)
        elif self.use_alibi:
            # ALiBi不需要位置编码embedding，但需要ALiBi模块
            # ALiBi会在attention中使用，这里先创建
            self.alibi = ALiBi(config.n_head, config.block_size)
        elif self.use_rope:
            # RoPE在attention中使用
            head_dim = config.n_embd // config.n_head
            self.rope = RoPE(head_dim, config.block_size)
        
        # Transformer blocks
        # 传递位置编码对象、MLP类和归一化层类给每个 Block
        rope_to_pass = self.rope if self.use_rope else None
        alibi_to_pass = self.alibi if self.use_alibi else None
        self.transformer['h'] = nn.ModuleList([
            Block(config, attention_class, mlp_class=self.mlp_class, 
                  norm_class=self.norm_class, rope=rope_to_pass, alibi=alibi_to_pass) 
            for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.transformer['ln_f'] = self.norm_class(config.n_embd)
        
        # 最后将Transformer的输出映射到词表空间，将[B, T, C] -> [B, T, vocab_size]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        # 这里进行权重共享，将token embedding的权重与最后输出层的权重共享，这样可以节省内存，并且使得模型更加稳定。
        # GPT-2的实现中，token embedding的权重与最后输出层的权重是共享的。
        # 我们也预期这么做是合理的，因为一个训练充分的模型，从id到embeding和从embedding到logits的映射应该是一样的。
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化模型权重"""
        init_method = getattr(self.config, "init_method", "default")
        init_distribution = getattr(self.config, "init_distribution", "normal")
        if isinstance(module, nn.Linear):
            # 初始化权重，使得输出方差与输入方差一致
            if init_method == "default":
                std = 0.02
                # 如果模块有NANOGPT_SCALE_INIT属性，则使用更小的初始化方差，这是为了防止初始化方差过大，导致模型训练不稳定。
                if hasattr(module, 'NANOGPT_SCALE_INIT'):
                    std *= (2 * self.config.n_layer) ** -0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            elif init_method == "xavier":
                if init_distribution == "uniform":
                    torch.nn.init.xavier_uniform_(module.weight)
                else:
                    torch.nn.init.xavier_normal_(module.weight)
            elif init_method == "kaiming":
                if init_distribution == "uniform":
                    torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                else:
                    torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            else:
                raise ValueError(f"未知初始化方法: {init_method}")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # 初始化embedding的权重，使得输出方差与输入方差一致
            if init_method == "default":
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif init_method == "xavier":
                if init_distribution == "uniform":
                    torch.nn.init.xavier_uniform_(module.weight)
                else:
                    torch.nn.init.xavier_normal_(module.weight)
            elif init_method == "kaiming":
                if init_distribution == "uniform":
                    torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                else:
                    torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            else:
                raise ValueError(f"未知初始化方法: {init_method}")

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        """
        前向传播
        
        Args:
            idx: token indices [B, T]
            targets: target token indices [B, T] (可选，用于计算loss)
            past_kv: 可选的KV cache列表（每层一个）
            use_cache: 是否返回当前KV cache
            
        Returns:
            logits: [B, T, vocab_size]
            loss: scalar (如果提供了targets)
        """
        # 输入idx的shape是[B, T]，其中B是batch size，T是序列长度
        B, T = idx.size()
        past_len = 0
        if past_kv is not None and len(past_kv) > 0 and past_kv[0] is not None:
            first_cache = past_kv[0][0] if isinstance(past_kv[0], (tuple, list)) else past_kv[0]
            if first_cache is not None:
                if first_cache.dim() == 4:
                    past_len = first_cache.size(2)
                elif first_cache.dim() == 3:
                    past_len = first_cache.size(1)
        # 确保序列长度不超过最大序列长度
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # Token embedding
        tok_emb = self.transformer.wte(idx)  # shape (B, T, n_embd)
        
        # Position encoding
        if self.use_learned_pe:
            # 可学习的位置编码
            pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)  # shape (T)
            pos_emb = self.transformer.wpe(pos)  # shape (T, n_embd)
            x = tok_emb + pos_emb
        elif self.use_sine_pe:
            # 正弦位置编码
            pos_emb = self.pos_encoder(tok_emb, start_pos=past_len)  # shape (B, T, n_embd)
            x = tok_emb + pos_emb
        elif self.use_alibi or self.use_rope:
            # ALiBi和RoPE不需要在这里添加位置编码，在attention中处理
            x = tok_emb
        else:
            # 其他情况，直接使用token embedding
            x = tok_emb
        
        # 前向传播Transformer的每个block，得到[B, T, n_embd]
        presents = [] if use_cache else None
        if use_cache:
            if past_kv is None:
                past_kv = [None] * len(self.transformer.h)
            for layer_idx, block in enumerate(self.transformer.h):
                x, present_kv = block(x, past_kv=past_kv[layer_idx], use_cache=True)
                presents.append(present_kv)
        else:
            for block in self.transformer.h:
                x = block(x)
        
        # 最后将Transformer的输出进行LayerNorm，得到[B, T, n_embd]
        x = self.transformer.ln_f(x)
        
        # 将Transformer的输出映射到词表空间，得到[B, T, vocab_size]
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # 如果targets不为None，则计算交叉熵损失
        loss = None
        if targets is not None:
            # 这里用的是原始logits，而不是softmax结果，因为F.cross_entropy会自动内部计算softmax，所以不需要再手动计算。
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if use_cache:
            return logits, loss, presents
        return logits, loss

    def get_moe_aux_loss(self):
        """
        收集所有MoE MLP的辅助损失（如果存在）
        Returns:
            aux_loss: scalar 或 None
        """
        aux_losses = []
        for block in self.transformer.h:
            aux = getattr(block.mlp, "last_aux_loss", None)
            if aux is not None:
                aux_losses.append(aux)
        if not aux_losses:
            return None
        return torch.stack(aux_losses).sum()
    
    @classmethod
    def from_pretrained(cls, model_type, attention_class=None, position_encoding_class=None,
                        mlp_class=None, norm_class=None):
        """
        从HuggingFace加载预训练的GPT-2模型权重
        
        Args:
            model_type: 模型类型 ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
            attention_class: 注意力机制类（可选）
            position_encoding_class: 位置编码类（可选）
            mlp_class: MLP类（可选）
            norm_class: 归一化层类（可选）
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config, attention_class, position_encoding_class,
                    mlp_class=mlp_class, norm_class=norm_class)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process):
        """
        配置优化器
        
        Args:
            weight_decay: 权重衰减系数
            learning_rate: 学习率
            device_type: 设备类型 ('cuda' 或 'cpu')
            master_process: 是否为主进程（用于打印日志）
            
        Returns:
            optimizer: AdamW优化器
        """
        # 获取所有需要梯度更新的参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # 创建优化组。任何2D的参数都会进行权重衰减，否则不进行权重衰减。
        # 即所有矩阵型的weight tensors in matmuls + embeddings都会进行权重衰减，所有偏置和LayerNorm的参数都不会进行权重衰减。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        # 创建优化组
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # 计算参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        # 打印参数信息
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # 创建AdamW优化器，并使用fused版本（如果可用）
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

