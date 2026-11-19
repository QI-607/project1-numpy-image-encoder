"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        B, N, D = x.shape
        # 把 768 维连续切成 H 个头（每个 head_dim 维），再对调头轴与 token 轴。
        # [B,N,D] -> reshape [B,N,H,head_dim] -> transpose [B,H,N,head_dim]
        return x.reshape(B, N, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # scores[b,h,i,j] = q[b,h,i]·k[b,h,j] / sqrt(head_dim)，只交换 k 的最后两轴。
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(np.float32(self.head_dim))

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # 用 softmax 权重对 value 加权求和，得到每个头的输出 context。
        context = coefficients @ v
        B, H, N, hd = context.shape
        # 合并 12 个头：transpose 把头轴挪回 token 后，再 reshape 回 [B,N,D]。
        merged = context.transpose(0, 2, 1, 3).reshape(B, N, H * hd)

        # TODO-03c: output projection, using out_weight and out_bias.
        # 权重存 [out,in]，转置后相乘再加偏置，返回 [B,N,D]。
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition.
        # 第一个残差：把注意力增量加回原始 x（不是 Norm1 之后的值）。
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # MLP：768 -> 3072（GELU 激活）-> 768，得到 mlp_update。
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias
        # 第二个残差：把 MLP 增量加回 residual。
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        B, C, H, W = images.shape
        P = 32                        # 每个补丁边长
        gh, gw = H // P, W // P       # 每边切成 7 块
        # reshape：把 224 各拆成 (7,32)，即网格轴(7) + 块内像素轴(32)。
        x = images.reshape(B, C, gh, P, gw, P)
        # transpose：把两个网格轴提到前面，块内轴 (C,32,32) 靠后。
        x = x.transpose(0, 2, 4, 1, 3, 5)
        # reshape：7*7=49 个补丁，每个 3*32*32=3072 维。
        return x.reshape(B, gh * gw, C * P * P)

    def embed(self, images):
        """TODO-05. Project patches, prepend CLS and add positions."""
        patches = self.patchify(images)  # [B,49,3072]
        # 补丁投影到 768 维（无偏置，权重 [out,in] 需转置）。
        x = patches @ self.patch_weight.T
        # 复制 CLS 行到每张图，拼在 token 轴最前面，再加位置编码。
        cls = np.tile(self.cls, (images.shape[0], 1, 1))
        tokens = np.concatenate([cls, x], axis=1)
        return tokens + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # 选第 0 行 CLS，做 post-Norm，再用 [768,512] 的投影矩阵（已是 in,out，不转置）。
        cls = tokens[:, 0, :]
        output = self.post_norm(cls) @ self.projection

        # 最后单位化（题目已给），得到 [B,512] 输出。
        return unit_vector(output)
