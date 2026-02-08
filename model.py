from torch import nn
from attention_mechanism import MultiHeadAttention, SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, intermediate_dim):
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

        self.attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = SwiGLU(hidden_dim, intermediate_dim)

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        ffn_out  = self.ffn(self.norm2(x + attn_out))
        return x + ffn_out


class rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return x + ffn_out


class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, intermediate_dim):
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, intermediate_dim) for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


def decode_attention(q_new, k_cache, v_cache, k_new, v_env):
    k_cache = torch.cat([k_cache, k_new], dim=-2)
    v_cache = torch.cat([v_cache, v_new], dim=-2)
    scores  = torch.matmul(q_new, k_cache.transpose(-2,-1)) / math.sqrt(q_new.shape[-1])
    out     = torch.matmul(torch.softmax(scores, dim=-1), v_cache)
    return out, k_cache, v_cache