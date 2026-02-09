import math
import torch
from torch import nn, F

def attention(q,k,v):
    d_k     = q.shape[-1] #(B, H, Lq, D)
    #k.transpose(-2, -1)-->(B, H, D, Lk)
    scores  = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k) #(B, H, Lq, Lk)--> multiplies (Lq, D) @ (D, Lk) for every (B, H) batch
    weights = torch.softmax(scores, dim=-1) #(B, H, Lq, Lk)
    #v = (B, H, Lk, Dv)
    return torch.matmul(weights, v) #(Lq, Lk) @ (Lk, Dv) per (B, H) result is (B, H, Lq, Dv)

def causal_attention(q,k,v):
    d_k    = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(d_k)
    mask   = torch.triu(
                            torch.ones(
                                scores.shape[-1],
                                scores.shape[-1]
                            ), #This will 
                            diagonal=1
                        )
    scores = scores.masked_fill(mask, float('-inf'))
    return torch.matmul(torch.softmax(scores, dim=-1), v)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12):
        self.num_heads = num_heads #12
        self.head_dim  = hidden_dim / num_heads #768/12 = 64
        self.q_proj    = nn.Linear(hidden_dim, hidden_dim) #(768,768)
        self.k_proj    = nn.Linear(hidden_dim, hidden_dim) #(768,768)
        self.v_proj    = nn.Linear(hidden_dim, hidden_dim) #(768,768)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim) #(768,768)

    def forward(self, x): #x shape = (2,512,768)
        B,S,_ = x.shape
        q     = self.q_proj(x).view(B,S,self.num_heads,self.head_dim).transpose(1,2) #(2,512,768)->view:(2,512,12,64)->transpose->(2,12,512,64)
        k     = self.k_proj(x).view(B,S,self.num_heads,self.head_dim).transpose(1,2) #(2,512,768)->view:(2,512,12,64)->transpose->(2,12,512,64) 
        v     = self.v_proj(x).view(B,S,self.num_heads,self.head_dim).transpose(1,2) #(2,512,768)->view:(2,512,12,64)->transpose->(2,12,512,64)
        out   = causal_attention(q,k,v) #(2, 12, 512, 64)
        return self.out_proj(
            out.transpose(1,2).contiguous().view(B,S,-1) #(2,512,768)
        ) 



class GroupedQueryAttention(nn.module):
    self __init__(self, hidden_dim=768, num_q_heads=12, num_kv_heads=4):
        self.num_q_heads  = num_q_heads #12
        self.num_kv_heads = num_kv_heads #3
        self.num_groups   = num_q_heads/num_kv_heads #4
        self.head_dim     = hidden_dim/num_q_heads #768/12 = 64
        self.q_proj       = nn.Linear(hidden_dim, num_q_heads*self.head_dim) #(768, 12*64)
        self.k_proj       = nn.Linear(hidden_dim, num_kv_heads*self.head_dim) #(768, 3*64)
        self.v_proj       = nn.Linear(hidden_dim, num_kv_heads*self.head_dim) #(768, 3*64)

    def forward(self, x): #x of shape (2,512,768)
        B, S, _ = x.shape
        q = self.q_proj(x).view(B,S,self.num_q_heads, self.head_dim) #(2,512,768)->view:(2,512,12,64)
        k = self.k_proj(x).view(B,S,self.num_kv_heads, self.head_dim) #(2,512,192)->view:(2,512,3,64)
        v = self.v_proj(x).view(B,S,self.num_kv_heads, self.head_dim) #(2,512,192)->view:(2,512,3,64)

        k = k.repeat_interleave(self.num_groups, dim=2) #repeate heads 4(group_size) times so num of head for k became 12 = query heads for attn calculation
        v = v.repeat_interleave(self.num_groups, dim=2) #repeate heads 4(group_size) times so num of head for k became 12 = query heads for attn calculation



class FeedForward(nn.Module):
    def __init__(self, hidden_dim=768, intermediate_dim=3072):
        super.__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


class SwiGLU(nn.Module):
    def __init__():
        super.__init__():
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w3 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(
            F.silu(self.w1(x)) * self.w3(x)
        )


class KVCache:
    def __init__(self, num_layers, max_seq_len, num_heads, head_dim, dtype):
        self.seq_len = 0

        shape        = (num_layers, max_seq_len, num_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype)
        self.v_cache = torch.zeros(shape, dtype=dtype)

    def update(self, layer_idx, k, v):
        n         = k.shape[1]
        start     = self.seq_len
        end       = self.seq_len + n

        self.k_cache[layer_idx, start:end] = k
        self.v_cache[layer_idx, start:end] = v

    def get(self, layer_idx):
        k = self.k_cache[layer_idx, :self.seq_len]
        v = self.v_cache[layer_idx, :self.seq_len]
        return k,v

def prefill_with_cache(model, prompt_ids, cache):
    x = model.embed(prompt_ids)

    for layer_idx, layer in enumerate(model.layers):
        q = layer.q_proj(x)
        k = layer.k_proj(x)
        v = layer.v_proj(x)

        cache.update(layer_idx, k, v)

        x = x + attention(q, k, v)
        x = x + layer.ffn(q, k, v)

    cache.advance(prompt_ids.shape[1])
    return model.lm_head(x)

def decode_with_cache(model, token_id, cache):
    x = model.embed(token_id)

    for layer_idx, layer in enumerate(model.layers):
        q       = layer.q_proj(x)
        k_new   = layer.k_proj(x)
        v_new   = layer.v_proj(x)

        cache.update(layer_idx, k_new, v_new)

        k_full, v_full = cache.get(layer_idx)

        x = x + attention(q, k_full, v_full)
        x = x + layer.ffn(x)

    cache.advance(1)

    return model.lm_head(x)[:,-1,:]
