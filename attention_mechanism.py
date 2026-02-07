import math
import torch

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

        k = k.repeat_interleave(self.num_groups, dim=2) #repeate heads 4(group_size) times so num of head for k became 12 = query heads for attn calculation.
        v = v.repeat_interleave(self.num_groups, dim=2) #repeate heads 4(group_size) times so num of head for k became 12 = query heads for attn calculation.