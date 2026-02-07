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
        self.num_heads = num_heads
        self.head_dim  = hidden_dim / num_heads
        self.q_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x): #x shape = (2,512,768)
        B,S,_ = x.shape
        q     = self.q_proj(x).view(B,S,self.num_heads,self.head_dim).transpose(1,2) #(2,512,12,64)->transpose->(2,12,512,64)
        k     = self.k_proj(x).view(B,S,self.num_heads,self.head_dim).transpose(1,2) #(2,512,12,64)->transpose->(2,12,512,64) 
        v     = self.v_proj(x).view(B,S,self.num_heads,self.head_dim).transpose(1,2) #(2,512,12,64)->transpose->(2,12,512,64)
        out   = causal_attention(q,k,v) #(2, 12, 512, 64)
        return self.out_proj(
            out.transpose(1,2).contiguous().view(B,S,-1) #(2,512,768
        ) 