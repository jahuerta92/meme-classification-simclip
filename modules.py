from torch import nn
import torch
import torch.nn.functional as F



class TransformerFuser(nn.Module):
    def __init__(self, input_dim, layers=1, heads=12, dropout=.1):
        super().__init__()
        module = nn.TransformerEncoderLayer(input_dim, heads, input_dim*4, 
                                            dropout=dropout, 
                                            activation=F.gelu, 
                                            batch_first=True, 
                                            norm_first=True,)
        self.transformer = nn.TransformerEncoder(module, layers)
    
    def forward(self, img, txt, mask=None):
        if mask is not None:
            img_msk = torch.zeros_like(img[:, :, 0], device=self.device)
            txt_msk = (1-mask)
            mask = torch.cat([img_msk, txt_msk], -1).bool()

        x = torch.cat([img, txt], dim=1)
        x = self.transformer(x,src_key_padding_mask=mask)
        txt_token_idx = img.shape[0]
        img_token, txt_token = x[:, 0], x[:, txt_token_idx]
        return torch.cat([img_token, txt_token, 
                          #img_token*txt_token, torch.abs(img_token-txt_token),
                          ], dim=-1)

class SimilarityFuser(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.cs = nn.CosineSimilarity(-1)
        self.proj = nn.Linear(input_dim, input_dim)
    def forward(self, img, txt):
        all_embs = [self.proj(txt*img), txt, img]
        sim_pairs = [(i, j) for i in range(len(all_embs)) for j in range(i) if i != j]
        similarities = [self.cs(all_embs[i], all_embs[j]).unsqueeze(-1) for i,j in sim_pairs] 
        return torch.cat(similarities, dim=-1)


class AlignedSimilarityFuser(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_network = nn.Sequential(nn.LayerNorm(6+input_dim*4),
                                            nn.Linear(6+input_dim*4, output_dim),
                                            nn.GELU(),
                                           )
        self.cs = nn.CosineSimilarity(-1)
        
    def forward(self, img, txt):
        dif = torch.abs(img-txt)
        mul = img*txt
        all_embs = [dif, mul, img, txt]
        sim_pairs = [(i, j) for i in range(len(all_embs)) for j in range(i) if i != j]
        similarities = [self.cs(all_embs[i], all_embs[j]).unsqueeze(-1) for i,j in sim_pairs] 
        combined = torch.cat(all_embs + similarities, dim=-1)

        return self.output_network(combined)

class AlignedFuser(nn.Module):
    def __init__(self, input_dim, proj_size, output_dim, dropout=.1):
        super().__init__()
        self.projector = nn.Sequential(nn.Linear(input_dim, proj_size),
                                       nn.GELU(),
                                        )
        self.output_network = nn.Sequential(nn.LayerNorm(proj_size*4),
                                            nn.Linear(proj_size*4, output_dim),
                                            nn.GELU(),
                                           )
        
    def forward(self, img, txt):
        img_p = self.projector(img)
        txt_p = self.projector(txt)

        dif = torch.abs(img_p-txt_p)
        mul = img_p*txt_p
        combined = torch.cat([dif, mul, img_p, txt_p], dim=-1)

        return self.output_network(combined), img_p, txt_p



class CrossAttentionBlock(nn.Module):
    #(N,L,E) when batch_first=True, where LL is the target sequence length, 
    #NN is the batch size, and EE is the embedding dimension embed_dim.
    def __init__(self, embed_dim, middle_size, heads=12, dropout=.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, heads, batch_first=True, dropout=dropout)
        self.linear_1 = nn.Linear(embed_dim, middle_size)
        self.linear_2 = nn.Linear(middle_size, embed_dim)
        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.act = F.gelu
        self.heads = heads
    
    def forward(self, a, b, mask=None):
        if mask is not None:
            mask = (1 - mask).bool()

        x, _ = self.cross_attention(a, b, b,
                                    key_padding_mask=mask)
        x = self.act(x)
        x = self.drop_1(x)
        out_1 = self.norm_1(x) + a
        x = self.act(self.linear_1(out_1))
        x = self.linear_2(x)
        x = self.drop_2(x)
        return self.norm_2(x) + out_1
    
class CrossAttentionFuser(nn.Module):
    def __init__(self, input_dim, output_dim, heads=12, dropout=.1):
        super().__init__()
        self.attention_img_to_txt = CrossAttentionBlock(input_dim, input_dim*4, heads=heads, dropout=dropout)
        self.attention_txt_to_img = CrossAttentionBlock(input_dim, input_dim*4, heads=heads, dropout=dropout)
        self.output_network = nn.Sequential(nn.Linear(input_dim*4, output_dim),
                                            nn.GELU(),
                                            )
                
    def forward(self, img_seq, txt_seq, img_att=None, txt_att=None):
        img_seq = self.attention_img_to_txt(img_seq, txt_seq, txt_att)
        txt_seq = self.attention_txt_to_img(txt_seq, img_seq, img_att)

        if img_att is not None:
            img_emb = torch.sum(img_seq * img_att.unsqueeze(-1), 1) / torch.sum(img_att, -1).unsqueeze(-1)
        else:
            img_emb = img_seq.mean(1)
        
        if txt_att is not None:
            txt_emb = torch.sum(txt_seq * txt_att.unsqueeze(-1), 1) / torch.sum(txt_att, -1).unsqueeze(-1)
        else:
            txt_emb = txt_seq.mean(1)

        dif = torch.abs(img_emb-txt_emb)
        mul = img_emb*txt_emb
        combined = torch.cat([dif, mul, txt_emb, img_emb], dim=-1)

        return self.output_network(combined), img_emb, txt_emb
    
class CrossAttentionFeatureBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(1, 1, batch_first=True)
        self.act = F.gelu
    
    def forward(self, a, b):
        a_f = a.unsqueeze(-1)
        b_f = b.unsqueeze(-1)
        x, _ = self.cross_attention(a_f, b_f, b_f)
        return self.act(x + a_f) 

class CrossAttentionFeatureFuser(nn.Module):
    def __init__(self, features, hidden_size, dropout=.1):
        super().__init__()
        self.attention_img_to_txt = CrossAttentionFeatureBlock()
        self.attention_txt_to_img = CrossAttentionFeatureBlock()
        self.output_network = nn.Sequential(nn.LayerNorm(features*4),
                                            nn.Linear(features*4, hidden_size),
                                            nn.GELU(),
                                            nn.Dropout(dropout)
                                            )
                
    def forward(self, img_features, txt_features):
        img_att = self.attention_img_to_txt(img_features, txt_features).squeeze()
        txt_att = self.attention_txt_to_img(txt_features, img_features).squeeze()

        dif = torch.abs(img_att-txt_att)
        mul = img_att*txt_att
        combined = torch.cat([dif, mul, 
                              img_att, txt_att],
                             dim=-1)

        return self.output_network(combined), img_att, txt_att

class AttentionFeatureMatrixBlock(nn.Module):
    def __init__(self, features, heads=12, dropout=.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(features, heads, batch_first=True, dropout=dropout)
        self.expander = nn.Linear(1, features)
        self.act = F.gelu
        self.heads = heads
    
    def forward(self, a, ab_matrix):
        a_f = self.expander(a.unsqueeze(-1))
        x, _ = self.cross_attention(a_f, ab_matrix, ab_matrix,
                                    )
        x = self.act(x) 
        return x
       
class AttentionFeatureMatrixFuser(nn.Module):
    def __init__(self, features, hidden_size, dropout=.1):
        super().__init__()
        self.attention_img_to_txt = AttentionFeatureMatrixBlock(features, dropout=dropout)
        self.attention_txt_to_img = AttentionFeatureMatrixBlock(features, dropout=dropout)
        self.output_network = nn.Sequential(nn.LayerNorm(features*2),
                                            nn.Linear(features*2, hidden_size),
                                            nn.GELU(),
                                            nn.Dropout(dropout)
                                            )
                
    def forward(self, img_features, txt_features):

        img_txt = img_features.unsqueeze(-1) @ txt_features.unsqueeze(1)

        img_att = self.attention_img_to_txt(img_features, img_txt).mean(1)
        txt_att = self.attention_txt_to_img(txt_features, torch.transpose(img_txt, 1, 2)).mean(1)

        combined = torch.cat([img_att+img_features, txt_att+txt_features],
                             dim=-1)

        return self.output_network(combined), img_att, txt_att
    

class ForwardModuleList(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.list_of_heads = nn.ModuleList(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return [head(*args, **kwargs) for head in self.list_of_heads]

class LossBalancer(nn.Module):
    def __init__(self, n_losses, eps=1e-8, pooling='mean', *args, **kwargs):
        super().__init__()
        self.loss_weights = nn.Parameter(torch.ones(size=(n_losses,)))
        self.eps=eps
        self.pooling = pooling

    def forward(self, loss):
        balanced = loss / (self.eps + self.loss_weights**2)
        regularized = torch.log(self.eps + self.loss_weights)
        if self.pooling == 'mean':
            return torch.nanmean(balanced + regularized)
        else:
            return balanced + regularized


class EMABalancer(nn.Module):
    def __init__(self, eps=1e-8, beta=.1, pooling='mean', *args, **kwargs):
        super().__init__()
        self.ema = None
        self.beta = beta
        self.eps=eps
        self.pooling = 'mean'

    def forward(self, loss):
        with torch.no_grad():
            new_avg = self.beta * loss
            nans = torch.isnan(new_avg)

            if self.ema is not None:
                old_avg = (1-self.beta) * self.ema
                new_avg[nans] = self.beta * self.ema[nans]
            else:
                old_avg = (1-self.beta)
                new_avg[nans] = self.beta

            self.ema = new_avg + old_avg
        balanced = loss / self.ema
        if self.pooling == 'mean':
            return torch.nanmean(balanced)
        else:
            return balanced
