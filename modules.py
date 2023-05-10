from torch import nn
import torch
import torch.nn.functional as F
import pytorch_lightning as L

class FFLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None, dropout=.1, norm=True):
        super().__init__()
        layers = []
        if out_dim is None:
            out_dim = in_dim
        if norm:
            layers.append(nn.LayerNorm(in_dim))
        layers.append(nn.Linear(in_dim, out_dim)),           
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
    
class FFResLayer(nn.Module):
    def __init__(self, in_dim, dropout=.1, norm=True):
        super().__init__()
        layers = []
        if norm:
            layers.append(nn.LayerNorm(in_dim))
        layers.append(nn.Linear(in_dim, in_dim)),           
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.layer = nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x) + x

class OutLayer(nn.Module):
    def __init__(self, in_dim, outputs, norm=True):
        super().__init__()
        layers = []
        if norm:
            layers.append(nn.LayerNorm(in_dim))
        layers.append(nn.Linear(in_dim, in_dim)),           
        layers.append(nn.GELU())
        layers.append(nn.Linear(in_dim, outputs)),           
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class FFHead(nn.Module):
    def __init__(self, hidden_dim, n_outputs, n_layers=1, dropout=.1, norm=True, residual=True):
        super().__init__()
        layertype = FFResLayer if residual else FFLayer 
        fflayers = [nn.Dropout(dropout)] + \
                   [layertype(hidden_dim, dropout=dropout, norm=norm) for _ in range(n_layers)] + \
                   [OutLayer(hidden_dim, n_outputs)]
        self.classifier = nn.Sequential(*fflayers)

    def forward(self, x):
        return self.classifier(x)

class NormalizeProject(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.projection_img = nn.Linear(hidden_dim, output_dim)
        self.projection_txt = nn.Linear(hidden_dim, output_dim)
        self.projection_prod = nn.Linear(hidden_dim, output_dim)
        self.projection_diff = nn.Linear(hidden_dim, output_dim)

    def forward(self, img, txt, prod=None, diff=None):
        if prod is None:
            prod = txt*img
        if diff is None:
            diff = torch.abs(txt-img)

        img_p = self.projection_img(F.normalize(img, dim=-1))
        txt_p = self.projection_txt(F.normalize(txt, dim=-1))
        prod_p = self.projection_prod(F.normalize(prod, dim=-1))
        diff_p = self.projection_diff(F.normalize(diff, dim=-1))

        return img_p, txt_p, prod_p, diff_p

class BatchnNormProject(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        return self.projection(F.normalize(self.batchnorm(x), dim=-1))


class WeightedHead(nn.Module):
    def __init__(self, hidden_dim, middle_dim, n_outputs, dropout=.1):
        super().__init__()
        self.preprocess = FFLayer(hidden_dim, middle_dim, dropout=dropout)
        self.attention = nn.Linear(hidden_dim, middle_dim)
        self.classifier = OutLayer(middle_dim, n_outputs)
    
    def forward(self, img, txt):
        representation = self.preprocess(torch.cat([img, txt, torch.abs(txt-img), txt*img], axis=-1))
        #x = self.preprocess(stacked)
        #att = torch.softmax(self.attention(stacked), dim=0)
        return self.classifier(representation)

class FullHead(nn.Module):
    def __init__(self, hidden_dim, n_outputs, n_layers=1, dropout=.1):
        super().__init__()
        self.d_att = nn.Linear(hidden_dim, hidden_dim)
        self.p_att = nn.Linear(hidden_dim, hidden_dim)
        fflayers = [FFResLayer(hidden_dim*4, dropout=dropout)] + \
                   [FFResLayer(hidden_dim*4, dropout=dropout) for _ in range(n_layers-1)]
        self.processor = nn.Sequential(*fflayers)
        self.classifier = OutLayer(hidden_dim*4, n_outputs)
    
    def _reduce_matrix(self, matrix, att_layer):
        matrix_t = torch.transpose(matrix, 1, 2)
        att_a = torch.softmax(att_layer(matrix.sum(-1)), -1)
        att_b = torch.softmax(att_layer(matrix_t.sum(-1)), -1)

        return torch.sum(matrix * att_b.unsqueeze(-1), 1) + torch.sum(matrix_t * att_a.unsqueeze(-1), 1)

    def forward(self, img, txt, diff_m, prod_m):
        diff = self._reduce_matrix(diff_m, self.d_att)
        prod = self._reduce_matrix(prod_m, self.p_att)
        all_features = torch.cat([img, txt, diff, prod], -1)
        
        return self.classifier(self.processor(all_features))

class FuseHead(nn.Module):
    def __init__(self, hidden_dim, n_outputs, n_layers=1, dropout=.1):
        super().__init__()
        concat_dim = hidden_dim*4
        fflayers = [FFResLayer(concat_dim, dropout=dropout)] + \
                   [FFResLayer(concat_dim, dropout=dropout) for _ in range(n_layers-1)]
        self.processor = nn.Sequential(*fflayers)
        self.classifier = OutLayer(concat_dim, n_outputs)
    
    def forward(self, img, txt):
        diff = torch.abs(img-txt)
        prod = img*txt
        all_features = torch.cat([img, txt, diff, prod], -1)
        return self.classifier(self.processor(all_features))

class ReductionHead(nn.Module):
    def __init__(self, hidden_dim, reduce_dim, n_outputs, n_layers=1, dropout=.1):
        super().__init__()
        
        self.reducer = nn.Linear(hidden_dim, reduce_dim)

        reduced_flat_dim = reduce_dim*reduce_dim
        fflayers = [FFResLayer(reduced_flat_dim, dropout=dropout)] + \
                   [FFResLayer(reduced_flat_dim, dropout=dropout) for _ in range(n_layers-1)]
        self.processor = nn.Sequential(*fflayers)
        self.classifier = OutLayer(reduced_flat_dim, n_outputs)
    
    def forward(self, img, txt):
        prod = self.reducer(img).unsqueeze(-1) @ self.reducer(txt).unsqueeze(1)
        bs = img.shape[0]
        prod = prod.view(bs, -1)
        return self.classifier(self.processor(prod))

class ReduceFuseHead(nn.Module):
    def __init__(self, hidden_dim, reduce_dim, n_outputs, n_layers=1, dropout=.1):
        super().__init__()
        
        self.reducer = nn.Linear(hidden_dim, reduce_dim)
        reduced_flat_dim = 2*reduce_dim*reduce_dim + hidden_dim*2
        fflayers = [FFResLayer(reduced_flat_dim, dropout=dropout)] + \
                   [FFResLayer(reduced_flat_dim, dropout=dropout) for _ in range(n_layers-1)]
        self.processor = nn.Sequential(*fflayers)
        self.classifier = OutLayer(reduced_flat_dim, n_outputs)
    
    def forward(self, img, txt):
        bs = img.shape[0]
        prod = (self.reducer(img).unsqueeze(-1) @ self.reducer(txt).unsqueeze(1)).view(bs, -1)
        diff = torch.abs(self.reducer(img).unsqueeze(-1) - self.reducer(txt).unsqueeze(1)).view(bs, -1)
        all_features = torch.cat([prod, diff, img, txt], dim=-1)
        return self.classifier(self.processor(all_features))

class FuseExtraHead(FuseHead):    
    def forward(self, img, txt, diff, prod):
        all_features = torch.cat([img, txt, diff, prod], -1)
        return self.classifier(self.processor(all_features))


class TransformerCMAFuser(nn.Module):
    def __init__(self, hidden_dim, layers=1, dropout=.1, epsilon=1e-8):
        super().__init__()

        module = nn.TransformerEncoderLayer(hidden_dim, 12, hidden_dim*4, 
                                            dropout=dropout, 
                                            activation=F.gelu, 
                                            batch_first=True, 
                                            norm_first=True,)
        self.siamese_attention = nn.Linear(hidden_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(module, layers)
        self.epsilon = epsilon

    def forward(self, img, txt, mask=None):
        if mask is not None:
            img_msk = torch.zeros_like(img[:, :, 0], device=self.device)
            txt_msk = (1-mask)
            mask = torch.cat([img_msk, txt_msk], -1).bool()

        x = torch.cat([img, 
                       txt], dim=1)
        x = self.transformer(x, src_key_padding_mask=mask)
        txt_token_idx = img.shape[0]
        img_token, txt_token = x[:, 0], x[:, txt_token_idx]
        
        e_i = torch.exp(self.siamese_attention(img_token)) 
        e_t = torch.exp(self.siamese_attention(txt_token)) 
        lmd = e_i / (e_i + e_t)
        return lmd * img_token + (1-lmd) * txt_token


class CMAFuser(nn.Module):
    def __init__(self, proj_dim, hidden_dim, epsilon=1e-8):
        super().__init__()

        self.attention_weights = nn.Linear(proj_dim, hidden_dim)
        self.projection_weights = nn.Linear(proj_dim, hidden_dim)
        self.epsilon = epsilon

    def forward(self, img, txt):
        e_i = torch.exp(self.attention_weights(img))
        e_t = torch.exp(self.attention_weights(txt))
        img_prj = torch.relu(self.projection_weights(img)) 
        txt_prj = torch.relu(self.projection_weights(txt))
        lmd = e_i / (e_i + e_t)
        return img_prj.pow(lmd) * txt_prj.pow(1-lmd)


class TransformerFuser(L.LightningModule):
    def __init__(self, hidden_dim, layers=1, dropout=.1):
        super().__init__()

        module = nn.TransformerEncoderLayer(hidden_dim, 12, hidden_dim*4, 
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

        x = torch.cat([img, 
                       txt], dim=1)
        x = self.transformer(x, src_key_padding_mask=mask)
        txt_token_idx = img.shape[0]
        
        return x[:, 0], x[:, txt_token_idx], #img, txt

class TransformerExtraFuser(TransformerFuser):
    def forward(self, img, txt, mask=None, output_all=False):
        if mask is not None:
            bs = img.shape[0]
            img_msk = torch.zeros_like(img[:, :, 0])
            txt_msk = (1-mask)
            multi_msk = torch.zeros((bs, 2), device=self.device)
            mask = torch.cat([multi_msk, img_msk, txt_msk], -1).bool()

        img_tok, txt_tok = img[:, 0], txt[:, 0]
        diff = torch.abs(img_tok-txt_tok).unsqueeze(1)
        mult = (img_tok*txt_tok).unsqueeze(1)
        
        x = torch.cat([diff, mult, img, txt], dim=1)
        x = self.transformer(x, src_key_padding_mask=mask)
        txt_token_idx = img.shape[0]+2

        if output_all:
            return x[:, 2:txt_token_idx], x[:, txt_token_idx:], x[:, 0], x[:, 1]
        else:
            return x[:, 2], x[:, txt_token_idx], x[:, 0], x[:, 1] #img, txt, diff, mult



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
