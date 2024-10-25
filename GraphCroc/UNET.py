import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F

from GraphUNET.ops import GCN, Pool, norm_g, Unpool
    
class Encoder(nn.Module):
    def __init__(self, ks, dim, act, drop_p):
        super(Encoder, self).__init__()
        self.ks = ks
        self.down_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.LNs = nn.ModuleList()
        self.bottom_gcn = GCN(dim, dim, act, drop_p)
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.LNs.append(nn.LayerNorm(dim))
    
    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []

        for i in range(self.l_n):
            g = norm_g(g)
            h1 = self.down_gcns[i](g, h)
            h = self.LNs[i](h + h1)
            down_outs.append(h)
            adj_ms.append(g)
            g, h, idx = self.pools[i](g, h)
            indices_list.append(idx)
        return g, h, adj_ms, down_outs, indices_list

class Decoder(nn.Module):
    '''
    gcn
    '''
    def __init__(self, ks, dim, act, drop_p) -> None:
        super(Decoder, self).__init__()
        self.inp_LNs = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.LNs = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.inp_LNs.append(nn.LayerNorm(dim))
            self.unpools.append(Unpool())
            self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.LNs.append(nn.LayerNorm(dim))

        self.out_ln = nn.LayerNorm(dim)

    def forward(self, h, ori_h, down_outs, adj_ms, indices_list):
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, idx)
            h1 = self.inp_LNs[i](down_outs[up_idx] + h)
            g = norm_g(g)
            h = self.up_gcns[i](g, h1)
            h = self.LNs[i](h + h1)
        h = self.out_ln(h + ori_h)
        return h


class Unet(nn.Module):
    '''
    two-way network
    '''
    def __init__(self, in_dim=None, args=None, s_gcn_state=None, encoder_state=None, s_ln_state=None) -> None:
        super(Unet, self).__init__()
        self.act = getattr(nn, args.act)()
        self.mask_ratio = args.mask_ratio

        self.s_gcn = GCN(in_dim, args.dim, self.act, args.drop_p)
        self.s_ln = nn.LayerNorm(args.dim)
        if s_gcn_state:
            self.s_gcn.load_state_dict(s_gcn_state)
            for param in self.s_gcn.parameters(): # freeze the grad of source gcn
                param.requires_grad = False
        if s_ln_state:
            self.s_ln.load_state_dict(s_ln_state)
            for param in self.s_ln.parameters(): # freeze the grad of source gcn
                param.requires_grad = False

        self.g_enc = Encoder(args.ks, args.dim, self.act, args.drop_p)
        if encoder_state:
            self.g_enc.load_state_dict(encoder_state)
            for param in self.g_enc.parameters(): # freeze the grad of encoder
                param.requires_grad = False

        self.bot_gcn = GCN(args.dim, args.dim, self.act, args.drop_p)
        self.bot_ln = nn.LayerNorm(args.dim)
        self.g_dec1 = Decoder(args.ks, args.dim, self.act, args.drop_p)
        self.g_dec2 = Decoder(args.ks, args.dim, self.act, args.drop_p)

        self.reduce1 = nn.Linear(args.dim, args.dim)
        self.reduce2 = nn.Linear(args.dim, args.dim)
    
    def forward(self, gs, hs):
        o_gs = self.embed(gs, hs)
        return self.customBCE(o_gs, gs), o_gs
    
    def embed(self, gs, hs):
        o_gs = []
        for g, h in zip(gs, hs):
            og = self.embed_one(g, h)
            o_gs.append(og)
        return o_gs

    def embed_one(self, g, h):
        g = norm_g(g)
        h = self.s_gcn(g, h)
        h = self.s_ln(h)
        ori_h = h
        g, h, adj_ms, down_outs, indices_list = self.g_enc(g, h)

        g = norm_g(g)
        h = self.bot_gcn(g, h)
        h = self.bot_ln(h)
        h1 = self.g_dec1(h, ori_h, down_outs, adj_ms, indices_list)
        h2 = self.g_dec2(h, ori_h, down_outs, adj_ms, indices_list)

        h1 = self.reduce1(h1)
        h2 = self.reduce2(h2)
        h = (h1 @ h2.T)

        return torch.sigmoid(h)
        # return torch.sigmoid((h+h.T)/2)

    def customBCE(self, o_gs, gs):
        loss = 0
        cnts = 0
        for og, g in zip(o_gs, gs):
            tn = g.numel()
            zeros = tn - g.sum()
            ones = g.sum()
            one_weight = tn / 2 / ones
            zero_weight = tn / 2 / zeros
            weights = torch.where(g == 0, zero_weight, one_weight)
            loss += F.binary_cross_entropy(og, g, weight=weights)
            cnts += 1
        loss /= cnts
        return loss