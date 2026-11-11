"""FT-Transformer (Gorishniy et al., 2021) for tabular data, implemented from scratch in PyTorch (no pretrained weights)."""
import numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
torch.set_num_threads(1)
class _Net(nn.Module):
    def __init__(s, n_num, cat_sizes, d=32, heads=4, layers=2, drop=0.1, ncls=3):
        super().__init__()
        s.wn = nn.Parameter(torch.randn(n_num, d) * 0.1); s.bn = nn.Parameter(torch.zeros(n_num, d))
        s.emb = nn.ModuleList([nn.Embedding(k, d) for k in cat_sizes]); s.cls = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        L = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, activation="gelu", batch_first=True, norm_first=True)
        s.enc = nn.TransformerEncoder(L, layers); s.out = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, ncls))
    def forward(s, xn, xc):
        t = xn.unsqueeze(-1) * s.wn + s.bn
        if len(s.emb): t = torch.cat([t, torch.stack([e(xc[:, i]) for i, e in enumerate(s.emb)], 1)], 1)
        t = torch.cat([s.cls.expand(len(t), -1, -1), t], 1); return s.out(s.enc(t)[:, 0])
class FTTransformer:
    def __init__(s, num, cat, epochs=150, lr=2e-3, wd=1e-4, bs=64, seed=0): s.num, s.cat, s.epochs, s.lr, s.wd, s.bs, s.seed = num, cat, epochs, lr, wd, bs, seed
    def _prep(s, X):
        xn = torch.tensor(s.sc.transform(X[s.num]), dtype=torch.float32)
        xc = torch.tensor(np.stack([X[c].map(s.vocab[i]).fillna(len(s.vocab[i])).astype(int).values for i, c in enumerate(s.cat)], 1), dtype=torch.long)
        return xn, xc
    def fit(s, X, y):
        torch.manual_seed(s.seed); np.random.seed(s.seed)
        s.sc = StandardScaler().fit(X[s.num]); s.vocab = [{v: j for j, v in enumerate(sorted(X[c].unique()))} for c in s.cat]
        xn, xc = s._prep(X); yt = torch.tensor(y, dtype=torch.long)
        s.net = _Net(len(s.num), [len(v) + 1 for v in s.vocab]); opt = torch.optim.AdamW(s.net.parameters(), lr=s.lr, weight_decay=s.wd)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=s.lr, total_steps=s.epochs * int(np.ceil(len(y) / s.bs)))
        for ep in range(s.epochs):
            s.net.train(); perm = torch.randperm(len(y))
            for i in range(0, len(y), s.bs):
                b = perm[i:i + s.bs]; opt.zero_grad(); nn.functional.cross_entropy(s.net(xn[b], xc[b]), yt[b]).backward(); opt.step(); sch.step()
        return s
    def predict_proba(s, X):
        s.net.eval(); xn, xc = s._prep(X)
        with torch.no_grad(): return torch.softmax(s.net(xn, xc), 1).numpy()
