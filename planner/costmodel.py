"""Cost model + planner for one "GEMM + collective" call site (innovation 2), v2.

Why offline: an online probe must build each candidate's symmetric-heap context, and on
this platform a context, once built, changes the speed of later engine ops even after it
is released (F44). So the planner predicts instead of trying.

Plan-time inputs (none builds a symmetric-heap context):
  G   vendor GEMM time of the call site (single card, torch.matmul)
  B   collective payload, MB (from the shape)
  op  rs | ag_up | ag_gu
Platform calibration, fitted once from operator-level measurements:
  C_rccl(B)  RCCL collective time, linear in B, per collective kind (rs / ag)
  C_dx(B)    our pure data-plane time (RS measured; AG latent k*B)
Candidate time models, fitted per op by RELATIVE least squares (weights 1/T):
  stock   T = G + C_rccl(B) + h0
  dx      T = a1*G + a2*Cx + a3*max(G, Cx) + h1
            blocked GEMMs (a1), pushes hidden behind them (max term), exposed tail and
            the fixed host/control cost of a fused call (h1)
  splitN  T = T_stock / r,  log r = w0 + w1*log(G/C_rccl) + w2*log B + w3*log(G/N)
            (N = 2, 4; the last term = per-slice GEMM size: small slices run inefficiently)
Guarded choice (the "never slower than stock" rule):
  each candidate model has an error band q_c = q-th percentile (default q = 0.80) of its relative
  over-optimism on the training cells, log(T_meas / T_pred). A candidate is chosen
  only if its pessimistic time T_pred * exp(q_c) beats stock by margin; else stock.
"""
import math
import numpy as np


def _kind(op):
    return "rs" if op == "rs" else "ag"


def _wlsq(A, y):
    """least squares on relative error: minimise sum ((A@th - y) / y)^2"""
    A = np.asarray(A, float); y = np.asarray(y, float)
    w = 1.0 / y
    return np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]


class CostModel:
    def __init__(self, mode="b2b", margin=0.02, q=0.80, guard=True):
        self.mode, self.margin, self.q, self.guard = mode, margin, q, guard

    # ------------------------------------------------------------------ fit
    def fit(self, cells):
        cells = [c for c in cells if not c.get("anomaly")]
        self.cal = {}
        for kind in ("rs", "ag"):
            cs = [c for c in cells if _kind(c["op"]) == kind]
            B = np.array([c["B"] for c in cs])
            self.cal[kind + "_rccl"] = _wlsq(np.c_[np.ones_like(B), B], [c["C"] for c in cs])
        cs = [c for c in cells if c["op"] == "rs"]
        B = np.array([c["B"] for c in cs])
        self.cal["rs_dx"] = _wlsq(np.c_[np.ones_like(B), B], [c["Cdx"] for c in cs])

        self.th, self.band, self.w = {}, {}, {}
        for op in ("rs", "ag_up", "ag_gu"):
            cs = [c for c in cells if c["op"] == op and ("dx_" + self.mode) in c]
            if len(cs) < 5:
                continue
            G = np.array([c["G"] for c in cs])
            Cr = np.array([self.C_rccl(op, c["B"]) for c in cs])
            ys = np.array([c["stock_" + self.mode] for c in cs])
            h0 = float(np.median(ys - G - Cr))
            self.th[op + "_stock"] = h0
            yd = np.array([c["dx_" + self.mode] for c in cs])
            if op == "rs":
                Cx = np.array([self.C_dx(op, c["B"]) for c in cs])
                self.th[op + "_dx"] = (None, self._fit_dx(G, Cx, yd))
            else:
                best = None
                for k in np.geomspace(0.002, 0.05, 40):
                    Cx = k * np.array([c["B"] for c in cs])
                    th = self._fit_dx(G, Cx, yd)
                    r = self._rel_rmse(G, Cx, yd, th)
                    if best is None or r < best[0]:
                        best = (r, k, th)
                self.th[op + "_dx"] = (best[1], best[2])
            # error bands on the training cells (relative over-optimism)
            ps = np.array([self.predict(op, c["G"], c["B"])["stock"] for c in cs])
            pd = np.array([self.predict(op, c["G"], c["B"])["dx"] for c in cs])
            self.band[(op, "stock")] = float(np.quantile(np.log(ys / ps), self.q))
            self.band[(op, "dx")] = float(np.quantile(np.log(yd / pd), self.q))

        for op in ("rs", "ag_up"):
            for nb in (2, 4):
                key = f"r_split{nb}_{self.mode}"
                cs = [c for c in cells if c["op"] == op and key in c]
                if len(cs) < 6:
                    continue
                X = np.c_[np.ones(len(cs)), [math.log(c["G"] / self.C_rccl(op, c["B"])) for c in cs],
                          [math.log(c["B"]) for c in cs], [math.log(c["G"] / nb) for c in cs]]
                y = np.array([math.log(c[key]) for c in cs])
                w = np.linalg.lstsq(X, y, rcond=None)[0]
                self.w[(op, nb)] = w
                # band: how much the measured ratio falls short of the predicted ratio
                self.band[(op, f"split{nb}")] = float(np.quantile(X @ w - y, self.q))
        return self

    @staticmethod
    def _fit_dx(G, Cx, y):
        return _wlsq(np.c_[G, Cx, np.maximum(G, Cx), np.ones_like(G)], y)

    @staticmethod
    def _rel_rmse(G, Cx, y, th):
        p = np.c_[G, Cx, np.maximum(G, Cx), np.ones_like(G)] @ th
        return float(np.sqrt(np.mean(((p - y) / y) ** 2)))

    # -------------------------------------------------------------- predict
    def C_rccl(self, op, B):
        a = self.cal[_kind(op) + "_rccl"]
        return a[0] + a[1] * B

    def C_dx(self, op, B):
        a = self.cal["rs_dx"]
        return a[0] + a[1] * B

    def predict(self, op, G, B):
        """predicted time (ms) per candidate for one call site."""
        out = {}
        Cr = self.C_rccl(op, B)
        out["stock"] = G + Cr + self.th.get(op + "_stock", 0.0)
        d = self.th.get(op + "_dx")
        if d is not None:
            k, th = d
            Cx = self.C_dx(op, B) if k is None else k * B
            out["dx"] = float(th @ np.array([G, Cx, max(G, Cx), 1.0]))
        for nb in (2, 4):
            w = self.w.get((op, nb))
            if w is not None and Cr > 0:
                r = math.exp(w @ np.array([1.0, math.log(G / Cr), math.log(B), math.log(G / nb)]))
                out[f"split{nb}"] = out["stock"] / r
        return out

    def choose(self, op, G, B, allowed=None):
        p = self.predict(op, G, B)
        if allowed is not None:
            p = {k: v for k, v in p.items() if k == "stock" or k in allowed}
        if self.guard:
            # pessimistic candidate time vs optimistic-free stock time
            score = {k: v * (math.exp(max(0.0, self.band.get((op, k), 0.0))) if k != "stock" else 1.0)
                     for k, v in p.items()}
        else:
            score = dict(p)
        best = min(score, key=score.get)
        if best != "stock" and score[best] > p["stock"] * (1.0 - self.margin):
            best = "stock"
        return best, p


def measured(c, mode):
    """measured time per candidate in main-harness units (split: stock / measured ratio)."""
    out = {"stock": c["stock_" + mode]}
    if ("dx_" + mode) in c:
        out["dx"] = c["dx_" + mode]
    for nb in (2, 4):
        r = c.get(f"r_split{nb}_{mode}")
        if r:
            out[f"split{nb}"] = c["stock_" + mode] / r
    return out
