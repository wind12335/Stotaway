"""Experimental model of one method's benefit relative to stock.

This does not select among different overlap libraries. It models one selected
method using its own measured stock/method pairs. Training and uncertainty
calibration keep model families disjoint; results require external validation.
"""
import math
import numpy as np


def _features(cell, comm_ms, split_feature=False):
    """Only plan-time shape, single-GPU GEMM and fitted communication curve."""
    m, k, n, g = (float(cell[x]) for x in ("M", "K", "N", "G"))
    c = float(comm_ms)
    if min(m, k, n, g, c) <= 0 or not all(map(math.isfinite, (m, k, n, g, c))):
        raise ValueError("shape and timing inputs must be finite and positive")
    features = [math.log(m), math.log(k), math.log(n), math.log(g),
                math.log(c), math.log(g / c)]
    if split_feature:
        ratio = float(cell['split_gemm_ratio'])
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError('split_gemm_ratio must be finite and positive')
        features.append(math.log(ratio))
    return np.array(features, dtype=float)


class RelativeModel:
    """Predict log(stock/method) with per-op ridge regression and group guard.

    residual_quantile is estimated on out-of-fold *models*, not training cells.
    This is a heuristic lower bound, not a guaranteed confidence interval.
    """
    def __init__(self, mode="b2b", ridge=10.0, quantile=0.8, margin=0.02, split_feature=False):
        if mode not in ("b2b", "single") or ridge <= 0 or not 0 <= quantile <= 1 or not 0 <= margin < 1:
            raise ValueError("invalid model settings")
        self.mode, self.ridge, self.quantile, self.margin = mode, ridge, quantile, margin
        self.split_feature = split_feature
        self.models, self.guards = {}, {}

    def _fit_op(self, cells):
        # At deployment the held-out shape's TP8 C has not been measured.
        # Fit the communication curve on this fold only, including in the
        # inner cross-validation used to construct the guard.
        sizes = np.array([c['B'] for c in cells], dtype=float)
        measured_comm = np.array([c['C'] for c in cells], dtype=float)
        if np.any(sizes <= 0) or np.any(measured_comm <= 0):
            raise ValueError('invalid communication calibration')
        a = np.column_stack((np.ones(len(cells)), sizes))
        weights = 1 / measured_comm
        curve = np.linalg.lstsq(a * weights[:, None], measured_comm * weights, rcond=None)[0]
        predicted_comm = np.maximum(a @ curve, 1e-6)
        x = np.stack([_features(c, pred, self.split_feature) for c, pred in zip(cells, predicted_comm)])
        y = np.array([math.log(c['stock_' + self.mode] / c['dx_' + self.mode]) for c in cells])
        mu, sd = x.mean(axis=0), x.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        z = (x - mu) / sd
        yc = y - y.mean()
        beta = np.linalg.solve(z.T @ z + self.ridge * np.eye(z.shape[1]), z.T @ yc)
        return mu, sd, float(y.mean()), beta, curve, self.split_feature

    @staticmethod
    def _predict_op(params, c):
        mu, sd, intercept, beta, curve, split_feature = params
        comm_ms = max(float(curve[0] + curve[1] * float(c['B'])), 1e-6)
        return float(intercept + ((_features(c, comm_ms, split_feature) - mu) / sd) @ beta)

    def fit(self, cells):
        cs = [c for c in cells if not c.get('anomaly') and
              'stock_' + self.mode in c and 'dx_' + self.mode in c]
        self.models, self.guards = {}, {}
        for op in sorted({c['op'] for c in cs}):
            group = [c for c in cs if c['op'] == op]
            families = sorted({c['model'] for c in group})
            if len(families) < 3:
                continue
            residuals = []
            for held in families:
                train = [c for c in group if c['model'] != held]
                p = self._fit_op(train)
                for c in group:
                    if c['model'] == held:
                        truth = math.log(c['stock_' + self.mode] / c['dx_' + self.mode])
                        residuals.append(self._predict_op(p, c) - truth)
            self.models[op] = self._fit_op(group)
            self.guards[op] = max(0.0, float(np.quantile(residuals, self.quantile)))
        return self

    def predict(self, cell):
        if cell['op'] not in self.models:
            raise ValueError('insufficient calibration for operation ' + str(cell['op']))
        value = self._predict_op(self.models[cell['op']], cell)
        return {'log_speedup': value, 'guard': self.guards[cell['op']],
                'lower_log_speedup': value - self.guards[cell['op']]}

    def choose(self, cell):
        p = self.predict(cell)
        # A lower bound on log(stock / method) must exceed the required margin.
        return ('dx' if p['lower_log_speedup'] > -math.log1p(-self.margin) else 'stock'), p
