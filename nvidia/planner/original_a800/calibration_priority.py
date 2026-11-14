"""Rank candidate shape probes by expected *net* horizon-level utility.

Inputs are caller-provided forecasts, not certified probability bounds. This
does not execute probes or use target measurements as a free prediction input.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Candidate:
    name: str
    predicted_delta_ms: float
    uncertainty_ms: float
    occurrences_per_step: int
    probe_cost_ms: float

    def __post_init__(self):
        if not isinstance(self.name,str) or not self.name:raise ValueError('name required')
        for value in (self.predicted_delta_ms,self.uncertainty_ms,self.probe_cost_ms):
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):raise ValueError('finite numeric values required')
        if self.uncertainty_ms<0 or self.probe_cost_ms<0 or not isinstance(self.occurrences_per_step,int) or isinstance(self.occurrences_per_step,bool) or self.occurrences_per_step<1:
            raise ValueError('invalid uncertainty, cost, or frequency')


def rank_probes(candidates,*,remaining_steps):
    if not isinstance(remaining_steps,int) or isinstance(remaining_steps,bool) or remaining_steps<1:
        raise ValueError('positive integer horizon required')
    rows=[]
    for c in candidates:
        # Symmetric normal uncertainty around estimated overlap-stock latency.
        # E[min(0,X)] compared with min(0,E[X]) is the value of knowing X
        # perfectly. Actual sampling is noisy, so this is an optimistic bound
        # only under the chosen normal model, not a real-world guarantee.
        mu,sigma=c.predicted_delta_ms,c.uncertainty_ms
        if sigma==0:perfect_info=0.
        else:
            z=mu/sigma;phi=math.exp(-.5*z*z)/math.sqrt(2*math.pi)
            cdf=.5*math.erfc(-z/math.sqrt(2))
            e_min=mu*(1-cdf)-sigma*phi
            perfect_info=max(0.,min(0.,mu)-e_min)
        gross=perfect_info*c.occurrences_per_step*remaining_steps
        rows.append({'name':c.name,'expected_perfect_information_ms_per_call':perfect_info,
                     'optimistic_gross_ms':gross,'probe_cost_ms':c.probe_cost_ms,
                     'optimistic_net_ms':gross-c.probe_cost_ms})
    return sorted(rows,key=lambda r:(-r['optimistic_net_ms'],r['name']))
