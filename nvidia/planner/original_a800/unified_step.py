"""Connect fixed-method route predictions to calibrated step selection.

Callers provide per-invocation stock latency and measured/calibrated whole-step
interaction terms. Unsupported, uncalibrated or incomplete inputs stay stock.
The model is diagnostic; this does not install adapters or claim runtime gains.
"""
import math
import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Optional

from .backend_family import Environment, Signature, MeasurementKey, Occurrence, OperatorEstimate, StepCalibration, plan
from .orthogonal_predictor import ContextRequirement, Route
from .predictor_registry import PredictorRegistry


@dataclass(frozen=True)
class Callsite:
    site: str
    phase: str
    route: Route
    shape: tuple  # M,N,K
    count: int
    stock_ms: float
    capability: ContextRequirement
    uncertainty_ms: float
    gemm_ms: Optional[float] = None
    comm_ms: Optional[float] = None


def _signature(route, env):
    config = (('config_family',route.config_family), ('memory',route.memory),
              ('sync',route.synchronization), ('timing_mode',route.timing_mode), ('warmup',str(route.warmup)))
    return Signature(env, route.compute, route.transport, route.direction, config, route.actual_path)


def _inventory_digest(calls):
    from dataclasses import asdict
    value=[asdict(c) for c in calls]
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def choose_step(registry: PredictorRegistry, calls, workload: str, *, common_ms=None,
                pairs_ms: Optional[Mapping[tuple[str,str],float]]=None,
                calibration_provenance=None, margin_ms=0.0, site_residual_ms=None):
    calls=tuple(calls)
    if not calls or not workload:
        raise ValueError('workload and callsites required')
    r=calls[0].route
    environment_fields=lambda route:(route.method,route.gpu,route.topology,route.tp,route.dtype,route.versions)
    if any(environment_fields(c.route)!=environment_fields(r) for c in calls):
        raise ValueError('one method and matching environment required')
    env=Environment(r.method,'hybrid','adapter_declared',r.gpu,r.topology,r.tp,r.dtype,r.versions)
    occurrences=[]; estimates={}; signatures=set(); diagnostics=[]
    for c in calls:
        if c.capability.route!=c.route or c.capability.shape!=c.shape:
            raise ValueError('capability route or shape mismatch')
        if not math.isfinite(c.stock_ms) or c.stock_ms<=0 or not math.isfinite(c.uncertainty_ms) or c.uncertainty_ms<0:
            raise ValueError('invalid timing or uncertainty')
        sig=_signature(c.route,env);signatures.add(sig)
        key=MeasurementKey(sig,c.phase,c.shape,c.route.timing_mode)
        result=registry.predict(c.route,c.shape,capability=c.capability,gemm_ms=c.gemm_ms,comm_ms=c.comm_ms)
        gain=result['predicted_log_speedup']
        supported=(gain is not None and result['choice'] in ('stock',c.route.method)
                   and result.get('estimate_valid',True)
                   and not result.get('forbid_step_selection',False))
        predicted=c.stock_ms*math.exp(-gain) if supported else c.stock_ms
        estimate=OperatorEstimate(key,c.stock_ms,predicted,c.uncertainty_ms,'route model',supported)
        if key in estimates and estimates[key]!=estimate:
            raise ValueError('conflicting estimates for same callsite key')
        estimates[key]=estimate;occurrences.append(Occurrence(c.site,key,c.count))
        diagnostics.append({'site':c.site,'phase':c.phase,'model_status':result['status'],
                            'predicted_log_speedup':gain,
                            'forbid_step_selection':result.get('forbid_step_selection',False)})
    if common_ms is None:
        if pairs_ms is not None or calibration_provenance is not None or site_residual_ms is not None:
            raise ValueError('step calibration must include common overhead')
        step=None
    else:
        if pairs_ms is None or not calibration_provenance:
            raise ValueError('explicit interaction map and step calibration provenance required')
        step=StepCalibration(env,workload,tuple(sorted(signatures,key=repr)),common_ms,pairs_ms,calibration_provenance,
                             site_residual_ms or {})
    output=plan(env,workload,occurrences,list(estimates.values()),step,margin_ms=margin_ms)
    output['diagnostics']=diagnostics
    output['call_inventory_sha256']=_inventory_digest(calls)
    output['step_calibrated']=step is not None
    if step is not None:
        record={'workload':workload,'common_ms':common_ms,
                'pairs_ms':{'|'.join(k):v for k,v in sorted(pairs_ms.items())},
                'provenance':calibration_provenance}
        if site_residual_ms is not None:
            record['site_residual_ms']=site_residual_ms
        output['calibration_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return output
