"""Simple amortization arithmetic for explicitly measured calibration cost."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Amortization:
    calibration_ms: float
    saving_ms_per_step: float

    def __post_init__(self):
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x)
               for x in (self.calibration_ms,self.saving_ms_per_step)) or self.calibration_ms<0:
            raise ValueError('finite calibration cost and step saving required')

    def evaluate(self, steps):
        if not isinstance(steps,int) or isinstance(steps,bool) or steps<0:
            raise ValueError('steps must be a nonnegative integer')
        net=steps*self.saving_ms_per_step-self.calibration_ms
        threshold=(math.ceil(self.calibration_ms/self.saving_ms_per_step)
                   if self.saving_ms_per_step>0 else None)
        return {'steps':steps,'gross_saving_ms':steps*self.saving_ms_per_step,
                'calibration_ms':self.calibration_ms,'net_saving_ms':net,
                'breakeven_steps':threshold,
                'warning':'input estimates, no confidence or wall-time guarantee'}
