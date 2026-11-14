"""A non-reenabling guard on already selected DX, based on completed chunks.

This deliberately contains no GPU or distributed operations. A caller must
measure every chunk, arrange identical rank decisions and account for costs.
"""
import math


class LatencyGuard:
    def __init__(self, stock_reference_ms, *, margin=.02, consecutive=2, initial='dx'):
        if (isinstance(stock_reference_ms,bool) or not isinstance(stock_reference_ms,(int,float)) or
                not math.isfinite(stock_reference_ms) or stock_reference_ms<=0):raise ValueError('invalid reference')
        if not math.isfinite(margin) or margin<0 or not isinstance(consecutive,int) or isinstance(consecutive,bool) or consecutive<1:
            raise ValueError('invalid guard')
        if initial not in ('stock','dx'):raise ValueError('invalid initial choice')
        self.reference=stock_reference_ms;self.margin=margin;self.consecutive=consecutive;self.choice=initial;self.streak=0

    def observe(self,choice,latency_ms):
        if choice!=self.choice:raise ValueError('observed path does not match current choice')
        if isinstance(latency_ms,bool) or not isinstance(latency_ms,(int,float)) or not math.isfinite(latency_ms) or latency_ms<=0:
            raise ValueError('invalid chunk latency')
        if self.choice=='dx':
            self.streak=self.streak+1 if latency_ms>self.reference*(1+self.margin) else 0
            if self.streak>=self.consecutive:self.choice='stock'
        return self.choice
