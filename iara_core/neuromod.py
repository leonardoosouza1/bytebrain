import math, torch

class Neuromod:
    """Hormônios COMPUTADOS por surpresa. Coeficientes vindos de código já validado:
    _plast() do iara_cerebro.py e a bateria de brain_hormones.py."""
    def __init__(self, ema_beta=0.98):
        self.ema = None; self.beta = ema_beta
        self.dop, self.cort, self.cur, self.fad = 0.2, 0.1, 0.3, 0.0

    def update(self, loss, dt):
        self.ema = loss if self.ema is None else self.beta*self.ema + (1-self.beta)*loss
        salience = loss / (self.ema + 1e-9)                 # >1 = surpreendente
        relax = math.exp(-dt / 3600.0)                      # o tempo relaxa os hormônios
        self.dop  = 0.2 + (self.dop  - 0.2)*0.92*relax + 0.4*max(0.0, min(salience-1, 3.0))/3
        self.cort = 0.1 + (self.cort - 0.1)*0.90*relax + 0.1*max(0.0, min(salience-1.5, 2.0))/2
        self.cur  = 0.3 + (self.cur  - 0.3)*0.97*relax
        self.fad  = min(1.0, self.fad*relax + 0.002)
        return self.as_dict()

    def dormir(self):
        self.fad = 0.0; self.cort = max(0.0, self.cort-0.2); self.dop = 0.2 + (self.dop-0.2)*0.9

    def precisa_dormir(self): return self.fad >= 0.5
    def plasticity(self):     return 0.06 + 0.55*self.dop - 0.15*self.fad
    def as_dict(self):        return {"dop":self.dop,"cort":self.cort,"cur":self.cur,"fad":self.fad}
    def vector(self):         return torch.tensor([self.dop,self.cort,self.cur,self.fad],dtype=torch.float32)
