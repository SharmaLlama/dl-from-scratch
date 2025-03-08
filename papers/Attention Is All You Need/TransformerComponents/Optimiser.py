class WarmupAdamOpt:
    def __init__(self, model_size, warmup, optimiser):
        self.optimiser = optimiser
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0
    
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimiser'}
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict) 
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimiser.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimiser.step()
        
    def rate(self, step = None):
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))) 