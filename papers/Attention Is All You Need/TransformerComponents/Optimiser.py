class WarmupAdamOpt:
    def __init__(self, model_size, warmup, optimiser):
        self.optimiser = optimiser
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0
        self.not_printed = True
    def state_dict(self):
        return {
                "warmup_opt" : {key: value for key, value in self.__dict__.items() if key != 'optimiser'},
                "optimiser" : self.optimiser.state_dict()
                }
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict['warmup_opt']) 
        self.optimiser.load_state_dict(state_dict['optimiser'])

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
        if step ** (-0.5) < step * self.warmup ** (-1.5) and self.not_printed:
            print("warm up done")
            self.not_printed = False

        return (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))) 
