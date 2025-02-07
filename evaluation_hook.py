import torch

class ForwardHook:
    def __init__(self, module):
        self.out = []
        self.inp = []
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        with torch.no_grad():
            out = output[0].detach()
            inp = input[0].detach()
            self.out.append(out.squeeze().cpu())
            self.inp.append(inp.squeeze().cpu())

    def close(self):
        self.hook.remove()
        del self.out
        del self.inp