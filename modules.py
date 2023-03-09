from torch import nn

class ForwardModuleList(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.list_of_heads = nn.ModuleList(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return [head(*args, **kwargs) for head in self.list_of_heads]

