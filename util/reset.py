from .imports import * 


def deep_reset_parameters(value: Any):
    if isinstance(value, nn.Linear):
        reset_linear_parameters(value)
    else:
        empty = True 
        
        for child in value.children() if hasattr(value, 'children') else []:
            deep_reset_parameters(child)
            empty = False 
            
        assert not empty 

            
def reset_linear_parameters(linear: nn.Linear,
                            method: str = 'kaiming_uniform'):
    if method == 'glorot':
        glorot_(linear.weight)
    elif method == 'uniform':
        bound = 1.0 / math.sqrt(linear.weight.size(-1))
        nn.init.uniform_(linear.weight.data, -bound, bound)
    elif method == 'kaiming_uniform':
        kaiming_uniform_(linear.weight, 
                         fan=linear.in_features,
                         a=math.sqrt(5))
    else:
        raise AssertionError 

    if linear.bias is not None:
        uniform_(linear.in_features, linear.bias)
            
            
def glorot_(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        empty = True 
        
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot_(v)
            empty = False 
            
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot_(v)
            empty = False 
            
        assert not empty 
            
            
def kaiming_uniform_(value: Any, fan: int, a: float):
    if isinstance(value, Tensor):
        bound = math.sqrt(6 / ((1 + a ** 2) * fan))
        value.data.uniform_(-bound, bound)
    else:
        empty = True 

        for v in value.parameters() if hasattr(value, 'parameters') else []:
            kaiming_uniform_(v, fan, a)
            empty = False 
            
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            kaiming_uniform_(v, fan, a)
            empty = False 
            
        assert not empty 
            
            
def uniform_(size: int, value: Any):
    if isinstance(value, Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)
    else:
        empty = True 
        
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            uniform_(size, v)
            empty = False 
            
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            uniform_(size, v)
            empty = False 
            
        assert not empty 


def constant_(value: Any, fill_value: float):
    if isinstance(value, Tensor):
        value.data.fill_(fill_value)
    else:
        empty = True 
        
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            constant_(v, fill_value)
            empty = False 
            
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            constant_(v, fill_value)
            empty = False 
            
        assert not empty 


def zeros_(value: Any):
    constant_(value, 0.)


def ones_(tensor: Any):
    constant_(tensor, 1.)


def normal_(value: Any, mean: float, std: float):
    if isinstance(value, Tensor):
        value.data.normal_(mean, std)
    else:
        empty = True 
        
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            normal_(v, mean, std)
            empty = False 
            
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            normal_(v, mean, std)
            empty = False 
            
        assert not empty 
