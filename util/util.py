from .imports import * 

_device = None 


def init_log(filename: Optional[str] = None):
    if filename:
        handlers = [
            logging.FileHandler(filename, 'w', encoding='utf-8'),
            logging.StreamHandler(),
        ]
    else:
        handlers = [logging.StreamHandler()] 
    
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers,
                        level=logging.INFO)


def get_set_mapping(_set: set[Any]) -> tuple[list[Any], dict[Any, int]]:
    idx2val = list(_set)
    val2idx = { v: i for i, v in enumerate(idx2val) }
    
    return idx2val, val2idx


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def seed_all(seed: Optional[int]):
    if not seed:
        return 
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def set_device(device_name: str):
    global _device
    _device = torch.device(device_name)


def to_device(obj: Any) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = to_device(value)

        return obj 
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = to_device(value)
            
        return obj 
    else:
        return obj.to(device=_device)


def clone_module(module: nn.Module,
                 cnt: int) -> nn.ModuleList:
    module_list = nn.ModuleList() 
    
    for _ in range(cnt):
        module.reset_parameters()
        module_list.append(module)
        
        module = copy.deepcopy(module)
        
    return module_list


def load_yaml(file_path: str) -> DotDict:
    with open(file_path, 'r', encoding='utf-8') as fp:
        obj = yaml.safe_load(fp)
        
    assert isinstance(obj, dict)
    
    def to_dotdict(_dict: dict) -> DotDict:
        for key, value in _dict.items():
            if isinstance(value, dict):
                _dict[key] = to_dotdict(value)
                
        return DotDict(_dict)
    
    return to_dotdict(obj)
