from .imports import * 

_device = torch.device('cpu')


def init_log(log_path: Optional[str] = './log.log',
             stdout: bool = True):
    handlers = []
             
    if log_path:
        handlers.append(logging.FileHandler(log_path, 'w', encoding='utf-8'))
    
    if stdout:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        format = '%(asctime)s [%(levelname)s] %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
        handlers = handlers,
        level = logging.INFO,
    )
    
    
def log_multi(**kwargs):
    seps = []
    
    for key, value in kwargs.items():
        if isinstance(value, float):
            seps.append(f"{key}: {value:.4f}") 
        else:
            seps.append(f"{key}: {value}")
            
    text = ', '.join(seps)
    
    logging.info(text)
            

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
    
    dgl.seed(seed)
    dgl.random.seed(seed)
    
    
def auto_set_device(use_gpu: bool = True) -> torch.device:
    global _device 

    if not use_gpu:
        _device = torch.device('cpu')
        return _device
    
    exe_res = os.popen('gpustat --json').read() 
    
    state_dict = json.loads(exe_res)
    
    gpu_infos = [] 
    
    for gpu_entry in state_dict['gpus']:
        gpu_id = int(gpu_entry['index'])
        used_mem = int(gpu_entry['memory.used'])

        gpu_infos.append((used_mem, gpu_id))
    
    gpu_infos.sort()
    
    _device = torch.device(f'cuda:{gpu_infos[0][1]}')
    
    return _device 


def set_device(device_name: str):
    global _device
    _device = torch.device(device_name)
    
    
def get_device() -> torch.device:
    return _device 


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
