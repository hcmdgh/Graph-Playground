from .imports import * 
from .util import * 

__all__ = ['sbert_embedding']

_sbert_model_dict: dict[str, nn.Module] = dict() 


def sbert_embedding(text_list: list[str],
                    model_name: str = 'all-mpnet-base-v2', 
                    batch_size: int = 64,
                    use_tqdm: bool = True) -> FloatArray:
    from sentence_transformers import SentenceTransformer
    
    if model_name not in _sbert_model_dict:
        _sbert_model_dict[model_name] = SentenceTransformer(model_name, device=get_device())
        
    bert_model = _sbert_model_dict[model_name]
    
    out = bert_model.encode(
        text_list, 
        batch_size = batch_size,
        show_progress_bar = use_tqdm,
    )
    
    return out 
