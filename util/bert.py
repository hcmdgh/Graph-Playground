# from .imports import * 
# from .util import * 
# import transformers

# _bert_model_dict: dict[str, nn.Module] = dict() 


# class SciBERTScivocabUncased(nn.Module):
#     def __init__(self):
#         super().__init__() 
        
#         self.device = get_device()
        
#         self.tokenizer = transformers.AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
#         self.model = transformers.AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        
#         self.to(self.device)
        
#     @torch.no_grad()
#     def forward(self, text_batch: list[str]) -> FloatArray:
#         inputs = self.tokenizer(
#             text_batch, 
#             return_tensors = "pt",
#             padding = True,
#             truncation = True,
#             max_length = 512,
#         ).to(self.device)
        
#         outputs = self.model(**inputs)

#         last_hidden_states = outputs.last_hidden_state

#         out = last_hidden_states.mean(dim=1)
#         out = out.detach().cpu().numpy() 
        
#         return out 


# class XLNetLargeCased(nn.Module):
#     def __init__(self):
#         super().__init__() 
        
#         self.device = get_device()
        
#         self.tokenizer = transformers.XLNetTokenizer.from_pretrained('xlnet-large-cased')
#         self.model = transformers.XLNetModel.from_pretrained('xlnet-large-cased')
        
#         self.to(self.device)
        
#     @torch.no_grad()
#     def forward(self, text_batch: list[str]) -> FloatArray:
#         inputs = self.tokenizer(
#             text_batch, 
#             return_tensors = "pt",
#             padding = True,
#             truncation = True,
#             max_length = 512,
#         ).to(self.device)
        
#         outputs = self.model(**inputs)

#         last_hidden_states = outputs.last_hidden_state

#         out = last_hidden_states.mean(dim=1)
#         out = out.detach().cpu().numpy() 
        
#         return out 


# class XLNetBaseCased(nn.Module):
#     def __init__(self):
#         super().__init__() 
        
#         self.device = get_device()
        
#         self.tokenizer = transformers.XLNetTokenizer.from_pretrained('xlnet-base-cased')
#         self.model = transformers.XLNetModel.from_pretrained('xlnet-base-cased')
        
#         self.to(self.device)
        
#     @torch.no_grad()
#     def forward(self, text_batch: list[str]) -> FloatArray:
#         inputs = self.tokenizer(
#             text_batch, 
#             return_tensors = "pt",
#             padding = True,
#             truncation = True,
#             max_length = 512,
#         ).to(self.device)
        
#         outputs = self.model(**inputs)

#         last_hidden_states = outputs.last_hidden_state

#         out = last_hidden_states.mean(dim=1)
#         out = out.detach().cpu().numpy() 
        
#         return out 
        
        
# def bert_embedding(model_name: str, 
#                    text_list: list[str],
#                    batch_size: int = 64,
#                    use_tqdm: bool = True) -> FloatArray:
#     if model_name not in _bert_model_dict:
#         if model_name == 'xlnet-large-cased':
#             _bert_model_dict[model_name] = XLNetLargeCased()
#         elif model_name == 'xlnet-base-cased':
#             _bert_model_dict[model_name] = XLNetBaseCased()
#         elif model_name == 'scibert_scivocab_uncased':
#             _bert_model_dict[model_name] = SciBERTScivocabUncased()
#         else:
#             raise AssertionError 
        
#     bert_model = _bert_model_dict[model_name]
    
#     batch_output_list = [] 
    
#     for i in tqdm(range(0, len(text_list), batch_size), desc='bert_embedding', disable=not use_tqdm):
#         text_batch = text_list[i: i + batch_size]
        
#         out = bert_model(text_batch)
        
#         batch_output_list.append(out)
    
#     out = np.concatenate(batch_output_list, axis=0)
    
#     return out 
