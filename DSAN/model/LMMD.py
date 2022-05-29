from util import * 


class LMMD(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 kernel_mul: float = 2.0, 
                 kernel_num: int = 5, 
                 fix_sigma: Optional[float] = None):
        super().__init__()
        
        self.num_classes = num_classes
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, 
                        src_batch: FloatTensor, 
                        tgt_batch: FloatTensor) -> FloatTensor:
        # src_batch: [src_batch_size x emb_dim]
        # tgt_batch: [tgt_batch_size x emb_dim]
        
        # num_samples: src_batch_size + tgt_batch_size
        num_samples = src_batch.shape[0] + tgt_batch.shape[0]

        # total: [(src_batch_size + tgt_batch_size) x emb_dim]
        total = torch.cat([src_batch, tgt_batch], dim=0)

        # total0: [num_samples x num_samples x emb_dim]
        # total1: [num_samples x num_samples x emb_dim]
        total0 = total.unsqueeze(dim=0).expand(total.shape[0], total.shape[0], total.shape[1])
        total1 = total.unsqueeze(dim=1).expand(total.shape[0], total.shape[0], total.shape[1])

        # L2_distance: [num_samples x num_samples]
        L2_distance = torch.sum((total0 - total1) ** 2, dim=2)

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.detach()) / (num_samples ** 2 - num_samples)
            
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i)
                          for i in range(self.kernel_num)]
        kernel_val = torch.stack([torch.exp(-L2_distance / bandwidth_temp)
                                  for bandwidth_temp in bandwidth_list])

        # kernels: [num_samples x num_samples]
        kernels = torch.sum(kernel_val, dim=0)

        return kernels 

    def calc_loss(self, 
                  src_batch: FloatTensor, 
                  tgt_batch: FloatTensor, 
                  src_label: IntTensor, 
                  tgt_label: FloatTensor) -> FloatTensor:
        # src_batch: float[batch_size x emb_dim]
        # tgt_batch: float[batch_size x emb_dim]
        # src_label: int[batch_size]
        # tgt_label: float[batch_size x num_classes]
        
        batch_size = src_batch.shape[0]
        
        weight_ss, weight_tt, weight_st = self.calc_weight(src_label=src_label.cpu().numpy(),
                                                          tgt_label=tgt_label.cpu().numpy())
        
        # weight_ss, weight_tt, weight_st: [batch_size x batch_size]
        weight_ss = to_device(torch.from_numpy(weight_ss))
        weight_tt = to_device(torch.from_numpy(weight_tt))
        weight_st = to_device(torch.from_numpy(weight_st))

        # kernels: [(batch_size * 2) x (batch_size * 2)]
        kernels = self.gaussian_kernel(src_batch=src_batch, 
                                       tgt_batch=tgt_batch)
        
        if torch.sum(torch.isnan(kernels)) > 0:
            logging.warn('Kernels have NAN value!')
            return to_device(torch.tensor(0.))
        
        # SS, TT, ST: [batch_size x batch_size]
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        # loss: float 
        loss = torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        
        return loss

    def convert_to_onehot(self, label: IntArray) -> FloatArray:
        # label: int[batch_size]
        
        # out: [batch_size x num_classes]
        out = np.eye(self.num_classes)[label]
        
        return out 

    def calc_weight(self, 
                    src_label: IntArray, 
                    tgt_label: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        # src_label: int[batch_size]
        # tgt_label: float[batch_size x num_classes]
        
        batch_size = len(src_label) 
        
        # src_vec_label: [batch_size x num_classes]
        src_vec_label = self.convert_to_onehot(src_label)

        # src_sum: [1 x num_classes]
        src_sum = np.sum(src_vec_label, axis=0, keepdims=True)
        src_sum[src_sum == 0.] = 100.  # TODO 100魔数？
        
        # src_vec_label: [batch_size x num_classes]
        src_vec_label = src_vec_label / src_sum

        # tgt_scalar_label: int[batch_size]
        tgt_scalar_label = np.argmax(tgt_label, axis=1) 

        tgt_vec_label = tgt_label 

        # tgt_sum: [1 x num_classes]
        tgt_sum = np.sum(tgt_vec_label, axis=0, keepdims=True)
        tgt_sum[tgt_sum == 0.] = 100. 

        # tgt_vec_label: [batch_size x num_classes]
        tgt_vec_label = tgt_vec_label / tgt_sum

        # mask_arr: [batch_size x num_classes]
        index = list(set(src_label) & set(tgt_scalar_label))
        mask_arr = np.zeros((batch_size, self.num_classes))
        mask_arr[:, index] = 1
        
        tgt_vec_label = tgt_vec_label * mask_arr
        src_vec_label = src_vec_label * mask_arr

        # weight_ss: [batch_size x batch_size]
        weight_ss = np.matmul(src_vec_label, src_vec_label.T)
        weight_tt = np.matmul(tgt_vec_label, tgt_vec_label.T)
        weight_st = np.matmul(src_vec_label, tgt_vec_label.T)

        index_len = len(index)

        if index_len > 0:
            weight_ss = weight_ss / index_len
            weight_tt = weight_tt / index_len
            weight_st = weight_st / index_len
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])

        weight_ss = weight_ss.astype(np.float32)
        weight_tt = weight_tt.astype(np.float32)
        weight_st = weight_st.astype(np.float32)
            
        # weight_ss, weight_tt, weight_st: [batch_size x batch_size]
        return weight_ss, weight_tt, weight_st 
