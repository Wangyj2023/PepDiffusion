import os
import math
import time


import torch

import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.bert.modeling_bert import BertConfig
import torch.distributed as dist




def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
class DModel(nn.Module):
    def __init__(self):
        super(DModel, self).__init__()
        self.input_channels = 128
        self.model_channels = 127
        self.out_channels = 128
        self.num_class = 2
        self.drop_out = 0.1
        self.max_length = 127
        config = BertConfig()
        config.hidden_dropout_prob = self.drop_out
        config.max_position_embeddings = self.max_length
        config.num_attention_heads = 6
        config.num_hidden_layers = 3

        self.control_embedding = nn.Embedding(self.num_class,config.hidden_size)
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels,config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size,config.hidden_size)
        )

        self.input_up_proj = nn.Sequential(nn.Linear(self.input_channels, config.hidden_size),nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        self.input_transformers = BertEncoder(config)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),nn.Tanh(), nn.Linear(config.hidden_size, self.out_channels))
    def forward(self,x,timesteps,control = None):
        emb = self.time_embed(timestep_embedding(timesteps,self.model_channels))
        # print(emb.shape)
        if control != None:
            emb = emb + self.control_embedding(control)
        # print(emb.shape)
        # print(x.shape)
        emb_x = self.input_up_proj(x)
        # print(emb_x.shape)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length]
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        h = self.output_down_proj(input_trans_hidden_states)
        h = h.type(x.dtype)
        return h
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
class MyDiffusion():
    def __init__(self,num_timesteps,betas):
        self.betas = betas
        alphas = 1.0 - betas
        self.num_timesteps = num_timesteps
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    def _scale_timesteps(self, t):
        return t.float() * (1000.0 / self.num_timesteps)
    def get_x_start(self, x_start_mean, std):
        noise = torch.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        return (x_start_mean + std * noise)
    def cal_loss(self,model,x_0,t,condition = None,rank = None):
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,torch.tensor([0]).to(x_0.device),x_0.shape)
        x_start = self.get_x_start(x_0, std)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_start, t, noise=noise)
        model_output = model(x_t, self._scale_timesteps(t),condition)
        target = x_start
        assert model_output.shape == target.shape == x_start.shape
        mes_loss = mean_flat((target - model_output) ** 2)
        t0_mask = (t == 0)
        t0_loss = mean_flat((x_0 - model_output) ** 2)
        mes_loss = torch.where(t0_mask, t0_loss, mes_loss)
        return mes_loss
    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    def p_mean_variance(
        self, model, x, t, clip_denoised=True, cond = None
    ):
        B, C = x.size(0), x.size(-1)
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), cond)
        model_variance, model_log_variance = (self.posterior_variance,self.posterior_log_variance_clipped)
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
        def process_xstart(x):
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        pred_xstart = process_xstart(model_output)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    def p_sample(self, model, x, t, clip_denoised=False, cond = None):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            cond=cond
        )
        noise = torch.randn_like(x)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"],'greedy_mean':out["mean"], 'out':out}
    def p_sample_loop_progressive_infill(self,model,shape,cond = None):
        assert isinstance(shape, (tuple, list))
        img = torch.randn(*shape).cuda()
        indices = list(range(self.num_timesteps))[::-1]
        for i in indices:
            t = torch.tensor([i] * shape[0]).cuda()
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised = False,
                    cond=cond
                )
                img = out["sample"]
                out["sample"] = img
                yield out







def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)
def get_named_beta_schedule(num_diffusion_timesteps):
    return betas_for_alpha_bar(num_diffusion_timesteps,lambda t: 1-np.sqrt(t + 0.0001),)
def create_gaussian_diffusion(steps=2000):
    betas = get_named_beta_schedule(steps)
    return MyDiffusion(num_timesteps = steps, betas = betas)
class MyDataset_nocond(Dataset):
    def __init__(self,floder_path):
        self.floder_path = floder_path
        self.file_list = os.listdir(floder_path)
        # self.data = np.load(floder_path).astype(np.float32) * 25
        # self.f = 10000
        # if train:
        #     self.data = np.load(floder_path)[:200000]
        # else:
        #     self.data = np.load(floder_path)[:20000]
    def __len__(self):
        return len(self.file_list) 
        # * 2024
        # return len(self.data)
    def __getitem__(self, item):
        return np.load(f"{self.floder_path}/{self.file_list[item]}") * 25
        # f = item // 2024
        # i = item % 2024
        # if self.f != f:
        #     self.f = f
        #     self.np = np.load(f"{self.floder_path}/{self.file_list[f]}").astype(np.float32) 
        #     return self.np[i] * 25
        # else:
        #     return self.np[i] * 25
        # return self.data[item]
class UniformSampler():
    def __init__(self, num_timesteps=2000):
        self._weights = np.ones([num_timesteps])
    def weights(self):
        return self._weights
    def sample(self, batch_size,device_id):
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device_id)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device_id)
        return indices, weights

class MyDataset_cond(Dataset):
    def __init__(self,pos_path,neg_path):
        # AMP 0 UnAMP 1
        self.pos_path = pos_path
        self.pos_list = os.listdir(pos_path)

        self.neg_path = neg_path
        self.neg_list = os.listdir(neg_path)

        self.pos_num = len(self.pos_list)
        self.neg_num = len(self.neg_list)
    def __len__(self):
        return self.pos_num + self.neg_num
    def __getitem__(self, item):
        if item < self.pos_num:
            return np.load(f"{self.pos_path}/{self.pos_list[item]}") * 25,0
        else:
            return np.load(f"{self.neg_path}/{self.neg_list[item-self.pos_num]}") * 25,1
def decode_mols(encoded_tensors, org_dict):
    mols = []
    for i in range(encoded_tensors.shape[0]):
        encoded_tensor = encoded_tensors.cpu().numpy()[i,:] - 1
        mol_string = ''
        for i in range(encoded_tensor.shape[0]):
            idx = encoded_tensor[i]
            if org_dict[idx] == '<end>':
                break
            elif org_dict[idx] == '_':
                pass
            else:
                mol_string += org_dict[idx]
        mols.append(mol_string)
    return mols






if __name__ == "__main1__":
    import numpy as np
    # np.set_printoptions(linewidth=300)
    # a = np.load(f"./memory/train/mem_2.npy")
    # print(a[0])
    config = BertConfig()
    print(config)

if __name__ == "__main111__":

        model = DModel().cuda()
        a=torch.randn([128,127,128]).cuda()
        sampler = UniformSampler(1000)
        time_steps, weights = sampler.sample(batch_size=a.shape[0], device_id=a.device)
        diffusion = create_gaussian_diffusion(1000)
        loss=diffusion.cal_loss(model,a,time_steps)
        asd=time.time()
        print(loss)
        print(time.time()-asd)

if __name__ == "__main_nocondition__":
    import torch.distributed as dist
    from torch.distributed import init_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    import argparse


    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",             default=0.0001,                 type=float  )
    parser.add_argument("--save_path",      default="./model_mulgpu",       type=str    )
    parser.add_argument("--epoch",          default=100,                    type=int    )
    parser.add_argument("--batch_size",     default=512,                    type=int    )
    parser.add_argument("--num_steps",      default=500,                    type=int    )
    parser.add_argument("--shuffle",        default=False,                  type=bool   )
    parser.add_argument("--num_workers",    default=8,                      type=int    )
    parser.add_argument("--pin_memory",     default=True,                   type=bool   )
    parser.add_argument("--drop_last",      default=True,                   type=bool   )
    args = parser.parse_args()

    init_process_group(backend='nccl')
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    
    num_steps = args.num_steps
    batch_size = args.batch_size
    shuffle = args.shuffle
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    drop_last=args.drop_last
    learn_rate = args.lr
    num_epochs = args.epoch
    save_path = args.save_path

    model = DModel().to(device_id)
    model = DDP(model,device_ids=[device_id],find_unused_parameters=True)

    diffusion = create_gaussian_diffusion(num_steps)
    sampler = UniformSampler(num_steps)

    train_data = MyDataset_nocond(f"./train.npy",True)
    val_data   = MyDataset_nocond(f"./val.npy",False)
    # train_data = MyDataset_nocond(f"./memory/train",True)
    # val_data   = MyDataset_nocond(f"./memory/val",False)
    train_sample = DistributedSampler(train_data)
    val_sample = DistributedSampler(val_data)
    train_loader = DataLoader(
        train_data,   
        batch_size=batch_size,
        sampler=train_sample,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
        )
    val_loader   = DataLoader(
        val_data,     
        batch_size=int(batch_size/10),
        sampler=val_sample,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
        )

    if rank == 0:
        os.makedirs(f"{save_path}",exist_ok=True)
        os.makedirs(f"{save_path}/model",exist_ok=True)
        log_filepath = f"{save_path}/train.log"
        try:
            f = open(log_filepath, 'r')
            f.close()
            already_wrote = True
        except FileNotFoundError:
            already_wrote = False
        log_file = open(log_filepath, 'a')
        if not already_wrote:
            log_file.write('epoch,batch_idx,data_type,tot_loss,run_time\n')
        log_file.close()
    opt = AdamW(model.parameters(), lr = learn_rate)
    if rank == 0:
        print(f"Train data : {len(train_data)}")
        print(f"Val data   : {len(val_data)}")
    best_loss = 1000
    for epoch in range(num_epochs):
        train_sample.set_epoch(epoch)
        train_loss = 0
        train_batch = 0
        start_time = time.time()
        model.train()
        start_batch_time = time.time()
        if rank == 0:
            train_run = tqdm(train_loader)
        else:
            train_run = train_loader
        start_time_abatch = time.time()
        for i,data in enumerate(train_run):
            # if rank == 0:
            #     print(f"start rank {rank} run time : {time.time()-start_time_abatch}")
            b = data.shape[0]
            # if rank == 0:
            #     print(f"start to device run time : {time.time()-start_time_abatch}")
            data = data.to(device_id)
            # if rank == 0:
            #     print(f"start zero_grad run time : {time.time()-start_time_abatch}")
            opt.zero_grad()
            # if rank == 0:
            #     print(f"start sample run time : {time.time()-start_time_abatch}")
            time_steps, weights = sampler.sample(batch_size=data.shape[0], device_id=device_id)
            # if rank == 0:
            #     print(f"start time_steps to cuda run time : {time.time()-start_time_abatch}")
            time_steps = time_steps.to(device_id)
            # if rank == 0:
            #     print(f"start cal loss run time : {time.time()-start_time_abatch}")
            loss = diffusion.cal_loss(model,data,time_steps,rank=rank).mean()
            # if rank == 0:
            #     print(f"start loss backward run time : {time.time()-start_time_abatch}")
            loss.backward()
            # if rank == 0:
            #     print(f"start opt step run time : {time.time()-start_time_abatch}")
            opt.step()
            train_loss += loss.item()
            if rank == 0:
                batch_time = time.time() - start_batch_time
                start_batch_time = time.time()
                log_file = open(log_filepath, 'a')
                log_file.write('{},{},{},{},{}\n'.format(
                                                        epoch,
                                                        i, 
                                                        'train',
                                                        loss.item(),
                                                        batch_time
                                                        ))
                log_file.close()
            train_batch += 1
        train_loss = train_loss/train_batch

        model.eval()
        val_loss = 0
        vak_batch = 0
        # start_time = time.time()
        if rank == 0:
            val_run = tqdm(val_loader)
        else:
            val_run = val_loader
        for i, vdata in enumerate(val_run):
            seq = vdata
            seq = seq.to(device_id)
            time_steps, weights = sampler.sample(batch_size=seq.shape[0],device_id=device_id)
            time_steps = time_steps.to(device_id)
            loss = diffusion.cal_loss(model,seq,time_steps).mean()
            val_loss += loss.item()
            if rank == 0:
                batch_time = time.time() - start_batch_time
                start_batch_time = time.time()
                log_file = open(log_filepath, 'a')
                log_file.write('{},{},{},{},{}\n'.format(
                                                        epoch,
                                                        i, 
                                                        'val',
                                                        loss.item(),
                                                        batch_time
                                                        ))
                log_file.close()
            vak_batch += 1
        val_loss = val_loss/vak_batch
        if rank == 0:
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.module.state_dict(),f"{save_path}/best_model.pth")
            if epoch % 1 == 0:
                torch.save(model.module.state_dict(),f"{save_path}/model/model_{epoch}_{train_loss}_{val_loss}.pth")
            print(f"########################################################################################")
            print(f"Epoch {epoch}, Train loss : {round(train_loss,5)}, Test loss : {round(val_loss,5)}")
            print(f"Run time : {round(time.time()-start_time,5)} s")










# if __name__ == "__main__":
#     model = DModel().cuda()
#     a = torch.randn([128,127,128]).cuda()
#     b = torch.randint(0,500,[128]).cuda()
#     c = torch.randint(0,1,[128]).cuda()
#     print(model(a,b,c).shape)


if __name__ == "__main__":
    import torch.distributed as dist
    from torch.distributed import init_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import random_split
    import argparse

    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",             default=0.0001,                 type=float  )
    parser.add_argument("--save_path",      default="./model_mulgpu",       type=str    )
    parser.add_argument("--epoch",          default=300,                    type=int    )
    parser.add_argument("--batch_size",     default=512,                    type=int    )
    parser.add_argument("--num_steps",      default=500,                    type=int    )
    parser.add_argument("--shuffle",        default=False,                  type=bool   )
    parser.add_argument("--num_workers",    default=8,                      type=int    )
    parser.add_argument("--pin_memory",     default=True,                   type=bool   )
    parser.add_argument("--drop_last",      default=True,                   type=bool   )
    args = parser.parse_args()

    init_process_group(backend='nccl')
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    
    num_steps = args.num_steps
    batch_size = args.batch_size
    shuffle = args.shuffle
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    drop_last=args.drop_last
    learn_rate = args.lr
    num_epochs = args.epoch
    save_path = args.save_path

    model = DModel().to(device_id)
    model.load_state_dict(torch.load(f"./best_model.pth"))
    model = DDP(model,device_ids=[device_id],find_unused_parameters=True)

    diffusion = create_gaussian_diffusion(num_steps)
    sampler = UniformSampler(num_steps)

    all_data = MyDataset_cond(f"./pos_data.npy",f"./neg_data.npy",True)
    length = len(all_data)
    train_size, validate_size = int(0.8 * length), length - int(0.8 * length)
    train_data, val_data = random_split(all_data,[train_size,validate_size],generator=torch.Generator().manual_seed(42))
    # val_data   = MyDataset_nocond(f"./val.npy",False)
    # train_data = MyDataset_nocond(f"./memory/train",True)
    # val_data   = MyDataset_nocond(f"./memory/val",False)
    train_sample = DistributedSampler(train_data)
    val_sample = DistributedSampler(val_data)
    train_loader = DataLoader(
        train_data,   
        batch_size=batch_size,
        sampler=train_sample,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
        )
    val_loader   = DataLoader(
        val_data,     
        batch_size=int(batch_size/10),
        sampler=val_sample,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
        )

    if rank == 0:
        os.makedirs(f"{save_path}",exist_ok=True)
        os.makedirs(f"{save_path}/model",exist_ok=True)
        log_filepath = f"{save_path}/train.log"
        try:
            f = open(log_filepath, 'r')
            f.close()
            already_wrote = True
        except FileNotFoundError:
            already_wrote = False
        log_file = open(log_filepath, 'a')
        if not already_wrote:
            log_file.write('epoch,batch_idx,data_type,tot_loss,run_time\n')
        log_file.close()
    opt = AdamW(model.parameters(), lr = learn_rate)
    if rank == 0:
        print(f"Train data : {len(train_data)}")
        print(f"Val data   : {len(val_data)}")
    best_loss = 1000
    for epoch in range(num_epochs):
        train_sample.set_epoch(epoch)
        train_loss = 0
        train_batch = 0
        start_time = time.time()
        model.train()
        start_batch_time = time.time()
        if rank == 0:
            train_run = tqdm(train_loader)
        else:
            train_run = train_loader
        start_time_abatch = time.time()
        for i,data__ in enumerate(train_run):
            data,label = data__
            b = data.shape[0]
            data = data.to(device_id)
            label = label.to(device_id)
            opt.zero_grad()
            time_steps, weights = sampler.sample(batch_size=data.shape[0], device_id=device_id)
            time_steps = time_steps.to(device_id)
            loss = diffusion.cal_loss(model,data,time_steps,label).mean()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            if rank == 0:
                batch_time = time.time() - start_batch_time
                start_batch_time = time.time()
                log_file = open(log_filepath, 'a')
                log_file.write('{},{},{},{},{}\n'.format(
                                                        epoch,
                                                        i, 
                                                        'train',
                                                        loss.item(),
                                                        batch_time
                                                        ))
                log_file.close()
            train_batch += 1
        train_loss = train_loss/train_batch
        model.eval()
        val_loss = 0
        vak_batch = 0
        if rank == 0:
            val_run = tqdm(val_loader)
        else:
            val_run = val_loader
        for i, data__ in enumerate(val_run):
            vdata,label = data__
            seq = vdata
            seq = seq.to(device_id)
            label = label.to(device_id)
            time_steps, weights = sampler.sample(batch_size=seq.shape[0],device_id=device_id)
            time_steps = time_steps.to(device_id)
            loss = diffusion.cal_loss(model,seq,time_steps,label).mean()
            val_loss += loss.item()
            if rank == 0:
                batch_time = time.time() - start_batch_time
                start_batch_time = time.time()
                log_file = open(log_filepath, 'a')
                log_file.write('{},{},{},{},{}\n'.format(
                                                        epoch,
                                                        i, 
                                                        'val',
                                                        loss.item(),
                                                        batch_time
                                                        ))
                log_file.close()
            vak_batch += 1
        val_loss = val_loss/vak_batch
        if rank == 0:
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.module.state_dict(),f"{save_path}/best_model.pth")
            if epoch % 1 == 0:
                torch.save(model.module.state_dict(),f"{save_path}/model/model_{epoch}_{train_loss}_{val_loss}.pth")
            print(f"########################################################################################")
            print(f"Epoch {epoch}, Train loss : {round(train_loss,5)}, Test loss : {round(val_loss,5)}")
            print(f"Run time : {round(time.time()-start_time,5)} s")