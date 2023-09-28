# Artificial intelligence by using a latent diffusion model enables the generation of diverse and potent antimicrobial peptides to combat antibiotic resistance



## Requirement

1. python==3.8.18
1. pytorch==1.13.1
1. torchvision==0.14.1
1. torchaudio==0.13.1
1. transformer==4.33.2
1. pandas==2.0.3
1. nltk==3.8.1

## Train VAE
Unzip the VAE_Train.zip file under data before training VAE.
```
conda env create -f PepDiffusion.yaml
conda activate PepDiffusion
```

## Train VAE

```
torchrun --nproc_per_node=8 main.py --work TransVAE 
```

## Pre-train Diffusion

```
python main.py --work GetMem_nc --vae_model_path {VAE_path}
python combine_mem.py
torchrun --nproc_per_node=8 main.py --work LatentDiffusion_nocondition 
```

## Fine-tune Diffusion

```
python main.py --work GetMem_c  --vae_model_path {VAE_path}
python combine_mem_c.py
torchrun --nproc_per_node=8 main.py --work LatentDiffusion_condition 
```

## Generation AMPs

```
python main.py --work Generate --Generate_VAE_model_path {VAE_path} --Generate_Diffusion_model_path {Diffusion_path}
```

