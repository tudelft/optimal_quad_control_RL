# optimal_quad_control_RL
Reinforcement learning for time optimal end-to-end quadcopter control

https://arxiv.org/abs/2504.21586

**Conda**

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

conda env list
conda create --name gncnet
conda activate gncnet
conda env update -n gncnet --file quad.yaml
```
