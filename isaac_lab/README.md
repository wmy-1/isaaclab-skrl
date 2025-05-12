# IsaacLab 安装 
教程来源：https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html

```bash
conda create -n isaaclab python=3.10
conda activate isaaclab

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

pip install --upgrade pip
# 40系显卡
pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com 

## 安装检查
isaacsim 

## 脚本启动
isaacsim isaacsim4.5.0/apps/isaacsim.exp.full.kit
```