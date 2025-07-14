# Installation 
教程来源：https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html


### isaacsim 环境配置
```bash
conda create -n lab python=3.10
conda activate lab

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

pip install --upgrade pip
# 40系显卡
pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com 

## 安装检查
isaacsim 

## 脚本启动
isaacsim isaacsim4.5.0/apps/isaacsim.exp.full.kit
```
#### 1. 基础环境配置
```bash
git clone git@github.com:isaac-sim/IsaacLab.git
sudo apt install cmake build-essential

cd IsaacLab
./isaaclab.sh --install # or "./isaaclab.sh -i"

# 安装检查
# Option 1: Using the isaaclab.sh executable
# note: this works for both the bundled python and the virtual environment
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
# Option 2: Using python in your virtual environment
python scripts/tutorials/00_sim/create_empty.py
```
#### 2. vscode 环境配置
* 基本设置
```bash
python IsaacLab/.vscode/tools/setup_vscode.py # 将自动增加launch.json、settings.json 配置 vscode
```
源码的函数定义可以索引
![alt text](icon/image-1.png)

* debug 插件安装

![alt text](icon/image.png)
isaacsim app 启用 vscode
![alt text](icon/image-2.png)
![alt text](icon/image-3.png)


#### 3. 本地加载 `assets`
```bash
# 离线下载 assets [ https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html ]
cd resource
wget -c https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-1%404.5.0-rc.36%2Brelease.19112.f59b3005.zip
wget -c https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-2%404.5.0-rc.36%2Brelease.19112.f59b3005.zip
wget -c https://download.isaacsim.omniverse.nvidia.com/isaac-sim-assets-3%404.5.0-rc.36%2Brelease.19112.f59b3005.zip
unzip *.zip   
```
修改 `source/isaaclab/isaaclab/utils/assets.py` 下的 `NUCLEUS_ASSET_ROOT_DIR`
```python
# NUCLEUS_ASSET_ROOT_DIR = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
NUCLEUS_ASSET_ROOT_DIR = ("/data1/wangmingyue/wmy/isaaclab/resource/Assets/Isaac/4.5")
```
