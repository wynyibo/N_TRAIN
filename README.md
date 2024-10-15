# 基于eBPF内核的神经网络实时入侵检测与防御

安装Pytorch环境

```c
sudo python3  pip install torch
```

安装 `sklearn`（即 `scikit-learn`）库，脚本中进行数据预处理

```c
sudo pip3 install scikit-learn --user
```

安装numpy库

```c
sudo pip3 install  numpy
```

安装scipy库

```c
sudo pip3 install scipy
```

环境搭建过程中需要确保numpy、scipy、scikit-learn库一致

报错如下：

```c
UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 2.0.0
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.0 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.
```

由于不同库的版本冲突引起的，具体是 `scipy` 需要一个较低版本的 `numpy`，当前的系统中安装了 `numpy 2.0.0`，这导致了 `scipy` 和 `sklearn` 无法正常工作。

版本进行降级

```c
pip3 install numpy==1.24.0 --user
```

如下命令确保版本一致

```c
python3 -m pip show numpy
python3 -m pip show scipy
python3 -m pip show scikit-learn
```

~/NN-eBPF下`make`

~/Desktop/NN-eBPF/src下运行

```c
python3 mlp_train.py ../dataset/reproduction-xdp-data.npy ../dataset/reproduction-xdp-label.npy
```

