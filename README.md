# Error Prediction for Adaptive Depth Sampling
## Into
This repository contains the python implementation of "Error Prediction for Adaptive Depth Sampling" research by Ilya Tcenov and Prof. Guy Gilboa, Electrical Engineering Faculty, Technion, Israel.

A paper will be available soon.

## Setup
Clone repo and activate conda environment as follows:
<pre>
git clone https://github.com/yliats1/adaptive_depth_sampling.git
cd adaptive_depth_sampling
conda env create --file environment.yml
conda activate py37
</pre>

In this work we use two networks:

* [FusionNet](https://github.com/wvangansbeke/Sparse-Depth-Completion) - a depth completion network by Van Gansbeke et al.

* [Pix2PixHD](https://github.com/NVIDIA/pix2pixHD) - an image-to-image translation network by Wang et al.


## Directory Structure
This section describes the dataset directory structure that is required to use our code. Note that RGB (images) and GT (dense depth ground truth maps) directories are required to be placed under train, validation, and test directories, while ImpMaps and LiDAR directories will be created during the execution of our algorithm.
<pre>
root
├──train
|  ├──RGB       # required
|  |  ├──1000000.png
|  |  ├──1000001.png
|  |  └──...
|  ├──GT        # required
|  |  ├──1000000.png
|  |  ├──1000001.png
|  |  └──...     
|  ├──ImpMaps   # calculated in this process
|  |  ├──impmaps1
|  |  |  ├──1000000.png
|  |  |  ├──1000001.png
|  |  |  └──...
|  |  ├──impmaps2
|  |  |  ├──1000000.png
|  |  |  ├──1000001.png
|  |  |  └──...
|  |  └──...
|  └──LiDAR     # calculated in this process
|     ├──lidar1
|     |  ├──1000000.png
|     |  ├──1000001.png
|     |  └──...
|     ├──lidar2
|     |  ├──1000000.png
|     |  ├──1000001.png
|     |  └──...
|     └──...
├──validation
|  └... # same composition as train
└──test
   └... # same composition as train
</pre>

