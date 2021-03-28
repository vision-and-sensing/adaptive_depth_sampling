# adaptive_depth_sampling
Error Prediction for Adaptive Depth Sampling

## Directory Structure
txt
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
