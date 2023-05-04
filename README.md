# Pointnet2.PyTorch

* PyTorch implementation of [PointNet++](https://arxiv.org/abs/1706.02413) based on [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).
* Faster than the original codes by re-implementing the CUDA operations. 

## Modified to adapt New CUDA Version
### Step 1
```cpp
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)
```
to

```cpp
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
```
### Step 2
comment following codes in every .cpp file：
```cpp
extern THCState *state;
cudaStream_t stream = THCState_getCurrentStream(state);
```

### Step 3
replace every kernel function parameter `stream` with `c10::cuda::getCurrentCUDAStream()`
just like
before:
```cpp
int ball_query_wrapper_fast(int b, int n, int m, float radius, int nsample, 
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    int *idx = idx_tensor.data<int>();
//    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
//    cudaStream_t stream = THCState_getCurrentStream(state);
    ball_query_kernel_launcher_fast(b, n, m, radius, nsample, new_xyz, xyz, idx, stream);
    return 1;
}
```
after:
```cpp
int ball_query_wrapper_fast(int b, int n, int m, float radius, int nsample, 
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    int *idx = idx_tensor.data<int>();
//    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
//    cudaStream_t stream = THCState_getCurrentStream(state);
    ball_query_kernel_launcher_fast(b, n, m, radius, nsample, new_xyz, xyz, idx, c10::cuda::getCurrentCUDAStream());
    return 1;
}
```


## Installation
### Requirements
* Linux (tested on Ubuntu 18.04)
* Python 3.6+
* PyTorch 1.0

CUDA11.0+Pytorch1.7.0 PASS!

### Install 
Install this library by running the following command:

```shell
cd pointnet2
python setup.py install
cd ../
```

## Examples
Here I provide a simple example to use this library in the task of KITTI ourdoor foreground point cloud segmentation, and you could refer to the paper [PointRCNN](https://arxiv.org/abs/1812.04244) for the details of task description and foreground label generation.

1. Download the training data from [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) website and organize the downloaded files as follows:
```
Pointnet2.PyTorch
├── pointnet2
├── tools
│   ├──data
│   │  ├── KITTI
│   │  │   ├── ImageSets
│   │  │   ├── object
│   │  │   │   ├──training
│   │  │   │      ├──calib & velodyne & label_2 & image_2
│   │  train_and_eval.py
```

2. Run the following command to train and evaluate:
```shell
cd tools
python train_and_eval.py --batch_size 8 --epochs 100 --ckpt_save_interval 2 
```



## Project using this repo:
* [PointRCNN](https://github.com/sshaoshuai/PointRCNN): 3D object detector from raw point cloud.

## Acknowledgement
* [charlesq34/pointnet2](https://github.com/charlesq34/pointnet2): Paper author and official code repo.
* [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch): Initial work of PyTorch implementation of PointNet++. 
