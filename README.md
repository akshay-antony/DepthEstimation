# DepthEstimation
Estimate depth from monocular image <br />
**Results from NYU-V2 dataset on MobileNet-v2 encoder** <br />
**Obtained SSIM of .97 and PSNR of 49.52**  <br />   <br /> 
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **1.input rgb image   &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; 
2.predicted depth image    &nbsp; &nbsp;   &nbsp; &nbsp; 
3.ground truth**<br />
<p float="left">
  <img src="https://github.com/akshay-antony/DepthEstimation/blob/main/results/100_color.png" width="200" />
  <img src="https://github.com/akshay-antony/DepthEstimation/blob/main/results/100_pred.png" width="200" /> 
  <img src="https://github.com/akshay-antony/DepthEstimation/blob/main/results/100_target.png" width="200" />
</p>

**Results from NYU-V2 dataset on VGG-19 encoder** <br /> 
**Obtained SSIM of .95 and PSNR of 43.14**  <br />   <br /> 
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **1.input rgb image   &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; 
2.predicted depth image   &nbsp; &nbsp; &nbsp;  &nbsp;
3.ground truth**<br />
<p float="left">
  <img src="https://github.com/akshay-antony/DepthEstimation/blob/main/results/100_color.png" width="200" />
  <img src="https://github.com/akshay-antony/DepthEstimation/blob/main/results/100_pred_vgg.png" width="200" /> 
  <img src="https://github.com/akshay-antony/DepthEstimation/blob/main/results/100_target.png" width="200" />
</p>

**Results from KITTI dataset center cropped by 224 by 224 on MobileNet-v2 encoder** <br /> <br />
**Obtained SSIM of .84 and PSNR of 31.32**  <br />   <br /> 

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **1.input rgb image  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 
&nbsp; 
2.predicted depth image  &nbsp; &nbsp;  &nbsp; &nbsp; 
3.ground truth**<br />
<p float="left">
  <img src="https://github.com/akshay-antony/DepthEstimation/blob/main/results/kitti_color.png" width="200" />
  <img src="https://github.com/akshay-antony/DepthEstimation/blob/main/results/kitti_pred.PNG" width="200" /> 
  <img src="https://github.com/akshay-antony/DepthEstimation/blob/main/results/target_kitti.PNG" width="200" />
</p>
**Reference** <br />
@article{Alhashim2018, <br />
  author    = {Ibraheem Alhashim and Peter Wonka}, <br />
  title     = {High Quality Monocular Depth Estimation via Transfer Learning}, <br />
  journal   = {arXiv e-prints}, <br />
  volume    = {abs/1812.11941}, <br />
  year      = {2018}, <br />
  url       = {https://arxiv.org/abs/1812.11941}, <br />
  eid       = {arXiv:1812.11941}, <br />
  eprint    = {1812.11941} <br />
} <br />
