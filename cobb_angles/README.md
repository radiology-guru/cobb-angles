# A transfer learning approach to Cobb angle estimation
This codebase is largely based on the [reproduce-chexnet](https://github.com/jrzech/reproduce-chexnet) python repository. The pre-trained DenseNet-121 checkpoint was used for subsequent fine-tuning for spine landmark localization and cobb angle estimation. The code for processing spine x-rays, along with a modified training loop, is provided here. RetinaNet object detection relied on the following codebase: [pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet).

## Training a model to predict landmark coordinates
1. Clone the [reproduce-chexnet](https://github.com/jrzech/reproduce-chexnet) repository.
2. Download the pre-trained DenseNet-121 [model checkpoint](https://github.com/jrzech/reproduce-chexnet/blob/master/pretrained/checkpoint) trained on [CheXpert Data](https://stanfordmlgroup.github.io/competitions/chexpert/) dataset.
3. Download 609 spinal anterior-posterior x-ray images used in the challenge from [SpineWeb](http://spineweb.digitalimaginggroup.ca/Index.php?n=Main.Datasets#Dataset_16.3A_609_spinal_anterior-posterior_x-ray_images).
4. Preprocess the data using ```create_datasets.py```.
5. Run ```python train.py``` by setting ```PATH_TO_IMAGES``` and ```PATH_TO_MODEL``` to where the preprocessed spine images and the downloaded mdoel checkpoint are stored, respectively. 
