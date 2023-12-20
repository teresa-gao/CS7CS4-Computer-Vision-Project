# Towards Face Encryption by Generating Adversarial Identity Masks


## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 18.04.3
- GPU: Geforce 2080 Ti or Tesla P100
- Cuda: 9.0, Cudnn: v7.03
- Python: 3.5.2
- TensorFlow: 1.9.0
- PyTorch: >= 1.4.0
- Torchvision: >= 0.4.0

### Data Preparation
- Please download [LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf), Put LFW dataset in `data`; add `pairs.txt` to `data/lfw/`.
- Put some specific target images to `targets`

### Crafting Identity Masks
```
python generate_all_image_lists.py --dir data/celeba/test/ --n 100 --save_filename subset_100.txt
python generate_all_image_lists.py --dir data/lfw/ --n 10 --save_filename subset_10.txt
python run.py --input_images data/celeba/test/subset_100.txt --output output/celeba/test/
```

### Citation
If you benefit from our work in your research, please consider to cite the following paper:

	@InProceedings{Yang_2021_ICCV,
	    author    = {Yang, Xiao and Dong, Yinpeng and Pang, Tianyu and Su, Hang and Zhu, Jun and Chen, Yuefeng and Xue, Hui},
	    title     = {Towards Face Encryption by Generating Adversarial Identity Masks},
	    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
	    month     = {October},
	    year      = {2021},
	    pages     = {3897-3907}
	}
