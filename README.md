# CS7CS4 Computer Vision Project
## Protection Against Unwanted Facial Recognition

The face masking model used in this project is from "Towards Face Encryption by Generating Adversarial Identity Masks."

	@InProceedings{Yang_2021_ICCV,
	    author    = {Yang, Xiao and Dong, Yinpeng and Pang, Tianyu and Su, Hang and Zhu, Jun and Chen, Yuefeng and Xue, Hui},
	    title     = {Towards Face Encryption by Generating Adversarial Identity Masks},
	    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
	    month     = {October},
	    year      = {2021},
	    pages     = {3897-3907}
	}

Link to original repos: https://github.com/ShawnXYang/TIP-IM
### Data Preparation

- Download [LFW](https://hal.inria.fr/file/index/docid/321923/filename/Huang_long_eccv2008-lfw.pdf). Put the LFW dataset in `data`; add `pairs.txt` to `data/lfw/`.
- Download the CelebA dataset to `data/`; partition into `train`, `validation`, and `test`.

### Crafting Identity Masks

This project is tested under the following environment settings:
- OS: Ubuntu 18.04.3
- GPU: Geforce 2080 Ti or Tesla P100
- Cuda: 9.0, Cudnn: v7.03
- Python: 3.5.2
- TensorFlow: 1.9.0
- PyTorch: >= 1.4.0
- Torchvision: >= 0.4.0

Example usage:

```
python generate_all_image_lists.py --dir data/celeba/test/ --n 100 --save_filename subset_100.txt
python generate_all_image_lists.py --dir data/lfw/ --n 10 --save_filename subset_10.txt
python run.py --input_images data/celeba/test/subset_100.txt --output output/celeba/test/
```


