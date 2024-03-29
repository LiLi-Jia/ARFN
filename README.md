# ARFN


Affective Region Recognition and Fusion Network for Target-Level Multimodal Sentiment Classification ([Paper](https://ieeexplore.ieee.org/abstract/document/10014688))

![model](https://github.com/LiLi-Jia/ARFN/assets/44886362/c8dccdaf-c6bb-4c6b-abc1-5a6e997e2084)


## Requirement

- Python 3.7
- NVIDIA GPU + CUDA cuDNN
- PyTorch 1.9.0

## Prepare
1. Following [TomBERT](https://github.com/jefferyYu/TomBERT) section: Download tweet images and set up image path.
2. Train a visual sentiment classification model based on the ResNet-152. This datasets is provided by [Yang J[1]](http://47.105.62.179:8081/sentiment_web/datasets/LDL.tar.gz).
3. The Object Score and IoU Score in the image are obtained using Yolov5. Also, the Senti_score is obtained using the pre-trained model from step 2.

## Run
1. search and replace relevant paths
   res_path = 'feature path'

2. train the model
```
python train_and_test.py --bert_model=bert-base-uncased
--output_dir=./outupt
--data_dir=./data/twitter2015 or 2017
--task_name=twitter2015 or 2017
--do_train
```
3. test the model
```
python train_and_test.py --bert_model=bert-base-uncased
--output_dir=./outupt
--data_dir=./data/twitter2015 or 2017
--task_name=twitter2015 or 2017
--do_eval
```


## Citation
If you find this useful for your research, please use the following.

```
@article{jia2023affective,
  title={Affective Region Recognition and Fusion Network for Target-Level Multimodal Sentiment Classification},
  author={Jia, Li and Ma, Tinghuai and Rong, Huan and Al-Nabhan, Najla},
  journal={IEEE Transactions on Emerging Topics in Computing},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgments
This code borrows from [TomBERT](https://github.com/jefferyYu/TomBERT).

[1] Yang J, Sun M, Sun X. Learning visual sentiment distributions via augmented conditional probability neural network[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2017, 31(1).
