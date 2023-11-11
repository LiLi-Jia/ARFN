# ARFN
11 ([Paper]())

## Requirement


## Prepare
1. Following [TomBERT](https://github.com/jefferyYu/TomBERT) section: Download tweet images and set up image path.
2. Train a visual sentiment classification model based on the ResNet-152 network. This data sets is provided by http://47.105.62.179:8081/sentiment_web/datasets/LDL.tar.gz.
3. 

## Run
1. train the model
```
python train.py --bert_model=bert-base-uncased
--output_dir=./user_outuptdir
--data_dir=./absa_data/twitter2015
--task_name=twitter2015
--do_train
```
2. test the model
```
python test_and_save.py --bert_model=bert-base-uncased
--output_dir=./user_outuptdir
--data_dir=./absa_data/twitter2015
--task_name=twitter2015
--do_eval
```


## Citation
If you find this useful for your research, please use the following.

```
@ARTICLE{ 
}
```

## Acknowledgments
This code borrows from [TomBERT](https://github.com/jefferyYu/TomBERT).
[1] Yang J, Sun M, Sun X. Learning visual sentiment distributions via augmented conditional probability neural network[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2017, 31(1).
