# ARFN
11 ([Paper]())

## Prerequisites


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
This code borrows from [TomBERT]([https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/jefferyYu/TomBERT)).
