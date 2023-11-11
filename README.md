# ARFN
11 ([Paper]())

## Requirement


## Prepare
1.following [TomBERT](https://github.com/jefferyYu/TomBERT). section Download tweet images and set up image path.

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