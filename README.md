# [TCTrack]

## 1. Environment setup
This code has been tested on Ubuntu 18.04, Python 3.8.3, Pytorch 0.7.0/1.6.0, CUDA 10.2.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

## 2. Test
Download [pretrained model](https://mega.nz/file/N0EiiLjB#59wv0Yeovl7KnrMRLTLx2tAKTjpdo6AwcCuXuMgObc4) and put it into `tools/snapshot` directory.

Download testing datasets and put them into `test_dataset` directory. 

```bash 
python test.py                                \
	--dataset UAV10fps                      \ # 
    --dataset_name
	--snapshot snapshot/general_model.pth  # tracker_name
```
The testing result will be saved in the `results/dataset_name/tracker_name` directory.

**Note:** The results of TCTrack can be [downloaded](https://pan.baidu.com/s/1-V4JbKvmVPm0aOKWTOQtyQ) (code:kh3e).

## 3. Train

### Prepare training datasets

Download the datasets：
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [Lasot](https://paperswithcode.com/dataset/lasot)
* [GOT-10K](http://got-10k.aitestunion.com/downloads)


**Note:** `train_dataset/dataset_name/readme.md` has listed detailed operations about how to generate training datasets.


### Train a model
To train the TCTrack model, run `train.py` with the desired configs:

```bash
cd tools
python train.py
```

## 4. Evaluation
If you want to evaluate the results of our tracker, please put those results into  `results` directory.
```
python eval.py 	                          \
	--tracker_path ./results          \ # result path
	--dataset UAV10fps                  \ # dataset_name
	--tracker_prefix 'general_model'   # tracker_name
```


**Note:** The code is implemented based on [pysot](https://github.com/STVIR/pysot). We would like to express our sincere thanks to the contributors.