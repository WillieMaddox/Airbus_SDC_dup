# Setup and Install
Use the following guide to reproduce my results and run the jupyter notebooks.  

## Clone this repo
Clone this repo and cd into the project.
```shell script
$ git clone https://github.com/WillieMaddox/Airbus_SDC_dup.git
$ cd Airbus_SDC_dup
```

## Setup the environment
I use virtualenv along with virtualenvwrapper to setup an isolated python environment:
```shell script
$ mkvirtualenv --python=python3.6 Airbus_SDC_dup
```
You should now see your command prompt prefixed with `(sdcdup)` indicating you are in the virtualenv.

If using conda, you can instead try using the make script to create your environment.
```shell script
$ make create_environment 
```
I do not use conda so I haven't had a chance to verify if this works.

## Install requirements
From the root of the project install all requirements.
```shell script
(Airbus_SDC_dup) $ pip install -r requirements.txt
```
or
```shell script
$ make requirements
```

## Download the data

The dataset for this project is hosted on Kaggle. [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/overview)
You'll need to sign in with your Kaggle username.  If you don't have an account, it's free to sign up.

You can extract the dataset to wherever you like.  I extracted it to `data/raw/train_768`

```
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- This README.
├── data
│   ├── raw            <- Data dump from the Airbu_SDC Kaggle competition goes in here.
│       ├── train_768  <- The images from train_v2.zip go in here.
│           ├── 00003e153.jpg
│           ├── 0001124c7.jpg
│           ├── ...
│           └── ffffe97f3.jpg
│       └── train_ship_segmentations_v2.csv <- The run length encoded ship labels.
│   ├── ...
├── ...
```

## Preprocess tiles and interim data

Once the raw dataset has been downloaded and extracted, run the image preprocessing scripts.

First generate the 256 x 256 image tiles:
```shell script
$ make data 
``` 
Note: The `data/processed/train_256` folder takes up ??? GB of disk space. It takes approx 30 min to run on my dev system.  YMMV.

Next generate the image feature metadata:
```shell script
$ make features
```
Note: The `data/interim` folder takes up ??? GB of disk space. It takes approx 2 hrs to run on my dev system.  YMMV.

The newly generated files will be placed into the interim and processed directories.
Once complete, your directory structure should look like the following:
```
├── ...
├── data
│   ├── raw
│   ├── interim
│       ├── image_bmh.pkl
│       ├── image_cmh.pkl
│       ├── image_sol.pkl
│       ├── image_md5.pkl
│       ├── image_shp.pkl
│       ├── matches_bmh_0.90234375_1.csv
│       ├── matches_bmh_0.90234375_2.csv
│       ├── matches_bmh_0.90234375_3.csv
│       ├── matches_bmh_0.90234375_4.csv
│       ├── matches_bmh_0.90234375_6.csv
│       ├── matches_bmh_0.90234375_9.csv
│       ├── overlap_bmh_1.pkl
│       ├── overlap_bmh_2.pkl
│       ├── overlap_bmh_3.pkl
│       ├── overlap_bmh_4.pkl
│       ├── overlap_bmh_6.pkl
│       ├── overlap_bmh_9.pkl
│       ├── overlap_cmh_1.pkl
│       ├── overlap_cmh_2.pkl
│       ├── ...
│       └── overlap_shp_9.pkl
│   ├── processed
│       └── train_256
│           ├── 00003e153_0.jpg
│           ├── 00003e153_1.jpg
│           ├── 00003e153_2.jpg
│           ├── 00003e153_3.jpg
│           ├── 00003e153_4.jpg
│           ├── 00003e153_5.jpg
│           ├── 00003e153_6.jpg
│           ├── 00003e153_7.jpg
│           ├── 00003e153_8.jpg
│           ├── 0001124c7_0.jpg
│           ├── ...
│           └── ffffe97f3_8.jpg
│   ├── ...
├── ...
```

