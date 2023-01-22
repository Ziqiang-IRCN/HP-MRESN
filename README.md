# HP-MRESN
The source code for "Multi-Reservoir Echo State Networks with Hodrick–Prescott Filter for Nonlinear Time-Series Prediction".
## Tested environment
* python 3.9.7  
* numpy 1.19.5  
* pytorch 1.10.2
* scikit-learn 1.0.2  
* statsmodels 0.12.2  
* h5py 3.6.0  
* pandas 1.3.5  
* tqdm 4.63.0
## Program
You can use HP-MRESN.py to reproduce the results of HP-MRESN by using searched hyperparameters reported in the paper.
## Datasets
Datasets corresponding to file names in `./data`
* Cardio [HK Cardio dataset] (https://github.com/RPcb/ARNN/tree/master/Data/HK%20hospital%20admission): `hk_data_v1.mat`
* Sunspot [SILSO] (https://www.sidc.be/silso/datafiles): `original_sunspot.txt`
* DMTMA [daily maximum temperature in Melbourne airport] (http://www.bom.gov.au/climate/averages/tables/cw_086282.shtml): `Melboune_airport_MTandsolar2006to2019.csv`
* Electricity: `https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv`
* Bike [Bike Sharing Dataset] (https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset): `bike_hour.csv`
* Traffic speed [Urban Traffic Speed Dataset of Guangzhou, China)] (https://zenodo.org/record/1205229#.Y800L-xBwbl): `roadid_61.csv`
* MGS17: `mackey_glass_t17_original.txt`
* NoisyMGS17: `mackey_glass_t17_normal_noised_mean=0,sigma=0.1.txt`
* Laser: `santa-fe-laser.txt`
* NoisyLaser: `laser_normal_noised_mean=0,sigma=0.2.txt`

Other details about datasets (such as the length, the partition, and the citation of each dataset) can be found in our paper.
## Citiation
> citiation is coming soon.
