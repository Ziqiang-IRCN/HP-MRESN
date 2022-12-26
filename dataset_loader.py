import numpy as np
import pandas
import h5py
from sklearn.preprocessing import MinMaxScaler

def input_target_spliter(data:np.ndarray,delay:int):
    input = data[:-delay]
    target = data[delay:]
    return input, target

class timeseries_loader():
    original_series = None
    @property
    def __get_original_sunspot(self):
        original_series = np.loadtxt('./data/original_sunspot.txt')
        return original_series
    @property
    def __get_bike_dataset(self):
        sharing_bike = pandas.read_csv('./data/bike_hour.csv')
        original_series = np.array(sharing_bike['cnt'])[3000:8001]
        return original_series
    @property
    def __get_cardio_dataset(self):
        with h5py.File('./data/hk_data_v1.mat', 'r') as file:
            original_series = np.squeeze(np.array(file['cardio']))
        return original_series
    @property
    def __get_traffic_speed_dataset(self):
        traffic_speed = pandas.read_csv('./data/roadid_61.csv')
        original_series = np.array(traffic_speed['speed'])
        return original_series
    @property
    def __get_maximal_Melboune_airport_temperature_dataset(self):
        data = pandas.read_csv('./data/Melboune_airport_MTandsolar2006to2019.csv')
        original_series = data['MaximumtemperatureDegreeC'].values
        return original_series
    @property
    def __get_opsd_germany_daily_electricity_dataset(self):
        url='https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv'
        data = pandas.read_csv(url,sep=",")
        original_series = data['Consumption'].values/24.0
        return original_series
    @property
    def __get_MGS17_dataset(self):
        MGS17 = np.loadtxt('./data/mackey_glass_t17_original.txt')
        original_series = MGS17[:6000]
        return original_series
    @property
    def __get_Noisy_MGS17_dataset(self):
        Noisy_MGS17 = np.loadtxt('./data/mackey_glass_t17_normal_noised_mean=0,sigma=0.1.txt')
        original_series = Noisy_MGS17[:6000]
        return original_series
    @property
    def __get_laser_dataset(self):
        laser = np.loadtxt('./data/santa-fe-laser.txt')
        original_series = laser[:5000]
        return original_series
    @property
    def __get_Noisy_laser_dataset(self):
        Noisy_laser = np.loadtxt('./data/laser_normal_noised_mean=0,sigma=0.2.txt')
        original_series = Noisy_laser[:5000]
        return original_series

    def __init__(self,dataset_name:str = None,validation=True,delay:int=1):
        self.loading_data = None
        self.dataset_name = dataset_name
        self.validation = validation
        self.delay=delay
        self.washout_set_index = None
        self.train_set_index = None
        self.validation_set_index = None
        self.test_set_index = None
        if dataset_name == 'Cardio':
            self.loading_data = self.__get_cardio_dataset
        elif dataset_name == 'Sunspot':
            self.loading_data = self.__get_original_sunspot
        elif dataset_name == 'Bike':
            self.loading_data = self.__get_bike_dataset
        elif dataset_name == 'Traffic':
            self.loading_data = self.__get_traffic_speed_dataset
        elif dataset_name == 'Melboune':
            self.loading_data = self.__get_maximal_Melboune_airport_temperature_dataset
        elif dataset_name == 'Electricity':
            self.loading_data = self.__get_opsd_germany_daily_electricity_dataset
        elif dataset_name == 'MGS17':
            self.loading_data = self.__get_MGS17_dataset
        elif dataset_name == 'Noisy_MGS17':
            self.loading_data = self.__get_Noisy_MGS17_dataset
        elif dataset_name == 'Laser':
            self.loading_data = self.__get_laser_dataset
        elif dataset_name == 'Noisy_Laser':
            self.loading_data = self.__get_Noisy_laser_dataset
        self._preprocessing()
        self._dataset_index_spliter()
        self._overall_target_maker()
    def _preprocessing(self):
        if self.loading_data.ndim >1:
            self.loading_data = np.squeeze(self.loading_data)
        else:
            scaler = MinMaxScaler()
            self.loading_data = scaler.fit_transform(np.expand_dims(self.loading_data,1))+0.0001
    
    def _overall_target_maker(self):
        _,self.target_y = input_target_spliter(self.loading_data,self.delay)
    
    def _dataset_index_spliter(self):
        if self.dataset_name == 'Cardio':
            if self.validation == True:
                self.washout_set_index = np.arange(0,131)
                self.train_set_index = np.arange(131,631)
                self.validation_set_index = np.arange(631,831)
                self.test_set_index = np.arange(831,1031-self.delay)
            else:
                self.washout_set_index = np.arange(0,131)
                self.train_set_index = np.arange(131,831)
                self.test_set_index = np.arange(831,1031-self.delay)
        elif self.dataset_name == 'Sunspot':
            if self.validation == True:
                self.washout_set_index = np.arange(0,249)
                self.train_set_index = np.arange(249,2249)
                self.validation_set_index = np.arange(2249,2749)
                self.test_set_index = np.arange(2749,3249-self.delay)
            else:
                self.washout_set_index = np.arange(0,249)
                self.train_set_index = np.arange(249,2749)
                self.test_set_index = np.arange(2749,3249-self.delay)
        elif self.dataset_name == 'Bike':
            if self.validation == True:
                self.washout_set_index = np.arange(0,100)
                self.train_set_index = np.arange(100,3000)
                self.validation_set_index = np.arange(3000,4000)
                self.test_set_index = np.arange(4000,5000-self.delay)
            else:
                self.washout_set_index = np.arange(0,100)
                self.train_set_index = np.arange(100,4000)
                self.test_set_index = np.arange(4000,5000-self.delay)
        elif self.dataset_name == 'Traffic':
            if self.validation == True:
                self.washout_set_index = np.arange(0,783)
                self.train_set_index = np.arange(783,4783)
                self.validation_set_index = np.arange(4783,6783)
                self.test_set_index = np.arange(6783,8783-self.delay)
            else:
                self.washout_set_index = np.arange(0,783)
                self.train_set_index = np.arange(783,6783)
                self.test_set_index = np.arange(6783,8783-self.delay)
        elif self.dataset_name == 'Melboune':
            if self.validation == True:
                self.washout_set_index = np.arange(0,112)
                self.train_set_index = np.arange(112,3112)
                self.validation_set_index = np.arange(3112,4112)
                self.test_set_index = np.arange(4112,5112-self.delay)
            else:
                self.washout_set_index = np.arange(0,112)
                self.train_set_index = np.arange(112,4112)
                self.test_set_index = np.arange(4112,5112-self.delay)
        elif self.dataset_name == 'Electricity':
            if self.validation == True:
                self.washout_set_index = np.arange(0,182)
                self.train_set_index = np.arange(182,2382)
                self.validation_set_index = np.arange(2382,3382)
                self.test_set_index = np.arange(3382,4382-self.delay)
            else:
                self.washout_set_index = np.arange(0,182)
                self.train_set_index = np.arange(182,3382)
                self.test_set_index = np.arange(3382,4382-self.delay)
        elif self.dataset_name == 'MGS17':
            if self.validation == True:
                self.washout_set_index = np.arange(0,100)
                self.train_set_index = np.arange(100,3100)
                self.validation_set_index = np.arange(3100,4100)
                self.test_set_index = np.arange(4100,5100)
            else:
                self.washout_set_index = np.arange(0,100)
                self.train_set_index = np.arange(100,4100)
                self.test_set_index = np.arange(4100,5100)
        elif self.dataset_name == 'Noisy_MGS17':
            if self.validation == True:
                self.washout_set_index = np.arange(0,100)
                self.train_set_index = np.arange(100,3100)
                self.validation_set_index = np.arange(3100,4100)
                self.test_set_index = np.arange(4100,5100)
            else:
                self.washout_set_index = np.arange(0,100)
                self.train_set_index = np.arange(100,4100)
                self.test_set_index = np.arange(4100,5100)
        elif self.dataset_name == 'Laser':
            if self.validation == True:
                self.washout_set_index = np.arange(0,100)
                self.train_set_index = np.arange(100,2100)
                self.validation_set_index = np.arange(2100,2600)
                self.test_set_index = np.arange(2600,3100)
            else:
                self.washout_set_index = np.arange(0,100)
                self.train_set_index = np.arange(100,2600)
                self.test_set_index = np.arange(2600,3100)
        elif self.dataset_name == 'Noisy_Laser':
            if self.validation == True:
                self.washout_set_index = np.arange(0,100)
                self.train_set_index = np.arange(100,2100)
                self.validation_set_index = np.arange(2100,2600)
                self.test_set_index = np.arange(2600,3100)
            else:
                self.washout_set_index = np.arange(0,100)
                self.train_set_index = np.arange(100,2600)
                self.test_set_index = np.arange(2600,3100)
        