import argparse
import numpy as np
from torch.nn.modules import Module
import statsmodels.api as sm
import ESN
from sklearn.linear_model import Ridge
from dataset_loader import timeseries_loader,input_target_spliter
import torch
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
#This is the implementation of HP-MRESN.
#You can refer to our paper for hyperparameter settings in HP-MRESN on various datasets.
#You can add PCA into HP-MRESN by yourself for building HP-MRESN(PCA) (e.g. sklearn.decomposition.PCA)
parser = argparse.ArgumentParser()
parser.add_argument('--in_channels', type=int, default=1,help='Dimension of inputs')
parser.add_argument('--out_channels', type=int, default=500, help = 'Dimension of embeddings')
parser.add_argument('--num_decomposer', type = int, default = 10, help = 'Number of HP decomposers')
parser.add_argument('--sf',type=list,default=[10,9,8,7,6,5,4,3,2,1],help='Smoothing factor for HP decomposers')
parser.add_argument('--lr', type = float, default = 0.4, help = 'Leaking rate, (0,1], default = 1.0')
parser.add_argument('--in_scale', type = float, default = 1, help = 'Input scaling, default = 1.0')
parser.add_argument('--res_density',type = float, default =0.1, help="Density of each reservoir, (0,1]")
parser.add_argument('--C',type=float,default=1e-6, help='Regularization factor of the ridge regression')
parser.add_argument('--dataset',type=str,default='Bike',help='Datasets used in the paper')
#We prepared datasets used in the paper with keywords: "Cardio", "Sunspot", "Bike", "Traffic", "Melboune", "Electricity", "MGS17", "Laser", "Nosiy_MGS17", and "Noisy_Laser".
parser.add_argument('--validation',type=bool,default=True,help='Validation mode or not, If not, the validation set will be merged into the training set')
parser.add_argument('--num_trials',type=int,default=20,help="Number of random trials")
parser.add_argument('--delay',type=int,default=1,help="K-step-ahead, you can choose the value of K.")
args = parser.parse_args()
createVar = locals()
createVar["components"] = dict()

def state_transform(state,flag = 1):
    if flag == 1:
        state_dim = state[0].shape[1]
        _state = torch.Tensor(len(state),state_dim)
        for col_idx in range(len(state)):
            _state[col_idx,:] = state[col_idx]
    else:
        state_dim = state[0].shape[0]
        _state = torch.Tensor(len(state),state_dim)
        for col_idx in range(len(state)):
            _state[col_idx,:] = state[col_idx]
    return _state

class Linearregressor(Module):
    def __init__(self,factor):
        super(Linearregressor, self).__init__()
        self.factor = factor
    def train(self,input_X,target_Y):
        self.regressor = Ridge(alpha=self.factor).fit(input_X,target_Y)
    def test(self,input_X):
        return self.regressor.predict(input_X)
    def extra_repr(self):
        return 'Regularization_factor={}'.format(
            self.factor)
        
class Decomposer(Module):
    def __init__(self,num_decomposer:int,sf:list):
        super(Decomposer,self).__init__()
        self.num_decomposer = num_decomposer
        self.sf = sf
        # sf means smoothing factor
        if num_decomposer == len(sf):
            self.decomposer = sm.tsa.filters.hpfilter
        else:
            raise(ValueError("The length of sf should be same with num_decomposer"))
    
    def extra_repr(self):
        return 'num_decomposer={}, smoothing_factor={}'.format(
        self.num_decomposer, self.sf)
        
    def forward(self,input_series):
        pending_data = None
        for nd in range(self.num_decomposer):
            if nd == 0:
                pending_data = input_series
            else:
                pending_data = createVar["components"]["cycle_{}".format(nd-1)]
            temp_cycle,temp_trend = self.decomposer(pending_data,self.sf[nd])
            createVar["components"]["trend_{}".format(nd)] = temp_trend
            createVar["components"]["cycle_{}".format(nd)] = temp_cycle
        return createVar["components"]

            
class Encoder(Module):
    def __init__(self,in_channels:int,out_channels:int,leaking_rate:float,in_scale:float,res_density:float,num_encoder:int):
        super(Encoder,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.leaking_rate = leaking_rate
        self.in_scale = in_scale
        self.res_density = res_density
        self.num_encoder = num_encoder
        for nd in range(self.num_encoder):
            setattr(self,'encoder_{}'.format(nd),ESN.ESN(input = in_channels,reservoir=out_channels,sr=0.95,scale_in=in_scale,leaking_rate=leaking_rate,density=res_density))

class Decoder(Module):
    def __init__(self,num_encoder:int,regularization_factor:float):
        super(Decoder,self).__init__()
        self.num_decoder = num_encoder
        self.regularization_factor = regularization_factor
        for nd in range(self.num_decoder):
            setattr(self,'decoder_{}'.format(nd),Linearregressor(factor=self.regularization_factor))
    
class HPMRESN(Module):
    def __init__(self, args):
        super(HPMRESN,self).__init__()
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.num_decomposer = args.num_decomposer
        self.num_encoder = self.num_decomposer + 1
        self.smoothing_factor = args.sf
        self.leaking_rate = args.lr
        self.in_scale = args.in_scale
        self.res_density = args.res_density
        self.regularization = args.C
        self.decomposer = Decomposer(num_decomposer = args.num_decomposer,sf=args.sf)
        self.decoder = Decoder(num_encoder = self.num_encoder,regularization_factor = self.regularization)
    
    def generate_encoder(self):
        self.MR = Encoder(in_channels = args.in_channels,out_channels=args.out_channels,leaking_rate=args.lr,in_scale=args.in_scale,res_density=args.res_density,num_encoder=self.num_encoder)
        
    def extra_repr(self):
        return 'decomposer={}, encoder={}'.format(
            self.decomposer, self.encoder)
    
    def forward(self):
        pass

if __name__ == "__main__":
    model = HPMRESN(args)
    dataset = timeseries_loader(dataset_name=args.dataset,validation=args.validation,delay=args.delay)
    components = model.decomposer(dataset.loading_data)
    components_X = dict()
    components_y = dict()
    if dataset.validation == True:
        NRMSE_validation = []
    NRMSE_test = []
    for r in tqdm(range(args.num_trials)):
        model.generate_encoder()
        if dataset.validation == True:
            partial_validation_outputs = dict()
            validation_outputs = 0
        partial_test_outputs = dict()
        test_outputs = 0
        for d in range(args.num_decomposer):
            components_X['{}'.format(d)], components_y['{}'.format(d)] = input_target_spliter(components['trend_{}'.format(d)],dataset.delay)
            components_X['{}'.format(d)] = torch.Tensor(components_X['{}'.format(d)])
            components_X['{}'.format(d)] = components_X['{}'.format(d)].unsqueeze(1)
            temp_state = state_transform(getattr(model.MR, "encoder_{}".format(d))(components_X['{}'.format(d)]))
            getattr(model.decoder,"decoder_{}".format(d)).train(temp_state[dataset.train_set_index],components_y['{}'.format(d)][dataset.train_set_index])
            if dataset.validation == True:
                validation_outputs += getattr(model.decoder,"decoder_{}".format(d)).test(temp_state[dataset.validation_set_index])
            test_outputs += getattr(model.decoder,"decoder_{}".format(d)).test(temp_state[dataset.test_set_index])
            
            if d == args.num_decomposer-1:
                components_X['{}'.format(d+1)], components_y['{}'.format(d+1)] = input_target_spliter(components['cycle_{}'.format(d)],dataset.delay)
                components_X['{}'.format(d+1)] = torch.Tensor(components_X['{}'.format(d+1)])
                components_X['{}'.format(d+1)] = components_X['{}'.format(d+1)].unsqueeze(1)
                temp_state = state_transform(getattr(model.MR, "encoder_{}".format(d+1))(components_X['{}'.format(d+1)]))
                getattr(model.decoder,"decoder_{}".format(d+1)).train(temp_state[dataset.train_set_index],components_y['{}'.format(d+1)][dataset.train_set_index])
                if dataset.validation == True:
                    validation_outputs += getattr(model.decoder,"decoder_{}".format(d+1)).test(temp_state[dataset.validation_set_index])
                test_outputs += getattr(model.decoder,"decoder_{}".format(d+1)).test(temp_state[dataset.test_set_index])
        if dataset.validation == True:
            NRMSE_validation.append(mean_squared_error(y_true=dataset.target_y[dataset.validation_set_index],y_pred=validation_outputs,squared=False)/np.std(dataset.target_y[dataset.validation_set_index]))
        NRMSE_test.append(mean_squared_error(y_true=dataset.target_y[dataset.test_set_index],y_pred=test_outputs,squared=False)/np.std(dataset.target_y[dataset.test_set_index]))
    if dataset.validation == True:
        print('mean RMSE and (std) on the validation set: {}({}) for {}'.format(np.mean(np.array(NRMSE_validation)),np.std(np.array(NRMSE_validation)),args.dataset))
    print('mean RMSE and (std) on the test set: {}({}) for {}'.format(np.mean(np.array(NRMSE_test)), np.std(np.array(NRMSE_test)),args.dataset))