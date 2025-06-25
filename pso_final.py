import time, joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.utils import shuffle

from helpers import *
from wgs_optim_helper import *

from pyswarms.utils.plotters import plot_cost_history

start = time.time()

#------------------------------------ 
# Reading WGS data file
#------------------------------------

filename="WGS_dataset.csv"
dforig=pd.read_csv(filename)
dforig=dforig.drop(['Total # of Data', 'Reference'], axis=1)

#------------------------------------ 
# compute thermodynamic equilibrium CO conversion
#------------------------------------ 

eqCO=computeEqCO(dforig['Temperature (C)'].values,dforig['H2 vol.%'].values,
                 dforig['CO vol.%'].values,dforig['H2O vol.%'].values,
                 dforig['CO2 vol.%'].values)

#------------------------------------ 
# compare thermodynamic equilibrium CO
# and experimental CO conversion %
#------------------------------------ 

compareEqExpCO(eqCO, dforig['CO_Conversion'].values, plot=False)

#------------------------------------ 
# add Equil. CO in the dataframe
#------------------------------------ 

dforig['Eq_CO_Conversion']=eqCO

#------------------------------------ 
# extract column names and truncate name if necessary
#------------------------------------

columns = dforig.columns.tolist()
columns = columns[:7] + columns[19:-2]
for i, c in enumerate(columns):
    if c == "Calc T (oC)":
        columns[i] = "CalcToC"
    elif c == "Calc T. (hr)":
        columns[i] = "CalcThr"
    elif c == "Temperature (C)":
        columns[i] = "Temperature"
    elif "vol.%" in c or "(min)" in c or "(mg.min/ml)" in c:
        columns[i] = c.split()[0]
    elif c == "Eq_CO_Conversion":
        columns[i] = "EqCO"
idxs = [i for i in range(len(columns))]
ref = {v: i for i, v in zip(idxs, columns)}

#------------------------------------ 
# shuffle the dataframe
#------------------------------------

df=dforig.copy()
for shid in [1,2,3]:
    df=shuffle(df,random_state=shid)
nb_xs=len(df.columns)-2
nb_ys=2
allx=df[df.columns[:nb_xs]].to_numpy()
ally=df.loc[:,['CO_Conversion','Eq_CO_Conversion']].to_numpy()

#------------------------------------ 
# delete data points 
# key 0 --> no deletion
# key 1 --> deletion of Eq_y<0
# key 2 --> deletion of y > Eq_y
#------------------------------------ 

[allx,ally]=deleteData(allx,ally,key=2)

#------------------------------------ 
# normalize ip/op features
#------------------------------------  

scalerX=MaxScaler(allx, verbose=False)
allx=scalerX.transform(allx)

#------------------------------------ 
# extract prep methods from X and create func to paste them back in
#------------------------------------  

prep = allx[:, 7:19]
allx = np.delete(allx, [i for i in range(7, 19)], axis=1)

def set_prep(alx: np.ndarray, methodlist: list =[]):
    """ Pastes prep methods columns back into the feature array X

    Args:
        alx (np.ndarray): ndarray of features without prep methods
        methodlist (list, optional): list of preferred prep methods. Defaults to [].

    Returns:
        newX (ndarray): feature array with prep methods
    """

    # convert prep method name to respective indexes
    methods = {"IWI": 0, "WI": 1, "CI": 2, "SI": 3, "SGP": 4, "CP": 5, "HDP": 6, "UGC": 7, "SCT": 8, "FSP": 9, "ME": 10, "DP": 11}

    if type(methodlist) == str:
        methodlist = [methodlist]

    # accomodate for edge case
    if len(alx.shape) == 1:
        base_arr = np.zeros(12)
        for method in methodlist:
            idx = methods[method]
            base_arr[idx] = 1
        newX = np.insert(alx, 7, base_arr)

    else:
        base_arr = np.zeros([alx.shape[0], 12])
        for method in methodlist:
            idx = methods[method]
            base_arr[:, idx] = 1
        newX = np.insert(alx, [7], base_arr, axis=1)
    return newX

print("Shape of X: ", allx.shape)

#------------------------------------ 
# define method to set bounds
#------------------------------------

def set_bounds( colnames, preferred):
    """ sets bounds for pso according to user preset values

    Args:
        colnames (dict): if user wants to set bounds, indicate colname: (lower, upper)
        preferred (dict): dict of preferred materials

    Raises:
        ValueError: raises if kwargs not correctly passed

    Returns:
        bounds (tuple): tuple of (lower bounds, upper bounds) format where
                        both bounds are ndarrays
    """

    lb = np.concatenate([scalerX.colmin[:7], scalerX.colmin[19:]])
    ub = np.concatenate([scalerX.colmax[:7], scalerX.colmax[19:]])
    means = np.concatenate([scalerX.colmean[:7], scalerX.colmean[19:]])
    maxs = np.copy(ub)

    # if bounds not specified, allow a 10% wider range on both maximum and minimum from dataset for exploration
    # below is derived from: lb, ub = (means - (means - lb) * 1.1) / maxs, (means + (ub - means) * 1.1) / maxs
    # must ensure lower bounds remain capped at 0 despite the 10% allowance
    lb, ub = (1.1 * lb - 0.1 * means) / maxs, (1.1 * ub - 0.1 * means) / maxs
    lb = np.where(lb > 0, lb, 0)

    for cat, allowed in preferred.items():
        if cat == "base":
            start, end = 0, 7
        elif cat == "support":
            start, end = 9, 37
        else:
            start, end = 37, 58

        if allowed == "all":
            pass
        else:
            allowed_idxs = list(map(lambda x: ref[x], allowed))
            for i in range(start, end):
                if i not in allowed_idxs:
                    lb[i], ub[i] = 0, 0

    # if bounds specified, do not impose 10% allowance (this is for the other columns)
    for col in colnames.keys():
        if col not in ref:
            raise ValueError(f"Parameter {col} not identified")
        elif type(colnames[col]) != tuple or len(colnames[col]) != 2:
            raise ValueError(f"Parameter {col} must be a tuple of size 2 in (lower bound, upper bound) format")
        elif colnames[col][0] > colnames[col][1]:
            raise ValueError(f"Parameter {col} lower bound must be smaller than or equal to upper bound")
        else:
            lower, upper = colnames[col]
            idx = ref[col]
            maxi = maxs[idx]
            lb[idx], ub[idx] = lower / maxi, upper / maxi
    bounds = (lb, ub)
    return bounds

#------------------------------------ 
# define the NN architecture (necessary to load in pre-trained model)
#------------------------------------ 

class NetCatalyst(nn.Module):
    
    # Constructor
    def __init__(self, Layers):
        super(NetCatalyst, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            self.hidden.append(linear)

    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.relu(linear_transform(x))
            else:
                x = linear_transform(x)
                x=torch.sigmoid(x) # additional activation function
        return x

#------------------------------------ 
# custom loss functions
#------------------------------------ 

def eqloss(predy: torch.Tensor, y1: torch.Tensor, temp: torch.Tensor=0, weights: list =[0.8, 0.2, 0]):
    """ Loss function that maximises CO conversion, ensures EqCO constraint and minimises temperature

    Args:
        predy: predicted CO conversion rate
        y1: Theoretical limit
        temp: temperature value (optional)
        weights (list): list of weights for predy violation, delta violation

    Returns:
        loss (torch.Tensor): eqLoss
    """

    delta = y1 - predy
    delta[delta > 0] = 0 # if CO conversion adheres to EqCO, don't penalise
    loss = weights[0] * (1 - predy) ** 2 + weights[1] * delta ** 2 +  weights[2] * temp # maximise CO conv and minimise temp
    return loss

#------------------------------------ 
# load model and weights
#------------------------------------ 

modelA = torch.load("annA.pt")
modelX = joblib.load("xgboost.pkl")

def model(X):
    """shortcut to combine both models
    """
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    A = modelA(X).detach().squeeze()
    xgb = torch.from_numpy(modelX.predict(X))
    return (A + xgb) / 2

###############################################################################
# 
# PSO parameters (this is the only part the user has to change)
#
###############################################################################

preferred       = {"base": ["Pt", "Au"], "support": ["CeO2", "Al2O3", "TiO2", "MgO"], "promoter": "all"} # preferred base, support and promoter materials
methodlist      = ["IWI", "DP"] # preferred prep methods
limits          = {"CO": (1e-1, 100), "H2O": (1e-1, 100), "Temperature": (200, 250)} # colname: (lower, upper) unnormalised bounds
options         = {"c1": 0.3, "c2": 0.3, "w": 0.7} # c1 exploration, c2 exploitation, w inertia
n_particles     = 100
iters           = 100 # iterations of PSO
log_cost_graph  = True # whether to plot cost graph
file_name       = "sample_run.xlsx" # to save the global best position history to (MUST BE EXCEL FILE!!)
boundary_strat  = "nearest" # boundary handling strategy for pso (if unsure, reset to "nearest")
rep             = 10 # intervals to log gbest
save            = True # whether to save to excel file

initial         = np.zeros([n_particles, allx.shape[1]]) # initial positions, if not using put None
use_initial     = False
ftol            = 1e-11 # tolerance to declare convergence, if not using put None
use_ftol        = False
ftol_iter       = 1 # iteration no. after which to check for convergence
verbose         = True # verbosity

#------------------------------------ 
# Other necessary initialisations 
#------------------------------------ 

bounds          = set_bounds(limits, preferred)
dimensions      = allx.shape[1]
temp_alpha      = 0.0 # weight for temperature in cost function (currently not in use)

#------------------------------------ 
# PSO algorithm implementation
#------------------------------------

def objective(X: np.ndarray):
    """ objective function for pso

    Args:
        X (ndarray): 
            particle values
        scaler (MaxScaler): 
            scaler used to normalise data earlier
        methodlist (list): 
            list of targeted prep methods
        temp_alpha (float): 
            weight for temperature component of loss function
        model (NetCatalyst): 
            pretrained ANN model

    Returns:
        total (ndarray): final loss for all particles
    """

    X = set_prep(X, methodlist) # paste back the prep methods
    X = torch.from_numpy(X).float()
    # X = torch.clamp(X, min=0.0)

    # derive original values for EqCO conversion calculation
    unnormX=scalerX.inverse_transform(X)
    y1 = computeEqCO(unnormX[:,-9], unnormX[:,-8], unnormX[:,-6], unnormX[:,-5], unnormX[:,-4])
    y1 = torch.from_numpy(y1).float()

    # prediction and eq loss
    predy = model(X)
    eqcoLoss = eqloss(predy, y1, unnormX[:,-9], weights=[0.5, 0.5, 0])
    total = eqcoLoss.detach().numpy()
    return total

if not use_initial:
    initial = None
if not use_ftol:
    ftol = None

# optimise
cost, pos, gbests, cost_his, pos_his = wgs_optimise(objective, iters=iters, report_iters=rep, scale=scalerX,
                                 n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds,
                                 bh_strategy=boundary_strat,
                                 init_pos=initial, ftol=ftol, ftol_iter=ftol_iter, verbose=verbose)

#------------------------------------ 
# boundary check and save file
#------------------------------------

# verify no boundary violations in training loop and final result
print("#######################")
check_final(pos, ref, bounds)
check_log(pos_his, bounds)
print("#######################")

#------------------------------------ 
# predict final CO conversion value and log cost if true
#------------------------------------

ipos = set_prep(pos, methodlist)
ipos = torch.from_numpy(ipos).float()
final = scalerX.inverse_transform(ipos)
print(f"Final predicted CO conversion: {model(ipos).item()}")
print("Final feature values:", end=" ")
print(final.numpy())

if log_cost_graph:
    plot_cost_history(cost_history=cost_his)
    plt.show()

#------------------------------------ 
# test similarity to the original data
#------------------------------------

sectional_rmse(allx, pos, sum=True, squared=False)

#------------------------------------ 
# log gbests
#------------------------------------
if save:
    csv_columns=df.columns.to_list()
    csv_columns.pop(-1) # remove EqCO
    csv_columns.pop(-1) # remove CO
    csv_columns.append("CO_pso") # add predicted CO conv by annA

    convs = np.array([])

    gbests = list(map(lambda x: set_prep(x, methodlist), gbests))
    for gbest in gbests:
        temp = torch.from_numpy(np.copy(gbest))
        conv = model(temp.float()).clone().detach().numpy()
        convs = np.concatenate((convs, conv), axis=0)
    gbests = list(map(lambda x: scalerX.inverse_transform(x), gbests))

    final = np.stack(gbests)
    final = np.concatenate((final, convs.reshape(-1, 1)), axis=1)

    output = pd.DataFrame(data=final, columns=csv_columns)

    if file_name.endswith(".xlsx"):
        writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
        output.to_excel(writer,index=False)
        writer._save()
    elif file_name.endswith(".csv"):
        output.to_csv(file_name, index=True)
    else:
        print("...Invalid file format specified, results will not be saved...")

print(f'# Time (s): {round(time.time() - start, 5)}')