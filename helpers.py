import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error as mse

class MaxScaler:
    def __init__(self,_Variables,verbose=True):
        self.colmax = np.max(np.abs(_Variables), axis=0)
        self.colmin = np.min(np.abs(_Variables), axis=0)
        self.colmean = np.mean(np.abs(_Variables), axis=0)
        if verbose:
            print("# Abs_maximum of features: ",self.colmax)
    def transform(self,_Variables):
        return _Variables/self.colmax
    def inverse_transform(self,_Variables):
        return _Variables*self.colmax

class TorchScaler:
    def __init__(self,colmax):
        self.colmax = torch.from_numpy(colmax)
    def transform(self,_Variables):
        return _Variables/self.colmax
    def inverse_transform(self,_Variables):
        return _Variables*self.colmax

class Data(Dataset):
    # Constructor
    def __init__(self,_X, _Y, _Y1):
        _X     = _X.astype(np.double)
        _Y     = _Y.astype(np.double)
        _Y1    = _Y1.astype(np.double)
        self.x = torch.from_numpy(_X) 
        self.y = torch.from_numpy(_Y)
        self.y1= torch.from_numpy(_Y1)
        self.len = self.x.shape[0]
        
    # Getter
    def __getitem__(self, index):    
        return self.x[index], self.y[index], self.y1[index]
    
    # Get length
    def __len__(self):
        return self.len

def computeEqCO(temps: np.ndarray, H2: np.ndarray, CO: np.ndarray, H2O: np.ndarray, CO2: np.ndarray) -> np.ndarray:
    """
    Computes the EqCO for the reaction
    """
    length = len(temps)

    EqCO = np.zeros([length]) 
    k = 1e+18 

    for ii in range(length): 
        if temps[ii] + 273 > 100:
            k = np.exp(4577.8 / (temps[ii] + 273) - 4.33)
            count_Tlo += 1
            
        coeffa = (1 - k) * CO[ii] * CO[ii]
        coeffb = CO[ii] * (H2[ii] + CO2[ii] + k * (CO[ii] + H2O[ii]))
        coeffc = H2[ii] * CO2[ii] - k * H2O[ii] * CO[ii]
        discriminant = max(0, coeffb ** 2 - 4 * coeffa * coeffc) # clamp discriminant to be >= 0
                
        disc_sqrt = np.sqrt(discriminant) 
        root1 = (0.5 / coeffa) * (-coeffb - disc_sqrt)
        root2 = (0.5 / coeffa) * (-coeffb + disc_sqrt)
        root  = np.max([root1, root2])

        if root > 1:
            root = np.min([root1, root2, 1])

        if root < 0:
            countNv += 1
                         
        EqCO[ii] = root

    EqCO = EqCO * 100 # convert in percentage
    EqCO[np.isnan(EqCO)] = 0.0
    return EqCO

def compareEqExpCO(eq: np.ndarray, exp: np.ndarray):
    deltaCO = eq - exp
    return np.count_nonzero(deltaCO < 0)

def filterInvalidData(x: np.ndarray, y: np.ndarray, key: int = 0) -> tuple[np.ndarray]:
    """
    Filters data based on key
    """
    if key == 0:
        return [x, y]
    
    alx, aly = [], []
    
    for ii in range(len(y)):
        if (key == 1 and y[ii][1] >= 0) or (key == 2 and y[ii][1] - y[ii][0] >= 0):
            alx.append(x[ii])
            aly.append(y[ii])
            
    return np.array(alx), np.array(aly)

def train_test_split_forcv(X: np.ndarray, y: np.ndarray, foldid: int, testpt_array: np.ndarray) -> tuple:
    test_stindex = int(np.sum(testpt_array[:foldid]))
    test_enindex = int(np.sum(testpt_array[:foldid+1]))

    trainx = np.delete(X, np.arange(test_stindex, test_enindex), axis=0)
    trainy = np.delete(y, np.arange(test_stindex, test_enindex), axis=0)

    testx = X[test_stindex:test_enindex,:]
    testy = y[test_stindex:test_enindex,:]
    return trainx, testx, trainy, testy 

def sectional_rmse(truth, pred, sum=False, squared=False):
    """Ensure pred and truth have preparation methods extracted out

    if squared == True, gives MSE. If false, gives RMSE
    if sum == False, gives average over columns. If True, gives sum over columns

    """

    errors = np.array([0, 0, 0, 0])
    for row in truth:
        base = mse(row[:7], pred[:7], squared=squared)
        promoter = mse(row[9:37], pred[9:37], squared=squared)
        support = mse(row[37:58], pred[37:58], squared=squared)
        row_others, pred_others = np.concatenate([row[7:9], row[58:]]), np.concatenate([pred[7:9], pred[58:]])
        others = mse(row_others, pred_others, squared=squared)
        error_iter = np.array([base, promoter, support, others])
        errors = np.add(errors, error_iter)
    errors /= truth.shape[0]

    if not sum:
        base        /= truth[:7].shape[0]
        promoter    /= truth[9:37].shape[0]
        support     /= truth[37:58].shape[0]
        others      /= row_others.shape[0]

    print("#######################")    
    print("RMSE similarities (the higher, the more novel)")
    print("Base materials     : ", errors[0])
    print("Promoter materials : ", errors[1])
    print("Support materials  : ", errors[2])
    print("Others             : ", errors[3])
    print("Overall            : ", np.sum(errors))
    print("#######################", end="\n\n")
    return

def check_final(final_pos: np.ndarray, ref: dict, bounds: tuple) -> None:
    """
    Verify no boundary violation or gas volume violation 
    and saves final position as csv file if no violations detected 

    Args:
        final_pos (np.ndarray): best position explored/exploited
        ref (dict): ref
        bounds (tuple): bounds of (lb, ub)
    
    Returns:
        None
    """

    violations = 0
    vol_sum = np.sum(final_pos[-8:-2])
    if vol_sum > 100:
        print(f"Final gas volume violation of {vol_sum} > 100%")
        violations += 1
    
    boundaries = zip(bounds[0], bounds[1])
    for idx, (l, u) in enumerate(boundaries):
        if final_pos[idx] > u or final_pos[idx] < l:
            for k, v in ref.items():
                if v == idx:
                    print(f"Final position violated bounds of {(l, u)} with value {final_pos[idx]} in the \"{k}\" column")
                    violations += 1
    
    if not violations:
        print("No violations in final position")

def check_log(history: list, bounds: tuple, check: bool = True) -> None:
    """ 
    Checks for boundary violations on all particles and iterations and 
    saves the history if no violations detected.

    Args:
        history (list): optim.pos_history
        bounds (tuple): bounds
        check (bool): whether to check for boundary violation. Default is True.
    
    Returns:
        None
    """

    if check:
        violations = 0
        boundaries = zip(bounds[0], bounds[1])
        for i, iter in enumerate(history):
            for p, particle in enumerate(iter):
                for col, (l, u) in enumerate(boundaries):
                    if particle[col] > u or particle[col] < l:
                        print(f"Particle {p} violated bounds of {(l, u)} with value {particle[col]} in column {col} on iteration {i}")
                        violations += 1
        if not violations:
            print("No violations in position history")