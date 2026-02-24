import os
import numpy as np
import pickle
from keras.models import load_model
import pandas as pd
# MAINDIR = r"your\working\directory"
MAINDIR = "/mnt/c/Users/calva/OneDrive/Documentos/Doutorado/Proposta_de_Tese/Codes_Optimization_Reservoir/codes_Proxy"
ModelDir = os.path.join(MAINDIR, "Egg - Four")


def objective(ga_instance, solution, solution_idx):
    os.chdir(MAINDIR)
    model = load_model("TrainedMLP - Second Period.hdf5", compile=False) # You train an MLP for each period. Save it and load it here for each period
    with open(os.path.join(ModelDir, "ACTIVE.INC"), "r") as file:
        c = file.readlines()[2:422]

    k = 0
    i = 0
    act = np.zeros((60,60,7))
    for line in c:
        if i == 60:
            k += 1
            i = 0
        if line != "\n":
            line.strip("\n")
            act[i,:,k] = np.array([int(x) for x in line.split()])
            i += 1
            
    for i in range(int(len(solution)/2)):
        if np.any(act[solution[i*2]-1, solution[i*2+1]-1]) == 0:
            return -100 # This is a penalty term for off-domain locations
        
    with open("Scaler - Second Period", "rb") as file:
        scaler = pickle.load(file)
    
    realizations = os.listdir("Realizations")

    perms = np.zeros((3,60,60,7))
    n=0
    for real in realizations:
        with open(os.path.join(MAINDIR, "Realizations", real), "r") as f:
            content = f.readlines()[2:4627]
        
        x = np.array([line.split() for line in content if line!="\n"]).astype(float)
        for k in range(perms.shape[3]):
            for j in range(perms.shape[2]):
                perms[n, :, j, k] = x[j*10:(j+1)*10,:].ravel()
            
        n+=1
    max_perm = 3500  # This is the maximum permeability value used to normalize permebaility realizations. It will differ if the realizations change
    max_cop = 43761.03  # Maximum cumulative oil production. It will differ if simulations change. For first period enter 14510.75. This should be determined based on your simulations
    perms = perms/max_perm
    x_test = np.array([[solution[0], solution[1], solution[2], solution[3]],
                        [solution[0], solution[1], solution[2], solution[3]],
                         [solution[0], solution[1], solution[2], solution[3]]])
    x_test = x_test - 1
    x_test = pd.DataFrame(x_test, columns = ["X1", "Y1", "X2", "Y2"])
    x_test = scaler.transform(x_test)
    preds = model.predict([x_test, perms])
    os.chdir(MAINDIR)
    return np.mean(preds)*max_cop
