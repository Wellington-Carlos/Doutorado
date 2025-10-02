import os
import shutil
import numpy as np
import time
from ecl.summary import EclSum

# MAINDIR = r"your\working\directory"
# MAINDIR = "C:/Users/calva/OneDrive/Documentos/WellingtonCodesOptimizationReservoirModels"
MAINDIR = "/mnt/c/Users/calva/OneDrive/Documentos/WellingtonCodesOptimizationReservoirModels"

TempFolder = os.path.join(MAINDIR, "Temp")
#ModelDir = os.path.join(MAINDIR, "Egg - Two") # Change this to Egg - Four for the second period
ModelDir = os.path.join(MAINDIR, "Egg - Four") # Change this to Egg - Four for the second period

if not os.path.exists(TempFolder):
    os.mkdir(TempFolder)
    
def objective(ga_instance, solution, solution_idx):
    
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
            return -100
        
    #realizations = ["PERM60_ECL.INC", "PERM76_ECL.INC", "PERM82_ECL.INC"]
    realizations = os.listdir("Realizations")
    try:
        if os.path.exists(os.path.join(TempFolder, "EGG.UNSMRY")):
            os.remove(os.path.join(TempFolder, "EGG.UNSMRY"))
    except:
        pass   
    shutil.copytree(ModelDir, TempFolder, dirs_exist_ok=True)
    
    os.chdir(TempFolder)
    
    with open("EGG.DATA", "r") as file:
        content = file.readlines()
        
    for c, line in enumerate(content):
        if "Realizations" in line:
            real_ln = c * 1
        if "WELSPECS" in line:
            for i in range(int(len(solution)/2)):
                tokens = content[c+i+1].split()
                tokens[2] = str(int(solution[i*2]))
                tokens[3] = str(int(solution[i*2+1]))
                content[c+i+1] = " ".join(tokens)+"\n"
            break
    vals = []
    for real in realizations:
        content[real_ln] = f"'../Realizations/{real}'"
        
        with open("EGG.DATA", "w") as file:
            file.writelines(content)
            
        _ = os.system("flow EGG.DATA")
        
        while not os.path.exists("EGG.UNSMRY"):
            time.sleep(1)
        if os.path.exists("EGG.UNSMRY"):            
           eclsum = EclSum("EGG.UNSMRY")
           fopt = eclsum.numpy_vector("FOPT")
           if fopt is not None and len(fopt) > 0:
               vals.append(float(fopt[-1]))
           else:
               vals.append(-100)

           try:
               os.remove(os.path.join(TempFolder, "EGG.UNSMRY"))
           except:
               pass
        else:
            return -100
    os.chdir(MAINDIR)
    return np.mean(vals)
