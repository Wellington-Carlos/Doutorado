# -*- coding: utf-8 -*-
"""
    In case of using the codes, please cite the related work:
       https://doi.org/10.1016/j.geoen.2024.213621
"""

import numpy as np
import re
import pandas as pd
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
import shutil

# MAINDIR = r"your\working\directory"
MAINDIR = "/mnt/c/Users/calva/OneDrive/Documentos/WellingtonCodesOptimizationReservoirModels"

# Enter the directory to which the simulation files should be copied
TempFolder = os.path.join(MAINDIR, "Temp")

ModelDir = os.path.join(MAINDIR, "Egg - Two") # Change this to Egg - Four for the second period

# Enter the working directory (containing codes and files)
Case_path = os.path.join(MAINDIR, "Egg - Two") # Change this to Egg - Four for the second period


if not os.path.exists(TempFolder):
    os.mkdir(TempFolder)

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

def modify_file(File_path, new_value, realization):
    
    try:
        if os.path.exists(os.path.join(TempFolder, "EGG.UNSMRY")):
            os.remove(os.path.join(TempFolder, "EGG.UNSMRY"))
    except:
        pass   
    
    shutil.copytree(File_path, TempFolder, dirs_exist_ok = True)
    os.chdir(TempFolder)
    # Read the content of the file
    with open("EGG.DATA", "r") as file:
        content = file.readlines()

    for c, line in enumerate(content):
        if "Realizations" in line:
            content[c] = f"'../Realizations/{realization}'"
        if "WELSPECS" in line:
            for i in range(int(len(new_value)/2)):
                tokens = content[c+i+1].split()
                tokens[2] = str(int(new_value[i*2])+1)
                tokens[3] = str(int(new_value[i*2+1])+1)
                content[c+i+1] = " ".join(tokens)+"\n"
            break
        
    with open("EGG.DATA", "w") as file:
        file.writelines(content)

def extract_value_from_file(file_path, target_line, target_column):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return float(lines[target_line].split()[target_column])
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_random_point(width, height):
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)
    while not np.any(act[x,y]):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
    return x, y

width, height = 60, 60        
generate_random_point(width, height)

num_data = 2500  # as three realizations are used, the total simulations will be 7500
cols = ['PRODX1', 'PRODY1','INJX1','INJY1','COP']
lst = []
Realizations = os.listdir("Realizations")
print("ESTOU AQUI 1")
for i in range(len(Realizations)):
    pointsp = [generate_random_point(width, height) for _ in range(num_data)]
    random.shuffle(pointsp)
    pointsi = [generate_random_point(width, height) for _ in range(num_data)]
    data_list1 = []
    for k in range(0,num_data):
        new_row = {
            'PX1': pointsp[k][0],
            'PY1': pointsp[k][1],
            'IX1': pointsi[k][0],
            'IY1': pointsi[k][1]
        }
        data_list1.append(new_row)
    data_list1=pd.DataFrame(data_list1)
    lst1=[]
    for e in range(len(data_list1)):
        modify_file(Case_path, data_list1.iloc[e].values, Realizations[i])

        os.system("flow EGG.DATA")
        while not os.path.exists("EGG.UNSMRY"):
            continue
        time.sleep(1)
        Cum_Oil = extract_value_from_file("EGG.UNSMRY", -1, 2)

        lst.append((data_list1.iloc[e].values[0],data_list1.iloc[e].values[1],data_list1.iloc[e].values[2],data_list1.iloc[e].values[3],Realizations[i], Cum_Oil))
        os.chdir(MAINDIR)
print("ESTOU AQUI 2")
df = pd.DataFrame(lst)
#df.to_excel("First.xlsx")
print("ESTOU AQUI 3")