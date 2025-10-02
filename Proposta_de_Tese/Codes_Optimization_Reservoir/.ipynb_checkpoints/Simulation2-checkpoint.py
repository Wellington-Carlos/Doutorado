import numpy as np
import pandas as pd
import random
import time
import os
import shutil
from ecl.summary import EclSum

# Caminho base (ajuste se necessário)
MAINDIR = "/mnt/c/Users/calva/OneDrive/Documentos/WellingtonCodesOptimizationReservoirModels"
TempFolder = os.path.join(MAINDIR, "Temp")
ModelDir = os.path.join(MAINDIR, "Egg - Two") # Altere para "Egg - Four" se necessário

if not os.path.exists(TempFolder):
    os.mkdir(TempFolder)

# Ler a máscara de células ativas
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

    shutil.copytree(File_path, TempFolder, dirs_exist_ok=True)
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

def generate_random_point(width, height):
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)
    while not np.any(act[x, y]):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
    return x, y

start = time.time()

width, height = 60, 60
num_data = 2500  # Número de amostras por realização

lst = []
Realizations = os.listdir(os.path.join(MAINDIR, "Realizations"))

for i, realization in enumerate(Realizations):
    pointsp = [generate_random_point(width, height) for _ in range(num_data)]
    random.shuffle(pointsp)
    pointsi = [generate_random_point(width, height) for _ in range(num_data)]
    data_list1 = []
    for k in range(num_data):
        new_row = {
            'PX1': pointsp[k][0],
            'PY1': pointsp[k][1],
            'IX1': pointsi[k][0],
            'IY1': pointsi[k][1]
        }
        data_list1.append(new_row)
    data_list1 = pd.DataFrame(data_list1)
    for e in range(len(data_list1)):
        # Prepara o caso no TempFolder
        modify_file(ModelDir, data_list1.iloc[e].values, realization)

        # Executa a simulação
        os.chdir(TempFolder)
        os.system("flow EGG.DATA")

        # Aguarda criação do arquivo
        wait_count = 0
        while not os.path.exists("EGG.UNSMRY"):
            time.sleep(1)
            wait_count += 1
            if wait_count > 300:
                print("Timeout esperando EGG.UNSMRY")
                Cum_Oil = -100
                break
        else:
            time.sleep(1)  # Espera escrita do arquivo
            try:
                eclsum = EclSum("EGG.UNSMRY")
                fopt = eclsum.numpy_vector("FOPT")
                if fopt is not None and len(fopt) > 0:
                    Cum_Oil = float(fopt[-1])
                else:
                    Cum_Oil = -100
            except Exception as e:
                print("Erro lendo UNSMRY:", e)
                Cum_Oil = -100

        lst.append((
            data_list1.iloc[e].values[0], data_list1.iloc[e].values[1],
            data_list1.iloc[e].values[2], data_list1.iloc[e].values[3],
            realization, Cum_Oil
        ))
        os.chdir(MAINDIR)

end = time.time()
elapsed = end-start
print(f"Elapsed: {elapsed/3600:.2f}")

# Salva o DataFrame no diretório principal
df = pd.DataFrame(lst, columns=['PX1', 'PY1', 'IX1', 'IY1', 'REALIZATION', 'COP'])
output_path = os.path.join(MAINDIR, "First.xlsx")
print("Salvando arquivo Excel em:", output_path)
df.to_excel(output_path, index=False)
print("Arquivo salvo com sucesso!")
