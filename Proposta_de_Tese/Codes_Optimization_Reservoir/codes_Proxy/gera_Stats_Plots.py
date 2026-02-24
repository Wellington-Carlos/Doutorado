import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Caminho onde as figuras serão salvas
save_dir = "/mnt/c/Users/calva/OneDrive/Documentos/Doutorado/Proposta_de_Tese/Deep_Learning/artigo/MDPI_template_ACS/Definitions"

# -----------------------------------------
# Escolha o arquivo CSV
df = pd.read_csv("First-Period.csv")
#df = pd.read_csv("Second-Period.csv")       # ou "Second-Period.csv"
period = "Period 1" 
#period = "Period 2"                        # ou "Period 2"
# -----------------------------------------

COP = df["COP"].values

# Divisão usada no seu MLP (70% train / 20% test / 10% val)
X_train, X_temp = train_test_split(COP, test_size=0.30, random_state=42)
X_test, X_val   = train_test_split(X_temp, test_size=1/3, random_state=42)

# Boxplot
plt.figure(figsize=(10,6))
plt.boxplot([X_train, X_test, X_val], patch_artist=True,
            boxprops=dict(facecolor='lightgray'))

plt.xticks([1,2,3], ["Training", "Testing", "Validation"])
plt.ylabel("COP (m³)", fontsize=12)
plt.title(f"Box plot of Training, Testing, and Validation Data Statistics\n{period}", fontsize=14)

plt.tight_layout()
plt.savefig(f"{save_dir}/BoxPlot_{period}.jpg", dpi=300)
plt.close()


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(15,5))

# Training
plt.subplot(1,3,1)
plt.hist(X_train, bins=50, color="blue")
plt.title("Training")
plt.xlabel("COP")
plt.ylabel("Frequency")

# Testing
plt.subplot(1,3,2)
plt.hist(X_test, bins=50, color="green")
plt.title("Testing")
plt.xlabel("COP")

# Validation
plt.subplot(1,3,3)
plt.hist(X_val, bins=50, color="red")
plt.title("Validation")
plt.xlabel("COP")

plt.tight_layout()
plt.savefig(f"{save_dir}/Hist_{period}.jpg", dpi=300)
plt.close()

#Primeiro periodo
#Parameters of the best solution : [5 36 10 51]
#Fitness value of the best solution = 14524.74767267704

#Segundo periodo
#Parameters of the best solution : [44 26 30  19]
#Fitness value of the best solution = 40035.078