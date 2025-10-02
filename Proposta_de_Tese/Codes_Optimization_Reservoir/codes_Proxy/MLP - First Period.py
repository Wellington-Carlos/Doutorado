# -*- coding: utf-8 -*-
"""
    In case of using the codes, the user must cite the related work:
       https://doi.org/10.1016/j.geoen.2024.213621
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir(r"your\working\directory")
import glob
from sklearn.model_selection import train_test_split as tts
from keras.layers import Dense, Input, Conv2D, Concatenate, Flatten, MaxPool2D
from keras.metrics import R2Score
from tensorflow.keras.optimizers import Adam
from keras.models import Model, load_model
from scipy import stats
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pickle
import seaborn as sns

data = pd.read_csv("First-Period.csv").dropna()

perm_files = glob.glob(r"your\working\directory\Realizations\*.INC")
perms = np.zeros((7500,60,60,7))
n=0
for file in perm_files:
    with open(file, "r") as f:
        content = f.readlines()[2:4627]
    
    x = np.array([line.split() for line in content if line!="\n"]).astype(float)
    for k in range(perms.shape[3]):
        for j in range(perms.shape[2]):
            perms[n*2500:(n+1)*2500, :, j, k] = x[j*10:(j+1)*10,:].ravel()
    
    n+=1

perms = perms/np.max(perms)
plt.imshow(perms[4000,...,0], cmap="jet")
        
# In[]
X = data.drop(["COP", "Perm"], axis=1)
y = data["COP"]/data["COP"].max()

zscores = stats.zscore(y, nan_policy="omit")
fig = plt.figure(figsize=(5,3))
ax = sns.boxplot(zscores, orient="h", fliersize=4, width=0.5)
plt.axvline(-3, ls="--", lw=1)
plt.axvline(3, ls="--", lw=1)
plt.xlim([-4,4])
plt.xlabel("Z-score")
plt.yticks([0], labels=["COP"])
plt.show()
fig.tight_layout()

to_del = np.union1d(np.where(zscores>3)[0], np.where(zscores<-3)[0])

X = X.drop(to_del, axis=0)
y = y.drop(to_del, axis=0)
perms = np.delete(perms, to_del, axis=0)

ts = 0.125
x_train, x_val, y_train, y_val = tts(X,y, test_size=ts, shuffle=True, random_state=40)
x_train, x_test, y_train, y_test = tts(x_train, y_train, test_size=ts/(1-ts), shuffle=True, random_state=40)

perm_train, perm_val = tts(perms, test_size=ts, shuffle=True, random_state=40)
perm_train, perm_test = tts(perm_train, test_size=ts/(1-ts), shuffle=True, random_state=40)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

with open("Scaler - First Period", "wb") as file:
    pickle.dump(scaler, file)


# In[]

def R2(y_true, y_pred):
    scorer = R2Score()
    scorer.update_state(y_true, y_pred)
    return scorer.result()

def scheduler(epoch, lr):
    if (epoch+1)%50 == 0:
        lr *= 0.8
    return lr

early = EarlyStopping(patience = 15, monitor="val_loss", restore_best_weights=True)
lr = ReduceLROnPlateau(patience=7, monitor="val_loss", factor=0.5, verbose=1, min_lr=0.00001)
lr_sched = LearningRateScheduler(scheduler, verbose=1)
inp1 = Input(shape=(X.shape[1],))
inp2 = Input(shape=(perms.shape[1], perms.shape[2], perms.shape[3]))
x1 = Dense(16, activation="relu")(inp1)
x1 = Dense(96, activation="relu")(x1)
x1 = Dense(336, activation="relu")(x1)

x2 = Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")(inp2)
x2 = MaxPool2D(pool_size=3)(x2)
x2 = Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")(x2)
x2 = Flatten()(x2)
x2 = Dense(432, activation="relu")(x2)
x2 = Dense(32, activation="relu")(x2)
x2 = Dense(336, activation="relu")(x2)
x2 = Dense(112, activation="relu")(x2)

x = Concatenate()([x1,x2])

out = Dense(1, activation="linear")(x)
model = Model([inp1, inp2], out)
model.compile(optimizer=Adam(learning_rate=0.0005), loss="mse", metrics=[R2Score()])
if os.path.exists("TrainedMLP - First Period.hdf5"):
    model = load_model("TrainedMLP - First Period.hdf5", compile=False)
else:
    history = model.fit([x_train,perm_train], y_train, epochs=300, batch_size=64,
                    validation_data=([x_val, perm_val], y_val),
                    callbacks=[early, lr_sched])
    model.save("TrainedMLP - First Period.hdf5")

# In[]
fig = plt.figure(figsize=(5,4))
plt.semilogy(np.arange(1,len(history.history["loss"])+1), history.history["loss"])
plt.semilogy(np.arange(1,len(history.history["loss"])+1), history.history["val_loss"])
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Loss vs. Epochs", fontsize=14)
plt.legend(["Train", "Validation"])
fig.tight_layout()
fig.savefig("Loss - Period 1.jpg", dpi=300, format="jpg")
with open("History - Period 1", "wb") as file:
    pickle.dump(history.history, file)
# In[]
fig = plt.figure(figsize=(5,4))
y_pred_train = model.predict([x_train, perm_train])
y_pred_test = model.predict([x_test, perm_test])
# y_pred_test = np.load("Test Predictions First.npy")
# y_pred_train = np.load("Train Predictions First.npy")
y_pred_test = y_pred_test.reshape(-1,1) # 
y_pred_train = y_pred_train.reshape(-1,1)#
plt.plot(y_test, y_pred_test, ".", ms=8)
plt.xlim([np.min(y_test)*0.9, np.max(y_test)*1.05])
plt.ylim([np.min(y_test)*0.9, np.max(y_test)*1.05])
plt.plot([np.min(y_test)*0.9, np.max(y_test)*1.05], [np.min(y_test)*0.9, np.max(y_test)*1.05], "--r")
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.legend([f"$R^2$ = {r2_score(y_test, y_pred_test):.2f}", "Ideal line"])
plt.text(x=0.4, y=0.9, s="Testing data", fontdict=dict(fontsize=12))
plt.title("True vs. Predicted Values with $R^2$ Score")
fig.tight_layout()
plt.show()
fig.savefig("Crossplot Test - Period 1.jpg", dpi=300, format="jpg")
print(r2_score(y_train, y_pred_train))
print(r2_score(y_test, y_pred_test))

# In[]
np.save("Test Predictions - Period 1", y_pred_test)
np.save("Train Predictions - Period 1", y_pred_train)
