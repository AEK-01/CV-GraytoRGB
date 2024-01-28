import numpy as np


estimations_yigit = np.load("yigit_estimations_test (1).npy")
estimations_eren= np.load("estimations_test_eren.npy")
# estimations_ersel = np.load("estimations_ersel.npy")
estimations_meric = np.load("meric_estimations.npy")
estimations_kadir = np.load("estimations_kadir.npy")

estimations1 = estimations_kadir
estimations2 = estimations_eren
acc = 0

for i, file in enumerate(estimations1):
    est1 = estimations1[i].reshape(-1).astype(np.int64)
    est2 = estimations2[i].reshape(-1).astype(np.int64)
    
    cur_acc = (np.abs(est2 - est1) < 12).sum() / est1.shape[0]
    acc += cur_acc
acc /= len(estimations1)
print(f"{acc:.2f}/1.00")