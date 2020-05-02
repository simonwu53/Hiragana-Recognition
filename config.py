# ---training set statistics---
TRAINSET_MEAN = 0.0933
TRAINSET_STD = 0.1803

# ---optimizer---
LR = 1e-3  # learning rate
L2 = 0.0  # L2 regularization

# ---data loader---
BS = 8  # batch size
SF = True  # shuffle
dropLast = True
numWorkers = 4
pinMem = True
timeOut = 5

# ---train details---
Epochs = 10
TBUpdate = 100  # tensorboard update interval
