SEED = 53
# ---training set statistics---
MEAN = 0.0935
STD = 0.1806
MEAN_R = 0.0935
MEAN_G = 0.0935
MEAN_B = 0.0935
STD_R = 0.1806
STD_G = 0.1806
STD_B = 0.1806

# ---optimizer---
LR = 1e-3  # learning rate
L2 = 0.0  # L2 regularization

# ---Dataset---
upSampling = 50
testSize = 0.15
trainSize = None

# ---Dataset(Canvas)---
trainLen = 2000
testLen = 20
maxCharacters = 10
minCharacters = 3

# ---DataLoader---
BS = 2  # batch size
SF = True  # shuffle
dropLast = True
numWorkers = 4
pinMem = True
timeOut = 5

# ---train details---
Epochs = 30
TBUpdate = 100  # tensorboard update interval
nmsIoU = 0.3
