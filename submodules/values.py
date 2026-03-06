# Important path values
stockfishpath="C:/Coding/Stockfish/stockfish-windows-x86-64-avx2.exe"
learning_data_path="data/default.data"
testing_data_path="data/test.data"

# Learning related values
device = "cpu"
learning_rate = 1e-3
batch_size = 2048
epochs = 100

# Model loading values
modelsdirectory="models/"
modelname="loji-0.1"

modelpath = lambda: modelsdirectory+modelname+".pth"