# Important path values
stockfishpath="C:/Coding/Stockfish/stockfish-windows-x86-64-avx2.exe" # Insert here path to your local stockfish
learning_data_path="data/default.data"
testing_data_path="data/test.data"

# Learning related values
device = "cpu"
learning_rate = 1e-5
batch_size = 2048
epochs = 2000

# Model loading values
modelsdirectory="models/"
modelname="loji-0.3"

modelpath = lambda: modelsdirectory+modelname+".pth"