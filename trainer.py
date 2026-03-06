import torch
import chess
from stockfish import Stockfish
from submodules import values
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from pathlib import Path
from submodules import tools
from torch import nn
from math import floor

class ChessDataset(Dataset):
    def __init__(self, chesspositions=[], chessmoves=[]):
        self.chesspositions = chesspositions
        self.moves = chessmoves
    def __len__(self):
        return len(self.chesspositions)
    
    @staticmethod
    def fromPath(path):
        self = ChessDataset()
        if not Path(path).is_file():
            return self
        self.chesspositions = []
        self.moves = []
        with open(path, "r") as file:
            data = file.read()
            splitted_data = data.split("\n")
            for data_line in splitted_data:
                if len(data_line)>0 and ':' in data_line:
                    fen, move = data_line.split(":")
                    self.chesspositions.append(fen)
                    self.moves.append(move)
            file.close()
        return self
    
    def __getitem__(self, i):
        position = self.chesspositions[i]
        temp_board = chess.Board()
        temp_board.set_fen(position)
        if len(self.chesspositions)>i:
            posten = tools.fromFenToTensor(self.chesspositions[i])
        else:
            posten = torch.zeros(0)
        if len(self.moves)>i:
            moveten = tools.fromMoveToTensor(self.moves[i], (temp_board.turn==chess.BLACK))
        else:
            moveten = torch.zeros(0)
        return posten, moveten
    
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9*8*2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 5)
        )
 
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
previous_loss = 1
    
def train_loop(dataloader, model: nn.Module, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (position, move) in enumerate(dataloader):
        position = position.to(device)
        move = move.to(device)

        pred = model(position)
        pred*=8
        loss = loss_fn(pred, move)
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * values.batch_size + len(position)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")
            
def test_loop(dataloader, model, loss_fn, device):
    global previous_loss
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for position, move in dataloader:
            position = position.to(device)
            move = move.to(device)
 
            pred = model(position)
            for i in range(len(pred)):
                for j in range(5):
                    pred[i][j]=abs(pred[i][j].item())*8
            for i in range(size):
                for j in range(4):
                    pred[i][j]*=8
                    pred[i][j]=floor(pred[i][j].item())
                    pred[i][j]/=8
                pred[i][4]*=6
                pred[i][4]=round(pred[i][4].item())
                pred[i][4]/=6
            test_loss += loss_fn(pred, move).item()
 
    test_loss /= num_batches
    print(f"Test Error: \n Accuracy: {((1-test_loss)*100):>4f}%, Avg loss: {test_loss:>8f} \nChange from previous loss: {((test_loss-previous_loss)/previous_loss*100):>4f}%\n")
    previous_loss = test_loss

if __name__ == '__main__':
    
    training_data = ChessDataset.fromPath(values.learning_data_path)
    testing_data = ChessDataset.fromPath(values.testing_data_path)
    training_dataloader = DataLoader(training_data, batch_size=values.batch_size, shuffle=True)
    testing_dataloader = DataLoader(testing_data, batch_size=values.batch_size, shuffle=True)
    model = NeuralNetwork()
    model.to(device=values.device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=values.learning_rate)
    
    for t in range(values.epochs):
        print(f"Epoch: {t+1}")
        train_loop(training_dataloader, model, loss_fn, optimizer, values.device)
        test_loop(testing_dataloader, model, loss_fn, values.device)
    print("Finished!")
    torch.save(model, values.modelpath())