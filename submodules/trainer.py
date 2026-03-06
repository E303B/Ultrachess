import torch
import chess
from stockfish import Stockfish
import values
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from pathlib import Path
import tools
from torch import nn
from math import floor

class ChessDataset(Dataset):
    def __init__(self, path):
        if not Path(path).is_file():
            return None
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
    def __len__(self):
        return len(self.chesspositions)
    
    def __getitem__(self, i):
        position = self.chesspositions[i]
        temp_board = chess.Board()
        temp_board.set_fen(position)
        return tools.fromFenToTensor(self.chesspositions[i]), tools.fromMoveToTensor(self.moves[i], (temp_board.turn==chess.BLACK))
    
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
            nn.Linear(512, 5),
            nn.Softmax(dim=1)
        )
 
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (position, move) in enumerate(dataloader):
        position = position.to(device)
        move = move.to(device)

        pred = model(position)
        loss = loss_fn(pred, move)

        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * values.batch_size + len(position)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for position, move in dataloader:
            position = position.to(device)
            move = move.to(device)
 
            pred = model(position)
            test_loss += loss_fn(pred, move).item()
            for i in range(size):
                for j in range(4):
                    pred[i][j]*=8
                    pred[i][j]=floor(pred[i][j].item())
                    pred[i][j]/=8
                pred[i][4]*=6
                pred[i][4]=round(pred[i][4].item())
                pred[i][4]/=6
                a = abs(1-abs(pred[i].type(torch.float).sum() - move[i].type(torch.float).sum()).item())
                correct += a
 
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>4f}%, Avg loss: {test_loss:>8f} \n")
    
if __name__ == '__main__':
    
    training_data = ChessDataset("data/default.data")
    testing_data = ChessDataset("data/test.data")
    training_dataloader = DataLoader(training_data, batch_size=values.batch_size, shuffle=True)
    testing_dataloader = DataLoader(testing_data, batch_size=values.batch_size, shuffle=True)
    model = NeuralNetwork()
    model.to(device=values.device)
     
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=values.learning_rate)
    
    for t in range(values.epochs):
        print(f"Epoch: {t+1}")
        train_loop(training_dataloader, model, loss_fn, optimizer, values.device)
        test_loop(testing_dataloader, model, loss_fn, values.device)
    print("Finished!")
    torch.save(model, "models/"+values.modelname+".pth")