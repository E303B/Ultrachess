import torch
from stockfish import Stockfish
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from submodules.tools import fromFenToTensor
from submodules import values
from trainer import NeuralNetwork
from trainer import ChessDataset
import chess
from math import floor

model = torch.load(values.modelpath(), weights_only=False)
model.eval()
if __name__ == "__main__":
    while True:
        fen = input("Input FEN position: ")
        if len(fen)==0:
            break
        board = chess.Board(fen)
        if board.is_valid():
            dataset = ChessDataset([fen])
            dataloader = DataLoader(dataset, batch_size=1)
            with torch.no_grad():
                for x, y in dataloader:
                    pred = model(x)
                    print(pred[0])
                    for i in range(4):
                        pred[0][i]*=64
                        pred[0][i]=floor(abs(pred[0][i]))
                    pred[0][4]*=6
                    pred[0][4]=floor(pred[0][4])
                    print(pred)
                    move=""
                    move+=chess.FILE_NAMES[int(pred[0][0].item())]
                    move+=chess.RANK_NAMES[int(pred[0][1].item())]
                    move+=chess.FILE_NAMES[int(pred[0][2].item())]
                    move+=chess.RANK_NAMES[int(pred[0][3].item())]
                    if pred[0][4].item()>0:
                        move+=chess.piece_symbol(int(pred[0][4].item()))
                    print(move)
        