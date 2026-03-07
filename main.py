import torch
from stockfish import Stockfish
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from submodules.tools import findClosestLegalMove
from submodules.tools import reverseMove
from submodules import values
from trainer import NeuralNetwork
from trainer import ChessDataset
import chess
from math import floor

model = torch.load(values.modelpath(), weights_only=False)
model.eval()

def calculateBestMove(fen:str)->str|None:
    result = "a1a1"
    if len(fen)==0:
        return None
    board = chess.Board(fen)
    if not board.is_valid():
        return None
    dataset = ChessDataset([fen])
    dataloader = DataLoader(dataset, batch_size=1)
    with torch.no_grad():
        for x, _ in dataloader:
            pred = model(x)
            for i in range(4):
                pred[0][i]*=64
                pred[0][i]=floor(abs(pred[0][i]))
            pred[0][4]*=6
            pred[0][4]=floor(pred[0][4])
            move=""
            move+=chess.FILE_NAMES[int(pred[0][0].item())]
            move+=chess.RANK_NAMES[int(pred[0][1].item())]
            move+=chess.FILE_NAMES[int(pred[0][2].item())]
            move+=chess.RANK_NAMES[int(pred[0][3].item())]
            if pred[0][4].item()>0:
                move+=chess.piece_symbol(int(pred[0][4].item()))
            fromSquare = chess.parse_square(move[0:2])
            toSquare = chess.parse_square(move[2:4])
            fromPiece =  board.piece_at(fromSquare)
            if fromPiece and fromPiece.color:
                moveObj = chess.Move.from_uci(move if fromPiece.piece_type==chess.PAWN else move[0:4])
                if board.is_legal(moveObj):
                    result = move
                else:
                    closestMove=findClosestLegalMove(board, fromSquare, toSquare)
                    if closestMove:
                        result= closestMove.uci()
                    else:
                        return None
            else:
                closestMove=findClosestLegalMove(board, fromSquare, toSquare)
                if closestMove:
                    result = closestMove.uci()
                else:
                    return None
    if board.turn == chess.BLACK:
        return reverseMove(result)
    else:
        return result
    

if __name__ == "__main__":
    while True:
        fen = input("Input FEN position: ")
        if len(fen)==0:
            break
        print(calculateBestMove(fen))