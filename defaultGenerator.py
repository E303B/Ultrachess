import chess
from stockfish import Stockfish
import values
import os
from multiprocessing import Pool
import shutil


def playParty(startMove=""):
    file = open("temp_files/"+startMove+".data", "a+")
    stockfish = Stockfish(path=values.stockfishpath)
    stockfish.set_elo_rating(2500)
    stockfish.set_skill_level(20)
    if len(startMove)>0:
        stockfish.make_moves_from_start([startMove])
    else:
        stockfish.make_moves_from_start()
    whiteTurn = True
    board = chess.Board(stockfish.get_fen_position())
    while True:
        board = chess.Board(stockfish.get_fen_position())
        best_move=stockfish.get_best_move_time(1000) or ""
        text = board.fen()+":"+best_move+"\n"
        file.write(text)
        move = chess.Move.from_uci(best_move)
        board.push(move)
        outcome = board.outcome()
        print("Playing match for starting move: "+startMove+", moves so far: "+str(board.fullmove_number))
        if outcome:
            if outcome.result=="1-0":
                print("White won!")
            elif outcome.result=="0-1":
                print("Black won!")
            else:
                print("Draw by stalemate")
            break
        if board.is_fivefold_repetition() or board.is_fifty_moves():
            print("Draw by "+("repetition" if board.can_claim_threefold_repetition() else "fifty moves rule"))
            break
        stockfish.set_fen_position(board.fen())
        whiteTurn = not whiteTurn
    file.close()
    return "temp_files/"+startMove+".data"


if __name__ == '__main__':
    defaultOutput = "data/default.data"
    legalMoves = []
    legalMoves.append("b1a3")
    legalMoves.append("b1c3")
    legalMoves.append("g1f3")
    legalMoves.append("g1h3")
    if os.path.isdir("temp_files"):
        shutil.rmtree("temp_files")
    os.mkdir("temp_files")
    for i in range(8):
        for j in range(2):
            legalMoves.append(chess.FILE_NAMES[i]+"2"+chess.FILE_NAMES[i]+("4" if j==0 else "3"))
    if os.path.exists(defaultOutput):
        os.remove(defaultOutput)
    file = open(defaultOutput, "a+")
    file.close()
    with Pool() as pool:
        results = pool.map(playParty, legalMoves)
        with open(defaultOutput, "a") as outputFile:
            for path in results:
                with open(path, "r") as file:
                    outputFile.write(file.read())
                os.remove(path)
            outputFile.close()
    print("Finished!")