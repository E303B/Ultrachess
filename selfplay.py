import chess
import main
from stockfish import Stockfish
from submodules import values
import chess.pgn
if __name__ == "__main__":
    playwithstockfish=False
    turn = chess.BLACK
    stockfish = Stockfish(values.stockfishpath)
    stockfish.set_skill_level(0)
    stockfish.make_moves_from_start(["e2e4"])
    board = chess.Board(stockfish.get_fen_position())
    game = chess.pgn.Game()
    game.headers["Event"] = "Ultrachess selfplay"
    game.headers["White"] = "Stockfish lowest level"
    game.headers["Black"] = "Ultrachess"
    previousNode = game.add_variation(chess.Move.from_uci("e2e4"))
    while True:
        stockfish.set_fen_position(board.fen())
        print(f"Current turn: "+("WHITE" if turn else "BLACK"), stockfish.get_board_visual(), sep="\n")
        if turn==chess.BLACK and  playwithstockfish:
            best_move = stockfish.get_best_move()
        else:
            best_move = main.calculateBestMove(board.fen())
        if not best_move:
            print("Failed to generate move")
            break
        print("Best move generated: ", best_move)
        move_obj = chess.Move.from_uci(best_move)
        board.push(move_obj)
        outcome = board.outcome()
        previousNode = previousNode.add_variation(move_obj)
        if outcome:
            if outcome.result=="1-0":
                previousNode.comment = "White won"
                print("White won!")
            elif outcome.result=="0-1":
                previousNode.comment = "Black won"
                print("Black won!")
            else:
                previousNode.comment = "Draw by stalemate"
                print("Draw by stalemate")
            break
        if board.is_fivefold_repetition() or board.is_fifty_moves():
            print("Draw by "+("repetition" if board.can_claim_threefold_repetition() else "fifty moves rule"))
            previousNode.comment = "Draw by "+("repetition" if board.can_claim_threefold_repetition() else "fifty moves rule")
            break
        turn = not turn
    print(game, file=open("output.txt", "w+"), end="\n\n")