import chess
import torch

def reverse_bits(n, num_bits=32):
    """
    Reverses all bits from left to right
    """
    result = 0
    for i in range(num_bits):
        result <<= 1
        result |= (n & 1)
        n >>= 1
    return result

def rs(n):
    """
    Reverses chess square 
    """
    return 63-n

def reverseBoard(fen: str)->str:
    """
    Reverses current chess board, as if playing for reverse color
    """
    old_board = chess.Board(fen)
    new_board = chess.Board.empty()
    new_board.castling_rights |= reverse_bits(old_board.castling_rights, 4)
    if old_board.ep_square:
        new_board.ep_square=rs(old_board.ep_square)
    new_board.halfmove_clock=old_board.halfmove_clock
    new_board.fullmove_number=old_board.fullmove_number
    new_board.turn = not old_board.turn
    for i in range(64):
        piece = old_board.piece_at(rs(i))
        if piece:
            piece.color = not piece.color
        new_board.set_piece_at(i, piece)
    return new_board.fen()

def reverseMove(move:str)->str:
    if len(move)>4:
        result = chess.square_name(rs(chess.parse_square(move[0:2])))+chess.square_name(rs(chess.parse_square(move[2:4])))+move[4]
    else:
        result = chess.square_name(rs(chess.parse_square(move[0:2])))+chess.square_name(rs(chess.parse_square(move[2:4])))
    return result

def fromFenToTensor(fen:str):
    posshape = (9, 8, 2)
    postensor = torch.zeros(posshape)
    temp_board = chess.Board()
    temp_board.set_fen(fen)
    if temp_board.turn==chess.BLACK:
        temp_board.set_fen(reverseBoard(temp_board.fen()))
    for i in range(8):
        for j in range(8):
            piece = temp_board.piece_type_at(chess.square(i, j))
            color = temp_board.color_at(chess.square(i, j))
            if piece:
                postensor[i][j][0] = piece
                postensor[i][j][1] = 1 if color else -1
            else:
                postensor[i][j][0]=0
                postensor[i][j][1]=0
    postensor[8][0][0]=1 if (temp_board.castling_rights&chess.BB_A1>0) else 0
    postensor[8][0][1]=1 if (temp_board.castling_rights&chess.BB_H1>0) else 0
    postensor[8][1][0]=1 if (temp_board.castling_rights&chess.BB_A8>0) else 0
    postensor[8][1][1]=1 if (temp_board.castling_rights&chess.BB_H8>0) else 0
    if temp_board.ep_square:
        postensor[8][2][0]=chess.square_file(temp_board.ep_square)
        postensor[8][2][1]=chess.square_rank(temp_board.ep_square)
    postensor[8][3][0]=temp_board.halfmove_clock
    postensor[8][3][1]=temp_board.fullmove_number
    postensor.requires_grad_()
    return postensor

def fromMoveToTensor(move:str, switch=False):
    if switch:
        move = reverseMove(move)
    moveshape = (5)
    movetensor = torch.zeros(moveshape)
    startmove = chess.parse_square(move[0:2])
    endmove = chess.parse_square(move[2:4])
    movetensor[0] = chess.square_file(startmove)/8
    movetensor[1] = chess.square_rank(startmove)/8
    movetensor[2] = chess.square_file(endmove)/8
    movetensor[3] = chess.square_rank(endmove)/8
    if len(move)>4:
        movetensor[4] = chess.Piece.from_symbol(move[4]).piece_type/6
    movetensor.requires_grad_()
    return movetensor