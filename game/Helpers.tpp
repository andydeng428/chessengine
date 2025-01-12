// In progress

#ifndef HELPERS_TPP
#define HELPERS_TPP

#include "MyHelpers.hpp"

#include "Board.hpp"
#include "Moves.hpp"
#include "Magic.hpp"
#include "Utils.hpp"

template <typename AttackFunc>
void generateMovesForPiece( Board& board, Moves* moveList, int piece, uint64_t pieceBitboard, AttackFunc attackFunc ) {
    while (pieceBitboard)
    {
        int sourceSquare = getLeastSigBitIndex(pieceBitboard);
        uint64_t attacks = attackFunc(sourceSquare, board.occupancies[both]);
        if (board.side == white)
            attacks &= ~board.occupancies[white];
        else
            attacks &= ~board.occupancies[black];

        while (attacks)
        {
            int targetSquare = getLeastSigBitIndex(attacks);

            bool isCapture = false;
            if (board.side == white) {
                isCapture = getBit(board.occupancies[black], targetSquare);
            } else {
                isCapture = getBit(board.occupancies[white], targetSquare);
            }
            addMove(
                moveList,
                encodeMove( sourceSquare, targetSquare, piece, 0, isCapture ? 1 : 0, 0, 0, 0 )
            );
            popBit(attacks, targetSquare);
        }

        popBit(pieceBitboard, sourceSquare);
    }
}

#endif // HELPERS_TPP
