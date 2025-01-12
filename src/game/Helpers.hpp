// In progress

#ifndef HELPERS_TPP
#define HELPERS_TPP

#include <cstdint>

struct Board;
struct Moves;

template <typename AttackFunc>
void generateMovesForPiece( Board& board, Moves* moveList, int piece, uint64_t pieceBitboard, AttackFunc attackFunc);

#endif // HELPERS_TPP