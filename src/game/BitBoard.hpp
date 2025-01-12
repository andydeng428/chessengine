#ifndef BITBOARDSTATE_HPP
#define BITBOARDSTATE_HPP
#include <cstdint>
#include "Util.hpp"

class Moves {
public:
    static inline uint32_t encodeMove(uint32_t source, uint32_t target, uint32_t piece, uint32_t promoted, 
            uint32_t capture, uint32_t doublePush, uint32_t enPassant, uint32_t castling) { return (source) | (target << 6) | (piece << 12) | (promoted << 16) | (capture << 20) | (doublePush << 21) | (enPassant << 22) | (castling << 23); }

    static inline uint32_t getMoveSource(uint32_t move) { return move & 0x3f; }
    static inline uint32_t getMoveTarget(uint32_t move) { return (move & 0xfc0) >> 6; }
    static inline uint32_t getMovePiece(uint32_t move) { return (move & 0xf000) >> 12; }
    static inline uint32_t getMovePromoted(uint32_t move) { return (move & 0xf0000) >> 16; }
    static inline bool getMoveCapture(uint32_t move) { return move & 0x100000; }
    static inline bool getMoveDouble(uint32_t move) { return move & 0x200000; }
    static inline bool getMoveEnPassant(uint32_t move) { return move & 0x400000; }
    static inline bool getMoveCastling(uint32_t move) { return move & 0x800000; }

    void addMove(int move);
    static void printMove(int move);
    static int moveToIndex(int move);
    static int indexToMove(int index, Moves* moveList);
    void printMoveList();
private:
    std::vector<int> moves;
};

struct BitBoardState{
    static inline bool getBit(uint64_t b, int s){return (b & (1ULL << s)) != 0;}
    static inline void setBit(uint64_t& b, int s){b |= (1ULL << s);}
    static inline void popBit(uint64_t& b, int s){b &= ~(1ULL << s);}
    static inline int countBits(uint64_t b){return __builtin_popcountll(b);}
    static inline int getLeastSigBitIndex(uint64_t b){return b ? __builtin_ctzll(b) : -1;}

    uint64_t bitBoard[12] = {0};
    uint64_t occupancies[3] = {0};
    int side = white;
    int enPassant = no_sq;
    int castle = 0;
    uint64_t hashKey = 0;
};

class Magics{
public:
    Magics();

    // precomputated masks
    static uint64_t pawnAttacks[2][64];
    static uint64_t knightAttacks[64];
    static uint64_t kingAttacks[64];
    static uint64_t bishopAttacks[64][512];
    static uint64_t rookAttacks[64][4096];
    static uint64_t bishopMasks[64];
    static uint64_t rookMasks[64];

    static int isSquareAttacked(const BitBoardState& board, int square, int side);
    void initLeaperAttacks();
    void initSliderAttacks(int bishop);

    uint64_t getBishopAttacks(int square, uint64_t occupancy);
    uint64_t getRookAttacks(int square, uint64_t occupancy);
    uint64_t getQueenAttacks(int square, uint64_t occupancy);

private:
    const uint64_t bishopMagicNumbers[64];
    const uint64_t rookMagicNumbers[64];

    const uint64_t NOT_IN_A_FILE;
    const uint64_t NOT_IN_H_FILE;
    const uint64_t NOT_IN_AB_FILE;
    const uint64_t NOT_IN_HG_FILE;

    const int bishopRelevantBit[64];
    const int rookRelevantBit[64];

    void initMagicNumbers();
    uint64_t maskPawnAttacks(int side, int square);
    uint64_t maskKnightAttacks(int square);
    uint64_t maskKingAttacks(int square);
    uint64_t maskBishopAttacks(int square);
    uint64_t maskRookAttacks(int square);
    uint64_t bishopAttacksOnTheFly(int square, uint64_t block);
    uint64_t rookAttacksOnTheFly(int square, uint64_t block);
    uint64_t setOccupancy(int index, int bitsInMask, uint64_t attackMask);
    uint64_t findMagicNumber(int square, int relevantBits, int bishops);
};

#endif // BITBOARDSTATE_HPP