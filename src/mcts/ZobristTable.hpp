#ifndef ZOBRISTTABLE_HPP
#define ZOBRISTTABLE_HPP

#include <cstdint>
#include "game/BitBoard.hpp"

// External declarations for global variables
extern uint64_t pieceKeys[12][64];
extern uint64_t enpassantKeys[64];
extern uint64_t castleKeys[16];
extern uint64_t sideKey;

// Constants for hash table size
constexpr int hashSize = 800000; // transposition table size
constexpr int noHashEntry = 100000;

// Transposition table hash flags
constexpr int hashFlagExact = 0;
constexpr int hashFlagAlpha = 1;
constexpr int hashFlagBeta = 2;

// Transposition table data structure
typedef struct {
    uint64_t hashKey;   // "almost" unique chess position identifier
    int depth;          // current search depth
    int flag;           // flag the type of node (fail-low/fail-high/PV) 
    int score;          // score (alpha/beta/PV)
} tt;

// External declaration for hashTable instance
extern tt hashTable[hashSize];

// Function declarations
void initRandomKeys();
uint64_t generateHashKey(const BoardState& board);
void clearHashTable();
int readHashEntry(const BoardState& board, int depth);
void writeHashEntry(const BoardState& board, int score, int depth, int hashFlag);

#endif // ZOBRISTTABLE_HPP
