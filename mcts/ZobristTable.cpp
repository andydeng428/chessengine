#include "ZobristTable.hpp"
#include "BitBoard.hpp"
#include "Util.hpp"
#include "Search.hpp"
#include <iostream>
#include <cstdint>

// Definitions for global variables
uint64_t pieceKeys[12][64];
uint64_t enpassantKeys[64];
uint64_t castleKeys[16];
uint64_t sideKey;
tt hashTable[hashSize];

void initRandomKeys(){
    // update pseudo random number state
    randomState = 1804289383;

    // loop over piece codes
    for (int piece = P ; piece <= k ; piece++){
        for (int square = 0; square < 64; square++){
            // init random piece keys
            pieceKeys[piece][square] = getRandomU64BitNumber();

        }
    }

    // loop over board squares
    for (int square = 0; square < 64 ; square++){
        // init random enpassant keys
        enpassantKeys[square] = getRandomU64BitNumber();
    }

    // loop over castling keys
    for (int index = 0 ; index < 16 ; index++){
        castleKeys[index] = getRandomU64BitNumber();
    }

    // init random side key
    sideKey = getRandomU64BitNumber();
}

uint64_t generateHashKey(const BoardState& board){
    // final hask key
    uint64_t finalKey = 0ULL;

    // temp piece bitbaord copy
    uint64_t bitboard;

    for (int piece = P; piece <= k ; piece++){
        bitboard = board.bitBoard[piece];
        while (bitboard){
            // init square occupied by the piece
            int square = getLeastSigBitIndex(bitboard);

            // hash peice
            finalKey ^= pieceKeys[piece][square];

            // pop lsbd
            popBit(bitboard, square);
        }
    }
    // if enpassant is on board
    if (board.enPassant != no_sq){
        finalKey ^= enpassantKeys[board.enPassant];
    }

    // hash castling rights
    finalKey ^= castleKeys[board.castle];

    // hash the side only if black is to move
    if (board.side == black){
        finalKey ^= sideKey;
    }

    // return generated hash key
    return finalKey;
}

// clear TT (hash table)
void clearHashTable()
{
    // loop over TT elements
    for (int index = 0; index < hashSize; index++)
    {
        // reset TT inner fields
        hashTable[index].hashKey = 0;
        hashTable[index].depth = 0;
        hashTable[index].flag = 0;
        hashTable[index].score = 0;
    }
}


// read hash entry data
int readHashEntry(const BoardState& board,int depth)
{
    // create a TT instance pointer to particular hash entry storing
    // the scoring data for the current board position if available
    tt *hashEntry = &hashTable[board.hashKey % hashSize];
    // hash table is global
    
    // make sure we're dealing with the exact position we need
    if (hashEntry->hashKey == board.hashKey)
    {
        // extract stored score from TT entry
        int score = hashEntry->score;
        //extract score independant form the actual path from root node
        // retrieve score independant from actual path
        if (score < -mateScore) score += ply;
        if (score > mateScore) score -= ply;

        // make sure that we match the exact depth our search is now at
        if (hashEntry->depth >= depth)
        {
            // match the exact (PV node) score 
            if (hashEntry->flag == hashFlagExact)
                // return exact (PV node) score
                {return score;}
        }
    }
    
    // if hash entry doesn't exist
    return noHashEntry;
}


void writeHashEntry(const BoardState& board, int score, int depth, int hashFlag)
{
    // create a TT instance pointer to particular hash entry storing
    // the scoring data for the current board position if available
    tt *hashEntry = &hashTable[board.hashKey % hashSize];

    // store score indepedent from actual path from root node to current position.
    if (score < -mateScore) score -= ply;
    if (score > mateScore) score += ply;

    // write hash entry data 
    hashEntry->hashKey = board.hashKey;
    hashEntry->score = score;
    hashEntry->flag = hashFlag;
    hashEntry->depth = depth;
}
