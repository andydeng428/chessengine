#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <cstdint>
#include <unordered_map>

// Enums

enum {
    a8, b8, c8, d8, e8, f8, g8, h8,
    a7, b7, c7, d7, e7, f7, g7, h7,
    a6, b6, c6, d6, e6, f6, g6, h6,
    a5, b5, c5, d5, e5, f5, g5, h5,
    a4, b4, c4, d4, e4, f4, g4, h4,
    a3, b3, c3, d3, e3, f3, g3, h3,
    a2, b2, c2, d2, e2, f2, g2, h2,
    a1, b1, c1, d1, e1, f1, g1, h1, no_sq
};
enum { white, black, both };
enum { P, N, B, R, Q, K, p, n, b, r, q, k };
enum { wk = 1, wq = 2, bk = 4, bq = 8 };
enum { rook, bishop };

enum { allMoves, onlyCaptures };


extern const std::string squareToCoordinates[64];

extern const std::string asciiPieces;

extern const std::string unicodePieces[12];

extern const std::unordered_map<char, int> charPieces;

extern const std::unordered_map<int, char> promotedPieces;

extern const std::unordered_map<int, int> indexToMoveCaptureCP;

extern const std::unordered_map<int,int> indexToMoveCP;

extern const std::unordered_map<int, int> moveToIndexCaptureCP;

extern const std::unordered_map<int,int> moveToIndexCP;

extern unsigned int randomState;

unsigned int getRandomU32BitNumber();
uint64_t getRandomU64BitNumber();
uint64_t generateMagicNumber();

int getTimeMs();



#endif // UTIL_HPP