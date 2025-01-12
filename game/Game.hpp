#ifndef GAME_HPP
#define GAME_HPP

#include <iostream>
#include <cstdint>
#include <string>
#include <vector> 
#include "BitBoard.hpp"
#include "Util.hpp"
#include "mcts/MCTS.hpp"


class Game{
public:
    void searchBestMove();
    void parseFen(std::string fenC);
    int makeMove(int move, int moveFlag);
    void generateMove(Moves *moveList);
    void printBoard();

    static void printBitboard(uint64_t bitBoard);
    static int parseMove(std::string moveString);
    void printAttackedSquares(int side);

private:
    BitBoardState board;
    Magics magic;
    MCTS mcts;
    int ply = 0;
    uint64_t repetitionTable[1000] = {0};
    int repetitionIndex = 0;

    // move generator elements
    static const int castlingRights[64];
};



#endif // GAME_HPP