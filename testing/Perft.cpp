#include "Perft.hpp"
#include <iostream>
#include <sys/time.h>


int getTimeMS(){
    struct timeval timeValue;
    gettimeofday(&timeValue, NULL);
    return timeValue.tv_sec * 1000 + timeValue.tv_sec / 1000;
}

// perft driver
void perftDriver(BoardState& board, int depth){
    //base case
    if (depth == 0){
        nodes++;
        return;
    }

    Moves moveList[1];
    generateMove(board, moveList);

    for (int moveCount = 0 ; moveCount < moveList->count ; moveCount ++){
        copyBoard();

        if (!makeMove(board, moveList -> moves[moveCount], allMoves)){
            continue;
        }
        // call perft drivesr recursively
        perftDriver(board, depth - 1);
        restoreBoard();
    }
}

void perftTest (BoardState& board, int depth){

    std::cout << "\n    performance tests\n";
    
    Moves moveList[1];
    
    generateMove(board, moveList);

    long start = getTimeMS();

    for (int moveCount = 0 ; moveCount < moveList->count ; moveCount ++){
        std::cout<< "arewestuckhere \n";
        copyBoard();

        if (!makeMove(board, moveList -> moves[moveCount], allMoves)){
            continue;
        }
        // call perft drivesr recursively

        // cummulative nodes
        long cummulativeNodes = nodes;
        std::cout<< "ahereeeeee \n";
        perftDriver(board, depth - 1);
        std::cout<< "ahhhhhhh \n";
        long oldNodes = nodes - cummulativeNodes;
        
        restoreBoard();

        std::cout<<"move: ";
        printMove(moveList -> moves[moveCount]);
        std::cout<< (getMovePromoted(moveList->moves[moveCount]) ? promotedPieces.at(getMovePromoted(moveList->moves[moveCount])) : ' ');
        std::cout<< " node: " << oldNodes << "\n";
    }

    std::cout<< "\n depth:" << depth;
    std::cout<< "\n nodes:" << nodes;
    std::cout<< "\n time:" << getTimeMS() - start<< " \n\n\n\n";
}