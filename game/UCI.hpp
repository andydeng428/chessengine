#ifndef UCI_HPP
#define UCI_HPP

#include "BitBoard.hpp"


// External declarations for global variables
extern int quit;
extern int movesToGo;
extern int moveTime;
extern int timeUCI;
extern int inc;
extern int startTime;
extern int stopTime;
extern int timeSet;
extern int stopped;

// Function declarations
int inputWaiting();
void readInput();
void parsePosition(BoardState& board, char *command);
void parseGo(BoardState& board, char *command);
void uciLoop(BoardState& board);
void communicate();

#endif // UCI_HPP
