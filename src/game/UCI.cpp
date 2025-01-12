#include "UCI.hpp"
#include <iostream>
#include <unistd.h>
#include <chrono>

#include "MCTS.hpp"


// Definitions for global variables
int quit = 0;
int movesToGo = 30;
int moveTime = -1;
int timeUCI = -1;
int inc = 0;
int startTime = 0;
int stopTime = 0;
int timeSet = 0;
int stopped = 0;


int inputWaiting()
{
    fd_set readfds;
    struct timeval tv;
    FD_ZERO (&readfds);
    FD_SET (fileno(stdin), &readfds);
    tv.tv_sec=0; tv.tv_usec=0;
    select(16, &readfds, 0, 0, &tv);
    return (FD_ISSET(fileno(stdin), &readfds));
}

// read GUI/user input
void readInput()
{
    // bytes to read holder
    int bytes;
    
    // GUI/user input
    char input[256] = "", *endC;

    // "listen" to STDIN
    if (inputWaiting())
    {
        // tell engine to stop calculating
        stopped = 1;
        
        // loop to read bytes from STDIN
        do
        {
            // read bytes from STDIN
            bytes = read(fileno(stdin), input, 256);
        }
        
        // until bytes available
        while (bytes < 0);
        
        // searches for the first occurrence of '\n'
        endC = strchr(input, '\n');
        
        // if found new line set value at pointer to 0
        if (endC) *endC = 0;
        
        // if input is available
        if (strlen(input) > 0)
        {
            // match UCI "quit" command
            if (!strncmp(input, "quit", 4))
            {
                // tell engine to terminate execution    
                quit = 1;
            }

            // match UCI "stop" command
            else if (!strncmp(input, "stop", 4))
            {
                // tell engine to terminate execution
                quit = 1;
            }
        }   
    }
}

// a bridge function to interact between search and GUI input
void communicate() {
    // if time is up break here
    if (timeSet == 1 && getTimeMs() > stopTime) {
        // tell engine to stop calculating
        stopped = 1;
    }
    
    // read GUI input
    readInput();
}

//parse UCI position command
void parsePosition(Game& game, char *command){
    // shift towards when the command starts
    command += 9;

    char* currentChar = command;

    // parse UCI "startpos" command
    if (strncmp(command, "startpos", 8) == 0)
        // init chess board with start position
        game.parseFen(startPosition);
    
    // parse UCI "fen" command 
    else
    {
        // make sure "fen" command is available within command string
        currentChar = strstr(command, "fen");
        
        // if no "fen" command is available within command string
        if (currentChar == NULL)
            // init chess board with start position
            game.parseFen(startPosition);
            
        // found "fen" substring
        else
        {
            // shift pointer to the right where next token begins
            currentChar += 4;
            
            // init chess board with position from FEN string
            game.parseFen(currentChar);
        }
    }

    currentChar = strstr (command, "moves");

        // moves available
    if (currentChar != NULL)
    {
        // shift pointer to the right where next token begins
        currentChar += 6;
        
        // loop over moves within a move string
        while(*currentChar)
        {
            // parse next move
            int move = parseMove(board, currentChar);
            
            // if no more moves
            if (move == 0)
                // break out of the loop
                break;

            // increment repetition index
            repetitionIndex++;

            // write hash key into repetiion table
            repetitionTable[repetitionIndex] = board.hashKey;
            
            // make move on the chess board
            makeMove(board, move, allMoves);
            
            // move current character mointer to the end of current move
            while (*currentChar && *currentChar != ' ') currentChar++;
            
            // go to the next move
            currentChar++;
        }
        
    }
    printBoard(board);
}

void parseGo(BoardState& board, char *command)
{
    // init depth
    int depth = -1;
    
    // init character pointer to the current depth argument
    char *argument = NULL;

    // infinite search
    if ((argument = strstr(command, "infinite"))) {}

    // match UCI "binc" command
    if ((argument = strstr(command, "binc")) && board.side == black)
        // parse black time increment
        inc = atoi(argument + 5);

    // match UCI "winc" command
    if ((argument = strstr(command, "winc")) && board.side == white)
        // parse white time increment
        inc = atoi(argument + 5);

    // match UCI "wtime" command
    if ((argument = strstr(command, "wtime")) && board.side == white)
        // parse white time limit
        timeUCI = atoi(argument + 6);

    // match UCI "btime" command
    if ((argument = strstr(command, "btime")) && board.side == black)
        // parse black time limit
        timeUCI = atoi(argument + 6);

    // match UCI "movestogo" command
    if ((argument = strstr(command, "movestogo")))
        // parse number of moves to go
        movesToGo = atoi(argument + 10);

    // match UCI "movetime" command
    if ((argument = strstr(command, "movetime")))
        // parse amount of time allowed to spend to make a move
        moveTime = atoi(argument + 9);

    // match UCI "depth" command
    if ((argument = strstr(command, "depth")))
        // parse search depth
        depth = atoi(argument + 6);

    // if move time is not available
    if(moveTime != -1)
    {
        // set time equal to move time
        timeUCI = moveTime;

        // set moves to go to 1
        movesToGo = 1;
    }

    // init start time
    startTime = getTimeMs();

    // init search depth
    depth = depth;

    // if time control is available
    if(timeUCI != -1)
    {
        // flag we're playing with time control
        timeSet = 1;

        // set up timing
        timeUCI /= movesToGo;
        
        if (timeUCI > 1500) timeUCI -= 50;

        stopTime = startTime + timeUCI + inc;
    }

    // if depth is not available
    if(depth == -1)
        // set depth to 64 plies (takes ages to complete...)
        depth = 64;

    // print debug info
    printf("time:%d start:%d stop:%d depth:%d timeSet:%d\n",
           timeUCI, startTime, stopTime, depth, timeSet);

    // search position
    searchPosition(board, depth);
}

void uciLoop(BoardState &board){
    //clear stdin and stout bugers
    setbuf(stdin, NULL);
    setbuf(stdout, NULL);

    //define user / gui input buffer
    char input[2000];

    //pengine info
    std::cout<< "idname BABYBLUE\n";
    std::cout<< "id name andookie\n";
    std::cout<< "uciok\n";

    while(1){
        // reset user /GUI input
        memset(input, 0, sizeof(input));
        
        // make sure output reaches the GUI
        fflush(stdout);
        
        // get user / GUI input
        if (!fgets(input, 2000, stdin))
            // continue the loop
            continue;
        
        // make sure input is available
        if (input[0] == '\n')
            // continue the loop
            continue;
        
        // parse UCI "isready" command
        if (strncmp(input, "isready", 7) == 0)
        {
            std::cout << "readyok\n";
            continue;
        }
        
        // parse UCI "position" command
        else if (strncmp(input, "position", 8) == 0){
            // call parse position function
            parsePosition(board, input);
        }
        else if (strncmp(input, "ucinewgame", 10) == 0){
            // call parse position function
            parsePosition(board, "position startpos");
            // clear hash table
            clearHashTable();
        }
        else if (strncmp(input, "go", 2) == 0){
            const BoardState tempBoard = board;
            MCTSmain(tempBoard);
        }
    }
}

/*

void uciLoop(BoardState &board){
    //clear stdin and stout bugers
    setbuf(stdin, NULL);
    setbuf(stdout, NULL);

    //define user / gui input buffer
    char input[2000];

    //pengine info
    std::cout<< "idname BABYBLUE\n";
    std::cout<< "id name andookie\n";
    std::cout<< "uciok\n";

    // uci loop
    while(1){
        // reset user /GUI input
        memset(input, 0, sizeof(input));
        
        // make sure output reaches the GUI
        fflush(stdout);
        
        // get user / GUI input
        if (!fgets(input, 2000, stdin))
            // continue the loop
            continue;
        
        // make sure input is available
        if (input[0] == '\n')
            // continue the loop
            continue;
        
        // parse UCI "isready" command
        if (strncmp(input, "isready", 7) == 0)
        {
            std::cout << "readyok\n";
            continue;
        }
        
        // parse UCI "position" command
        else if (strncmp(input, "position", 8) == 0){
            // call parse position function
            parsePosition(board, input);

            // clear hash table
            clearHashTable();
        }
        // parse UCI "ucinewgame" command
        else if (strncmp(input, "ucinewgame", 10) == 0){
            // call parse position function
            parsePosition(board, "position startpos");
            // clear hash table
            clearHashTable();
        }

        
        // parse UCI "go" command
        else if (strncmp(input, "go", 2) == 0)
            // call parse go function
            parseGo(board, input);
        
        // parse UCI "quit" command
        else if (strncmp(input, "quit", 4) == 0)
            // quit from the chess engine program execution
            break;
        
        // parse UCI "uci" command
        else if (strncmp(input, "uci", 3) == 0)
        {
            //pengine info
            std::cout<< "idname BABYBLUE\n";
            std::cout<< "id name andookie\n";
            std::cout<< "uciok\n";
        }

    }
}

*/

