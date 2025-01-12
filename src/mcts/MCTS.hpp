#ifndef MCTS_HPP
#define MCTS_HPP

#include <iostream>
#include <unordered_map>
#include <cmath>
#include <memory>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include "BitBoard.hpp"
#include "MoveGen.hpp"
#include "MaskGen.hpp"
#include "Util.hpp"
#include "NeuralNet.hpp"
#include "ZobristTable.hpp"

enum TerminalType {
    None,     
    Checkmate,
    Stalemate,
};

struct MCTSNode {
    MCTSNode* parent;
    BoardState nodeBoard;
    std::vector<std::unique_ptr<MCTSNode>> childrenList;
    Moves nodeMoveList;

    // MCTS statistics
    std::atomic<int> visitCount;  
    std::atomic<double> valueSum;
    std::atomic<int> virtualLoss;
    double priorProb;
    int priorMove;
    int depth;
    TerminalType terminalType;
    double simulatedValue;
    std::mutex nodeMutex;

    MCTSNode(MCTSNode* parentNode, const BoardState& curBoard);

    ~MCTSNode();

    bool isFullyExpanded();
    bool isTerminal();
};

double getUCB(MCTSNode* child, MCTSNode* node);
MCTSNode* MCTSselect(MCTSNode* node);
MCTSNode* MCTSselectFinal(MCTSNode* node);
void MCTSexpand(MCTSNode* node);
double MCTSsimulate(MCTSNode* node);
void MCTSbackpropogate(MCTSNode* node, double value);
MCTSNode* MCTSsearch(MCTSNode* root, int timeLimit);
void searchThread(MCTSNode* root, int timeLimit, std::atomic<bool>& stopFlag);n
int MCTSmain(const BoardState& board);

#endif // MCTS_HPP
