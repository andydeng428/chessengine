#include "MCTS.hpp"

static std::atomic<int> nodeCount{0};
static std::atomic<int> maxDepth{0};
static std::atomic<int> terminalMoves{0};
BoardState lastTerminalBoard; // for debugging

MCTSNode::MCTSNode(MCTSNode* parentNode, const BoardState& curBoard)
    : parent(parentNode),
      nodeBoard(curBoard),
      childrenList(),
      nodeMoveList(),
      visitCount(0),
      valueSum(0.0),
      virtualLoss(0),
      priorProb(0.0),
      priorMove(0),
      depth(0),
      terminalType(None),
      simulatedValue(0.0) {

    generateMove(nodeBoard, &nodeMoveList);

    // Filter all illegal moves from the pseudo legal move list
    for (auto it = nodeMoveList.moves.begin(); it != nodeMoveList.moves.end();) {
        BoardState tmpBoard = nodeBoard;
        if (makeMove(tmpBoard, *it, allMoves) != 1) {
            it = nodeMoveList.moves.erase(it);
            nodeMoveList.count--;
        } else {
            ++it;
        }
    }

    // Check for stalemate and checkmate
    if (nodeMoveList.count == 0) {
        bool inCheck = isSquareAttacked(
            nodeBoard,
            (nodeBoard.side == white)
                ? getLeastSigBitIndex(nodeBoard.bitBoard[K])
                : getLeastSigBitIndex(nodeBoard.bitBoard[k]),
            (nodeBoard.side == white) ? black : white
        );
        if (inCheck) {
            terminalType = Checkmate;
            terminalMoves++;
            lastTerminalBoard = nodeBoard;
        } else {
            terminalType = Stalemate;
            terminalMoves++;
            lastTerminalBoard = nodeBoard;
        }
    }
    if (parentNode != nullptr) {
        depth = parentNode->depth + 1;
        int oldDepth = maxDepth.load();
        while (depth > oldDepth) {
            maxDepth.compare_exchange_weak(oldDepth, depth);
        }
    }

    // Increment the global node count
    nodeCount.fetch_add(1);
}

MCTSNode::~MCTSNode() {}

bool MCTSNode::isFullyExpanded() {
    return ((int)childrenList.size() == nodeMoveList.count);
}

bool MCTSNode::isTerminal() {
    return (terminalType != None);
}

double getUCB(MCTSNode* child, MCTSNode* node) {
    double c = 1.5; // exploration constant
    double childVisits = (double)child->visitCount.load(std::memory_order_relaxed); // Don't are about memory order or synchroncation
    double parentVisits = (double)node->visitCount.load(std::memory_order_relaxed);
    double qValue = (childVisits > 0.0) ? (child->valueSum.load(std::memory_order_relaxed) / childVisits) : 0.0;
    double ucbValue = qValue + c * child->priorProb * (std::sqrt(parentVisits) / (1.0 + childVisits));
    return ucbValue;
}

MCTSNode* MCTSselect(MCTSNode* node) {
    std::lock_guard<std::mutex> lock(node->nodeMutex);

    MCTSNode* bestChild = nullptr;
    double bestUCB = -1e15;

    for (auto& childPtr : node->childrenList) {
        MCTSNode* child = childPtr.get();
        double ucb = getUCB(child, node);
        if (ucb > bestUCB) {
            bestUCB = ucb;
            bestChild = child;
        }
    }

    // Add virtual loss so other threads don't pick to expand
    if (bestChild) {
        bestChild->virtualLoss.fetch_add(1, std::memory_order_relaxed);
    }
    return bestChild;
}

// Select root node's most visited child
MCTSNode* MCTSselectFinal(MCTSNode* node) {
    std::lock_guard<std::mutex> lock(node->nodeMutex);

    int maxVisit = -1;
    MCTSNode* bestChild = nullptr;

    for (auto& childPtr : node->childrenList) {
        MCTSNode* child = childPtr.get();
        int visits = child->visitCount.load(std::memory_order_relaxed);
        std::cout << "[MCTSselectFinal] Move "
                  << squareToCoordinates[getMoveSource(child->priorMove)]
                  << squareToCoordinates[getMoveTarget(child->priorMove)]
                  << " visit count is " << visits << "\n";

        if (visits > maxVisit) {
            maxVisit = visits;
            bestChild = child;
        }
    }

    if (bestChild) {
        std::cout << "\n[MCTSselectFinal] Best child after search: "
                  << squareToCoordinates[getMoveSource(bestChild->priorMove)]
                  << squareToCoordinates[getMoveTarget(bestChild->priorMove)]
                  << std::endl;
    }
    return bestChild;
}

void MCTSexpand(MCTSNode* node) {
    // Lock the node to safely update childrenList
    std::lock_guard<std::mutex> lock(node->nodeMutex);

    if (node->isFullyExpanded() || node->isTerminal()) return;

    // Get eval from NN 
    std::pair<std::vector<float>, float> nnOutput = NNevaluate(node->nodeBoard);

    for (auto m : node->nodeMoveList.moves) {
        int moveIdx = moveToIndex(m);
        float moveProb = 0.0f;
        if (moveIdx >= 0 && moveIdx < (int)nnOutput.first.size()) {
            moveProb = nnOutput.first[moveIdx];
        }
        BoardState newBoard = node->nodeBoard;
        if (makeMove(newBoard, m, allMoves) == 1) {
            // Create the child node
            std::unique_ptr<MCTSNode> child(new MCTSNode(node, newBoard));
            child->priorProb = moveProb;
            child->priorMove = m;
            node->childrenList.push_back(std::move(child));
        }
    }
}

double MCTSsimulate(MCTSNode* node) {
    if (node->isTerminal()) {
        double valueWhite = 0.0;
        if (node->terminalType == Checkmate) {
            valueWhite = (node->nodeBoard.side == white) ? -1.0 : 1.0;
        }
        if (rootSide == black) valueWhite = -valueWhite;
        return valueWhite;
    }
    {
        // Check TT table for hash
        int ttValue = readHashEntry(node->nodeBoard, node->depth);
        if (ttValue != noHashEntry) {
            double valTT = static_cast<double> ttValue / 1000.0;
            if (rootSide == black) valTT = -valTT;
            return valTT;
        }
    }

    if (node->childrenList.empty()) {
        // Evaluate with the NN
        std::pair<std::vector<float>, float> result = NNevaluate(node->nodeBoard);
        double whiteValue = static_cast<double>result.second;
        if (rootSide == black) {
            whiteValue = -whiteValue;
        }
        // Store NN result scaled by 1000
        int storeScore = static_cast<int>(result.second * 1000);
        writeHashEntry(node->nodeBoard, storeScore, node->depth, hashFlagExact);
        return whiteValue;
    }
}

void MCTSbackpropogate(MCTSNode* node, double value) {
    while (node != nullptr) {
        {
            std::lock_guard<std::mutex> lock(node->nodeMutex);

            // veverse the virtualLoss
            node->virtualLoss.fetch_sub(1, std::memory_order_relaxed);

            int oldVisits = node->visitCount.fetch_add(1, std::memory_order_relaxed);
            double oldSum = node->valueSum.load(std::memory_order_relaxed);
            node->valueSum.store(oldSum + value, std::memory_order_relaxed);
        }

        // Flip the perspective for the parent
        value = -value;
        node = node->parent;
    }
}

void searchThread(MCTSNode* root, int timeLimit, std::atomic<bool>& stopFlag) {
    auto startTime = std::chrono::high_resolution_clock::now();

    while (!stopFlag.load(std::memory_order_relaxed)) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();

        // stop once reached time
        if (elapsedMs >= timeLimit) {
            stopFlag.store(true, std::memory_order_relaxed);
            break;
        }

        MCTSNode* node = root;
        while (true) {
            if (!node->isFullyExpanded() || node->isTerminal()) {
                break;
            }
            MCTSNode* selectedChild = MCTSselect(node);
            if (!selectedChild) break;
            node = selectedChild;
        }
        if (!node->isFullyExpanded() && !node->isTerminal()) {
            MCTSexpand(node);
        }
        double value = MCTSsimulate(node);
        MCTSbackpropogate(node, value);
    }
}

MCTSNode* MCTSsearch(MCTSNode* root, int timeLimit) {
    // Max threads!!!!
    int threadCount = std::max(1, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    std::atomic<bool> stopFlag(false);

    for (int i = 0; i < static_cast<int>threadCount; i++) {
        pool.emplace_back(searchThread, root, timeLimit, std::ref(stopFlag));
    }
    for (auto& t : pool) {
        t.join();
    }

    std::cout << "\nMCTSsearch Selecting best child node of root node: \n\n";
    MCTSNode* bestChild = MCTSselectFinal(root);

    return bestChild;
}

int MCTSmain(const BoardState& board) {
    rootSide = board.side;

    std::unique_ptr<MCTSNode> root(new MCTSNode(nullptr, board));
    MCTSNode* bestChild = MCTSsearch(root.get(), 1500); // time in MS
    if (!bestChild) {
        std::cout << "[MCTSmain] No best child found. Returning -1.\n";
        return -1;
    }
    std::cout << "\nNodeCount: " << nodeCount.load() << "\n";
    std::cout << "maxdepth = " << maxDepth.load() << "\n";
    std::cout << "terminalmoves = " << terminalMoves.load() << "\n";
    std::cout << "bestmove ";
    printMove(bestChild->priorMove);
    std::cout << "\n";

    return bestChild->priorMove;
}
