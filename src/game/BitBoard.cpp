#include "BitBoard.h"


// ----------------------
// Implementation: Move
// ----------------------

void Moves::addMove(int move){
    moves.push_back(move);
}

static void Moves::printMove(int move){
    if(getMovePromoted(move)){
            std::cout<<squareToCoordinates[getMoveSource(move)]
             << squareToCoordinates[getMoveTarget(move)]
             << promotedPieces.at(getMovePromoted(move));
    }
    else{
            std::cout<<squareToCoordinates[getMoveSource(move)] << squareToCoordinates[getMoveTarget(move)];
    }
}

void Moves::printMoveList(){
    if (moveList -> moves.size() == 0){
        std::cout<< "no moves in move list";
        return;
    }
    std::cout<< "\n    move    piece   capture   double    enpass    castling\n\n";
    for (int i = 0 ; i < (moveList -> moves.size()) ; i ++){
        int move = moveList->moves[i];
        std::cout<< "    " <<squareToCoordinates[getMoveSource(move)]<<
                    squareToCoordinates[getMoveTarget(move)] <<
                    (getMovePromoted(move) ? promotedPieces.at(getMovePromoted(move)) : ' ') << "    " <<
                    asciiPieces[getMovePiece(move)] << "       " <<
                    (getMoveCapture(move) ? 1 : 0) << "       " <<
                    (getMoveDouble(move) ? 1 : 0) << "        " <<
                    (getMoveEnPassant(move) ? 1 : 0 )<< "          " <<
                    (getMoveCastling(move) ? 1 : 0) << "\n";

    }
    std::cout << "\n total number of moves:" << moveList -> moves.size();
}


static int Moves::moveToIndex(int move) {

    if (moveToIndexCP.find(move) != moveToIndexCP.end()){
        return moveToIndexCP.at(move);
    }
    if (moveToIndexCaptureCP.find(move) != moveToIndexCaptureCP.end()){
        return moveToIndexCaptureCP.at(move) + 4096;
    }

    int sourceIdx = getMoveSource(move);
    int targetIdx = getMoveTarget(move);
    
    int sourceRank = sourceIdx / 8;
    int sourceFile = sourceIdx % 8;
    int targetRank = targetIdx / 8;
    int targetFile = targetIdx % 8; 

    int newSourceIdx = (7 - sourceRank) * 8 + sourceFile;
    int newtargetIdx = (7 - targetRank) * 8 + targetFile;

    return 64 * newSourceIdx + newtargetIdx;
}



static int Moves::indexToMove(int index, Moves* moveList) {
    int sourceIdx = index / 64;
    int targetIdx = index % 64;

    // capture promotion
    if (index >= 4096){
        for (int moveCount = 0; moveCount < moveList->count; moveCount++) {
            int move = moveList->moves[moveCount];
            if (indexToMoveCaptureCP.at(index - 4096) == move) {
                return move;
            }
        }
    }
    // non capture promotion
    else if (sourceIdx == targetIdx){
        // this is fishy right here
        for (int moveCount = 0; moveCount < moveList->count; moveCount++) {
            int move = moveList->moves[moveCount];
            if (indexToMoveCaptureCP.at(index) == move) {
                return move;
            }
        }
    }
    // regular move
    else{
        // must convert from one encoding (a1 as 0, h8 as 63 to a8 as 0, h1 as 63)
        int newSourceIdx = (7 - sourceIdx / 8) * 8 + sourceIdx % 8;
        int newTargetIdx = (7 - targetIdx / 8) * 8 + targetIdx % 8;
        //std::cout<< "\n newsoureidx : "<< newSourceIdx <<" and target:" << newTargetIdx;
        for (int moveCount = 0; moveCount < moveList->count; moveCount++) {
            int move = moveList->moves[moveCount];
            if (newSourceIdx == getMoveSource(move) && newTargetIdx == getMoveTarget(move)) {
                return move;
            }
        }
    }
    return -1;
}

// ----------------------
// Implementation: Magics
// ----------------------

Magics::Magics() {
    // Initialize attack masks
    initLeaperAttacks();
    initSliderAttacks(bishop);  // Initialize bishop attacks
    initSliderAttacks(rook); // Initialize rook attacks
}

// ----------------------
// Private members: Magics
// ----------------------

const int Magics::bishopRelevantBit[64] = {
    6, 5, 5, 5, 5, 5, 5, 6, 
    5, 5, 5, 5, 5, 5, 5, 5, 
    5, 5, 7, 7, 7, 7, 5, 5, 
    5, 5, 7, 9, 9, 7, 5, 5, 
    5, 5, 7, 9, 9, 7, 5, 5, 
    5, 5, 7, 7, 7, 7, 5, 5, 
    5, 5, 5, 5, 5, 5, 5, 5, 
    6, 5, 5, 5, 5, 5, 5, 6
};

const int Magics::rookRelevantBit[64] = {
    12, 11, 11, 11, 11, 11, 11, 12, 
    11, 10, 10, 10, 10, 10, 10, 11, 
    11, 10, 10, 10, 10, 10, 10, 11, 
    11, 10, 10, 10, 10, 10, 10, 11, 
    11, 10, 10, 10, 10, 10, 10, 11, 
    11, 10, 10, 10, 10, 10, 10, 11, 
    11, 10, 10, 10, 10, 10, 10, 11, 
    12, 11, 11, 11, 11, 11, 11, 12
};

const uint64_t Magics::NOT_IN_A_FILE  = 18374403900871474942ULL;
const uint64_t Magics::NOT_IN_H_FILE  = 9187201950435737471ULL;
const uint64_t Magics::NOT_IN_AB_FILE = 18229723555195321596ULL;
const uint64_t Magics::NOT_IN_HG_FILE = 4557430888798830399ULL;

const uint64_t bishopMagicNumbers[64] = {
    0x40040844404084ULL, 0x2004208a004208ULL, 0x10190041080202ULL, 0x108060845042010ULL,
    0x581104180800210ULL, 0x2112080446200010ULL, 0x1080820820060210ULL, 0x3c0808410220200ULL,
    0x4050404440404ULL, 0x21001420088ULL, 0x24d0080801082102ULL, 0x1020a0a020400ULL,
    0x40308200402ULL, 0x4011002100800ULL, 0x401484104104005ULL, 0x801010402020200ULL,
    0x400210c3880100ULL, 0x404022024108200ULL, 0x810018200204102ULL, 0x4002801a02003ULL,
    0x85040820080400ULL, 0x810102c808880400ULL, 0xe900410884800ULL, 0x8002020480840102ULL,
    0x220200865090201ULL, 0x2010100a02021202ULL, 0x152048408022401ULL, 0x20080002081110ULL,
    0x4001001021004000ULL, 0x800040400a011002ULL, 0xe4004081011002ULL, 0x1c004001012080ULL,
    0x8004200962a00220ULL, 0x8422100208500202ULL, 0x2000402200300c08ULL, 0x8646020080080080ULL,
    0x80020a0200100808ULL, 0x2010004880111000ULL, 0x623000a080011400ULL, 0x42008c0340209202ULL,
    0x209188240001000ULL, 0x400408a884001800ULL, 0x110400a6080400ULL, 0x1840060a44020800ULL,
    0x90080104000041ULL, 0x201011000808101ULL, 0x1a2208080504f080ULL, 0x8012020600211212ULL,
    0x500861011240000ULL, 0x180806108200800ULL, 0x4000020e01040044ULL, 0x300000261044000aULL,
    0x802241102020002ULL, 0x20906061210001ULL, 0x5a84841004010310ULL, 0x4010801011c04ULL,
    0xa010109502200ULL, 0x4a02012000ULL, 0x500201010098b028ULL, 0x8040002811040900ULL,
    0x28000010020204ULL, 0x6000020202d0240ULL, 0x8918844842082200ULL, 0x4010011029020020ULL
};

const uint64_t rookMagicNumbers[64] = {
    0x8a80104000800020ULL, 0x140002000100040ULL, 0x2801880a0017001ULL, 0x100081001000420ULL,
    0x200020010080420ULL, 0x3001c0002010008ULL, 0x8480008002000100ULL, 0x2080088004402900ULL,
    0x800098204000ULL, 0x2024401000200040ULL, 0x100802000801000ULL, 0x120800800801000ULL,
    0x208808088000400ULL, 0x2802200800400ULL, 0x2200800100020080ULL, 0x801000060821100ULL,
    0x80044006422000ULL, 0x100808020004000ULL, 0x12108a0010204200ULL, 0x140848010000802ULL,
    0x481828014002800ULL, 0x8094004002004100ULL, 0x4010040010010802ULL, 0x20008806104ULL,
    0x100400080208000ULL, 0x2040002120081000ULL, 0x21200680100081ULL, 0x20100080080080ULL,
    0x2000a00200410ULL, 0x20080800400ULL, 0x80088400100102ULL, 0x80004600042881ULL,
    0x4040008040800020ULL, 0x440003000200801ULL, 0x4200011004500ULL, 0x188020010100100ULL,
    0x14800401802800ULL, 0x2080040080800200ULL, 0x124080204001001ULL, 0x200046502000484ULL,
    0x480400080088020ULL, 0x1000422010034000ULL, 0x30200100110040ULL, 0x100021010009ULL,
    0x2002080100110004ULL, 0x202008004008002ULL, 0x20020004010100ULL, 0x2048440040820001ULL,
    0x101002200408200ULL, 0x40802000401080ULL, 0x4008142004410100ULL, 0x2060820c0120200ULL,
    0x1001004080100ULL, 0x20c020080040080ULL, 0x2935610830022400ULL, 0x44440041009200ULL,
    0x280001040802101ULL, 0x2100190040002085ULL, 0x80c0084100102001ULL, 0x4024081001000421ULL,
    0x20030a0244872ULL, 0x12001008414402ULL, 0x2006104900a0804ULL, 0x1004081002402ULL
};

// ----------------------
// Public functions: Magics
// ----------------------

uint64_t Magics::findMagicNumber (int square, int relaventBits, int bishops){
    uint64_t occupancies[4096];
    uint64_t attacks[4096];
    uint64_t usedAttacks[4096];
    uint64_t attackMask = bishop ? maskBishopAttacks(square) : maskRookAttacks(square);
    //init occupancy indicies
    int occupancyIndices = 1 << relaventBits;
    for (int i = 0 ; i < occupancyIndices ; i ++){
        occupancies[i] = setOccupancy(i, relaventBits, attackMask);
        attacks[i] = bishop ? bishopAttacksOnTheFly(square, occupancies[i]) : rookAttacksOnTheFly(square, occupancies[i]); 
    }
    //test magic number loop
    for (int i = 0 ; i < 10000000000 ; i++ ){
        //geenrate magic number candidate
        uint64_t magicNumber = generateMagicNumber();
        //skip bad magic number
        if (countBits((attackMask * magicNumber) & 0xFF00000000000000) < 6){
            continue;
        }
        memset(usedAttacks, 0ULL, sizeof(usedAttacks));
        // init index and fail flag
        int index, fail;
        //test magic index loop
        for (index = 0 , fail = 0; ! fail && index < occupancyIndices; index++){
            int magicIndex = static_cast<int>((occupancies[index] * magicNumber) >> (64 - relaventBits));

            //if magic index works
            if (usedAttacks[magicIndex] == 0ULL){
                // init used attacks
                usedAttacks[magicIndex] = attacks[index];
            }
            else if (usedAttacks[magicIndex] != attacks[index]){
                fail = 1;
            }
        }
        if (!fail){
            return magicNumber;
        }
    }
    std::cout << std::endl << "magic number fails" << std::endl;
    return 0ULL;

}

void Magics::initMagicNumbers(){ 
    for (int square = 0 ; square < 64 ; square ++){
        std::cout << findMagicNumber(square, bishopRelevantBit[square] , bishop) <<"\n";
        bishopMagicNumbers[square] = findMagicNumber(square, bishopRelevantBit[square] , bishop);
    }
    std::cout << "\n\n\n";
    for (int square = 0 ; square < 64 ; square ++){
        std::cout << findMagicNumber(square, rookRelevantBit[square] , rook) <<"\n";
        rookMagicNumbers[square] = findMagicNumber(square, rookRelevantBit[square] , rook);
    }
}

uint64_t Magics::maskPawnAttacks(int side , int square){
    uint64_t attacks = 0ULL;

    uint64_t bitBoard = 0ULL;
    setBit(bitBoard, square);

    // white pawns
    if (!side){
        //handles capture to the right
        if ((bitBoard >> 7) & notInAFile){
            attacks |= (bitBoard >> 7); 
        }
        //handles capture to the left
        if ((bitBoard >> 9) & notInHFile){
            attacks |= (bitBoard >> 9); 
        }
    }
    // black pawns
    else{
        //handles capture to the right
        if ((bitBoard << 7) & notInHFile){
            attacks |= (bitBoard << 7); 
        }
        //handles capture to the left
        if ((bitBoard << 9) & notInAFile){
            attacks |= (bitBoard << 9); 
        }
    }
    return attacks;
}


uint64_t Magics::maskKnightAttacks(int square){
    uint64_t attacks = 0ULL;

    uint64_t bitBoard = 0ULL;
    setBit(bitBoard, square);
    attacks |= (bitBoard >> 17) & notInHFile;
    attacks |= (bitBoard >> 15) & notInAFile;
    attacks |= (bitBoard >> 10) & notInHGFile;
    attacks |= (bitBoard >> 6)  & notInABFile;
    attacks |= (bitBoard << 17) & notInAFile;
    attacks |= (bitBoard << 15) & notInHFile;
    attacks |= (bitBoard << 10) & notInABFile;
    attacks |= (bitBoard << 6)  & notInHGFile;
    return attacks;
}

uint64_t Magics::maskKingAttacks(int square){
    uint64_t attacks = 0ULL;

    // set bitboard square to mask all attacks`
    uint64_t bitBoard = 0ULL;
    setBit(bitBoard, square);

    attacks |= (bitBoard >> 8);              
    attacks |= (bitBoard >> 9) & notInHFile;
    attacks |= (bitBoard >> 7) & notInAFile; 
    attacks |= (bitBoard >> 1) & notInHFile;
    attacks |= (bitBoard << 8);              
    attacks |= (bitBoard << 9) & notInAFile;
    attacks |= (bitBoard << 7) & notInHFile;
    attacks |= (bitBoard << 1) & notInAFile;    
    return attacks;
}

uint64_t Magics::maskBishopAttacks(int square){
    uint64_t attacks = 0ULL;
    // current file and rank
    int r, f;

    // target file and rank
    int tr = square / 8;
    int tf = square % 8;

    // do not mask the squares at the edge of the board because bishop movement is limited on edges
    for (r = tr + 1, f = tf + 1; r <= 6 && f <= 6; r++, f++) attacks |= (1ULL << (r * 8 + f));
    for (r = tr - 1, f = tf + 1; r >= 1 && f <= 6; r--, f++) attacks |= (1ULL << (r * 8 + f));
    for (r = tr + 1, f = tf - 1; r <= 6 && f >= 1; r++, f--) attacks |= (1ULL << (r * 8 + f));
    for (r = tr - 1, f = tf - 1; r >= 1 && f >= 1; r--, f--) attacks |= (1ULL << (r * 8 + f));
  
    return attacks;
}

uint64_t Magics::maskRookAttacks(int square){
    uint64_t attacks = 0ULL;
    // current file and rank
    int r, f;
    
    // target file and rank
    int tr = square / 8;
    int tf = square % 8;

    for (r = tr + 1; r <= 6; r++) attacks |= (1ULL << (r * 8 + tf));
    for (r = tr - 1; r >= 1; r--) attacks |= (1ULL << (r * 8 + tf));
    for (f = tf + 1; f <= 6; f++) attacks |= (1ULL << (tr * 8 + f));
    for (f = tf - 1; f >= 1; f--) attacks |= (1ULL << (tr * 8 + f));
    
    return attacks;
}

uint64_t Magics::bishopAttacksOnTheFly(int square, uint64_t block){
    uint64_t attacks = 0ULL;

    // current file and rank
    int r, f;

    // target file and rank
    int tr = square / 8;
    int tf = square % 8;

    // don't distinguish white and black, move generator will do that
    for (r = tr + 1, f = tf + 1; r <= 7 && f <= 7; r++, f++){
        attacks |= (1ULL << (r * 8 + f));
        if ((1ULL << (r * 8 + f)) & block){
            break;
        }
    }
    for (r = tr - 1, f = tf + 1; r >= 0 && f <= 7; r--, f++) {
        attacks |= (1ULL << (r * 8 + f));
        if ((1ULL << (r * 8 + f)) & block){
            break;
        }
    }
    for (r = tr + 1, f = tf - 1; r <= 7 && f >= 0; r++, f--) {
        attacks |= (1ULL << (r * 8 + f));
        if ((1ULL << (r * 8 + f)) & block){
            break;
        }
    }
    for (r = tr - 1, f = tf - 1; r >= 0 && f >= 0; r--, f--) {
        attacks |= (1ULL << (r * 8 + f));
        if ((1ULL << (r * 8 + f)) & block){
            break;
        }
    }
    return attacks;
}

uint64_t Magics::rookAttacksOnTheFly(int square, uint64_t block){
    uint64_t attacks = 0ULL;
    // current file and rank
    int r, f;
    
    // target file and rank
    int tr = square / 8;
    int tf = square % 8;

    for (r = tr + 1; r <= 7; r++) {
        attacks |= (1ULL << (r * 8 + tf));
        if ((1ULL << (r * 8 + tf)) & block){
            break;
        }
    }
    for (r = tr - 1; r >= 0; r--) {
        attacks |= (1ULL << (r * 8 + tf));
        if ((1ULL << (r * 8 + tf)) & block){
            break;
        }
    }
    for (f = tf + 1; f <= 7; f++) {
        attacks |= (1ULL << (tr * 8 + f));
        if ((1ULL << (tr * 8 + f)) & block){
            break;
        }
    }
    for (f = tf - 1; f >= 0; f--) {
        attacks |= (1ULL << (tr * 8 + f));
        if ((1ULL << (tr * 8 + f)) & block){
            break;
        }
    }
    
    return attacks;
}


void Magics::initLeaperAttacks(){
    for (int square = 0 ; square < 64 ; square++){
        //initialize pawn attacks array only
        pawnAttacks[white][square] = maskPawnAttacks(white, square);
        pawnAttacks[black][square] = maskPawnAttacks(black, square);

        //initialize knight attacks array only
        knightAttacks[square] = maskKnightAttacks(square);

        //initialize king attack array only
        kingAttacks[square] = maskKingAttacks(square);
    }
}

uint64_t Magics::setOccupancy (int index, int bitsInMask, uint64_t attackMask){
    uint64_t occupancy = 0ULL;

    for (int i = 0 ; i < bitsInMask ; i++){
        int square = getLeastSigBitIndex(attackMask);
        popBit(attackMask, square);

        if (index & ( 1 << i)){
            occupancy |= (1ULL << square);
        }
    }
    return occupancy; 
} 

void Magics::initSliderAttacks(int bishop){
    for (int square = 0 ; square < 64 ; square++){
        bishopMasks[square] = maskBishopAttacks(square);
        rookMasks[square] = maskRookAttacks(square);

        u_int64_t attackMask = bishop? bishopMasks[square] : rookMasks[square];

        //init relavent occupancy bit coin
        int relevantBitsCount = countBits(attackMask);
        int occupancyIndicies = 1 << relevantBitsCount;

        for (int index = 0 ; index < occupancyIndicies ; index++){
            if (bishop){
                u_int64_t occupancy = setOccupancy(index, relevantBitsCount, attackMask);
                
                //init magic index
                int magicIndex = (occupancy * bishopMagicNumbers[square]) >> (64 - bishopRelevantBit[square]);

                bishopAttacks[square][magicIndex] = bishopAttacksOnTheFly(square, occupancy);
            }else{
                u_int64_t occupancy = setOccupancy(index, relevantBitsCount, attackMask);
                
                //init magic index
                int magicIndex = (occupancy * rookMagicNumbers[square]) >> (64 - rookRelevantBit[square]);

                rookAttacks[square][magicIndex] = rookAttacksOnTheFly(square, occupancy);
            }
        }

    }
}

uint64_t Magics::getBishopAttacks(int square, uint64_t occupancy){
    occupancy  &= bishopMasks[square];
    occupancy *= bishopMagicNumbers[square];
    occupancy >>= 64 - bishopRelevantBit[square];

    return bishopAttacks[square][occupancy];
}

uint64_t Magics::getRookAttacks(int square, uint64_t occupancy){
    occupancy  &= rookMasks[square];
    occupancy *= rookMagicNumbers[square];
    occupancy >>= 64 - rookRelevantBit[square];

    return rookAttacks[square][occupancy];
}

uint64_t Magics::getQueenAttacks(int square, uint64_t occupancy){
    uint64_t queenAttacks = 0ULL;
    u_int64_t rookOccunpancies = occupancy;
    u_int64_t bishopOccunpancies = occupancy;

    // combination of bishop and rook attacks
    bishopOccunpancies  &= bishopMasks[square];
    bishopOccunpancies *= bishopMagicNumbers[square];
    bishopOccunpancies >>= 64 - bishopRelevantBit[square];
    queenAttacks = bishopAttacks[square][bishopOccunpancies];
    rookOccunpancies  &= rookMasks[square];
    rookOccunpancies *= rookMagicNumbers[square];
    rookOccunpancies >>= 64 - rookRelevantBit[square];
    queenAttacks |= rookAttacks[square][rookOccunpancies];

    return queenAttacks;
}

static int Magics::isSquareAttacked(const BitBoardState & board,int square, int side){
    // check if attacked by white pawns
    if ((side == white) && (pawnAttacks[black][square] & board.bitBoard[P]) != 0) return 1;

    // check if attacked by black pawns
    if ((side == black) && (pawnAttacks[white][square] & board.bitBoard[p]) != 0) return 1;

    //check if attacked by knight
    if ((knightAttacks[square] & ((side == white) ? board.bitBoard[N] : board.bitBoard[n])) != 0) return 1;
    
    // check if attakced by bishop
    if (getBishopAttacks(square, board.occupancies[both]) & ((side == white) ? board.bitBoard[B] : board.bitBoard[b])) return 1;

    // check if attakced by rook
    if (getRookAttacks(square, board.occupancies[both]) & ((side == white) ? board.bitBoard[R] : board.bitBoard[r])) return 1;


    // check if attakced by queen
    if (getQueenAttacks(square, board.occupancies[both]) & ((side == white) ? board.bitBoard[Q] : board.bitBoard[q])) return 1;

    //check if attacked by king 
    if ((kingAttacks[square] & ((side == white) ? board.bitBoard[K] : board.bitBoard[k])) != 0) return 1;
    return 0;
}


// ----------------------
// Implementation: Magics
// ----------------------



