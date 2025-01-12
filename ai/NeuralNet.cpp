// neuralnet.cpp

#include "NeuralNet.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>
#include <numeric>

NeuralNet::NeuralNet(){
    if (!initializeEngine("/workspace/src/model.engine")) {
        std::cerr << "Failed to initialize the engine." << std::endl;
    }
}

NeuralNet::~NeuralNet(){
    destroyEngine();
}

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

static Logger gLogger;

// Global variables for the engine
static nvinfer1::IRuntime* runtime = nullptr;
static nvinfer1::ICudaEngine* engine = nullptr;
static nvinfer1::IExecutionContext* context = nullptr;

static int inputIndex = -1;
static int policyOutputIndex = -1;
static int valueOutputIndex = -1;

static int batchSize = 1;
static size_t inputSize = 0;
static size_t policyOutputSize = 0;
static size_t valueOutputSize = 0;

static void** buffers = nullptr;  // Pointers to device buffers
static int nbBindings = 0;

void* safeCudaMalloc(size_t memSize) {
    void* deviceMem;
    cudaError_t status = cudaMalloc(&deviceMem, memSize);
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed!");
    }
    return deviceMem;
}

bool NeuralNet::initializeEngine(const std::string& engineFilePath) {
    std::ifstream file(engineFilePath, std::ios::binary);
    if (!file) {
        std::cerr << "Cant open file: " << engineFilePath << std::endl;
        return false;
    }

    file.seekg(0, file.end);
    size_t fsize = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(fsize);
    file.read(engineData.data(), fsize);
    file.close();

    // Create the runtime and engine
    runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Cant create TensorRT runtime." << std::endl;
        return false;
    }

    engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
    if (!engine) {
        std::cerr << "Can't deserialize CUDA engine." << std::endl;
        return false;
    }

    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Cant create execution context." << std::endl;
        return false;
    }

    // Get binding indices
    inputIndex = engine->getBindingIndex("input");
    policyOutputIndex = engine->getBindingIndex("policy_output");
    valueOutputIndex = engine->getBindingIndex("value_output");

    nbBindings = engine->getNbBindings();
    buffers = new void*[nbBindings];
    for (int i = 0; i < nbBindings; ++i) {
        buffers[i] = nullptr;
    }

    // Allocate buffers
    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        size_t size = batchSize;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
        }
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        size_t eltSize;
        if (dtype == nvinfer1::DataType::kFLOAT) {
            eltSize = sizeof(float);
        } else if (dtype == nvinfer1::DataType::kHALF) {
            eltSize = sizeof(__half);
        }
        buffers[i] = safeCudaMalloc(size * eltSize);
        if (i == inputIndex) {
            inputSize = size;
        } else if (i == policyOutputIndex) {
            policyOutputSize = size;
        } else if (i == valueOutputIndex) {
            valueOutputSize = size;
        }
    }
    return true;
}

void NeuralNet::destroyEngine() {
    // Release the buffers
    if (buffers) {
        for (int i = 0; i < nbBindings; ++i) {
            if (buffers[i]) {
                cudaFree(buffers[i]);
                buffers[i] = nullptr;
            }
        }
        delete[] buffers;
        buffers = nullptr;
    }

    // Destroy the execution context, engine, and runtime
    if (context) {
        context->destroy();
        context = nullptr;
    }
    if (engine) {
        engine->destroy();
        engine = nullptr;
    }
    if (runtime) {
        runtime->destroy();
        runtime = nullptr;
    }
}

bool NeuralNet::prepareInputTensor(const BitBoardState& boardState, std::vector<float>& inputTensor) {
    // Initialize the input tensor
    inputTensor.resize(17 * 8 * 8, 0.0f);  // 17 planes of 8x8
    int pieceToPlane[12] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11 
    };

    // Populate the planes with pieces
    for (int piece = P; piece <= k; piece++) {
        uint64_t bitboard = boardState.bitBoard[piece];
        int plane = pieceToPlane[piece];
        while (bitboard) {
            int square = __builtin_ctzll(bitboard);
            bitboard &= bitboard - 1;
            int row = (square / 8);
            int col = (square % 8);  

            int index = plane * 64 + row * 8 + col;
            inputTensor[index] = 1.0f;
        }
    }

    float sideToMove = (boardState.side == 0) ? 0.0f : 1.0f;
    for (int i = 0; i < 64; ++i) {
        inputTensor[12 * 64 + i] = sideToMove;
    }

    // Castling rights
    for (int i = 0; i < 64; ++i) {
        inputTensor[(12 + 1) * 64 + i] = (boardState.castle & 0b0001) ? 1.0f : 0.0f;
        inputTensor[(12 + 2) * 64 + i] = (boardState.castle & 0b0010) ? 1.0f : 0.0f;
        inputTensor[(12 + 3) * 64 + i] = (boardState.castle & 0b0100) ? 1.0f : 0.0f;
        inputTensor[(12 + 4) * 64 + i] = (boardState.castle & 0b1000) ? 1.0f : 0.0f;
    }
    return true;
}


bool NeuralNet::runInference(const BitBoardState& boardState, std::vector<float>& policyOutput, float& valueOutput) {
    if (!engine || !context) {
        std::cerr << "Engine not initialized." << std::endl;
        return false;
    }

    // Prepare the input tensor
    std::vector<float> inputTensor;
    if (!prepareInputTensor(boardState, inputTensor)) {
        std::cerr << "Failed to prepare input tensor." << std::endl;
        return false;
    }

    // Copy input data to device buffer
    cudaMemcpy(buffers[inputIndex], inputTensor.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Execute inference
    bool status = context->executeV2(buffers);
    if (!status) {
        std::cerr << "Failed inferencing" << std::endl;
        return false;
    }

    // Copy outputs back to host
    policyOutput.resize(policyOutputSize);
    cudaMemcpy(policyOutput.data(), buffers[policyOutputIndex], policyOutputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(&valueOutput, buffers[valueOutputIndex], valueOutputSize * sizeof(float), cudaMemcpyDeviceToHost);

    return true;
}

std::pair<std::vector<float>,float> NNevaluate(BitBoardState board){
    std::vector<float> policyOutput;
    float valueOutput;

    if (runInference(board, policyOutput, valueOutput)) {
        std::cout << "Inference succeeded" << std::endl;
        std::cout << "Value output: " << valueOutput << std::endl;
        
    } else {
        std::cerr << "Inference failed" << std::endl;
    }


    return std::make_pair(policyOutput, valueOutput);
}