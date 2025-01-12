// neuralnet.hpp

#ifndef NEURALNET_HPP
#define NEURALNET_HPP

#include <string>
#include <vector>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include "game/BitBoard.hpp"

class NeuralNet{
public:
    NeuralNet();
    ~NeuralNet();
    std::pair<std::vector<float>,float> NNevaluate(BitBoardState board);
private:
    bool initializeEngine(const std::string& engineFilePath);
    void destroyEngine();
    bool prepareInputTensor(const BitBoardState& boardState, std::vector<float>& inputTensor);
    bool runInference(const BitBoardState& boardState, std::vector<float>& policyOutput, float& valueOutput);
};
// Function declarations

#endif // NEURALNET_HPP
