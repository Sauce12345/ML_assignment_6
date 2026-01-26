/**
 * @brief Dense layer implementation details.
 */
#include <stdexcept>

#include "ml/act_func/type.h"
#include "ml/dense_layer/dense.h"
#include "ml/factory/factory.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::dense_layer
{
// -----------------------------------------------------------------------------
Dense::Dense(const std::size_t inputSize, const std::size_t outputSize,
             const act_func::Type actFunc)
    : myInputGradients{}
    , myBias{}
    , myWeights{}
    , myOutput{}
    , myError{}
    , myActFunc{nullptr}
{
    checkParameters(inputSize, outputSize);
    initialize(inputSize, outputSize, actFunc);
}

// -----------------------------------------------------------------------------
std::size_t Dense::inputSize() const noexcept 
{ 
    return myWeights.empty() ? 0U : myWeights[0U].size(); 
}

// -----------------------------------------------------------------------------
std::size_t Dense::outputSize() const noexcept { return myOutput.size(); }

// -----------------------------------------------------------------------------
const std::vector<double>& Dense::output() const noexcept { return myOutput; }

// -----------------------------------------------------------------------------
const std::vector<double>& Dense::inputGradients() const noexcept { return myInputGradients; }

// -----------------------------------------------------------------------------
bool Dense::feedforward(const std::vector<double>& input) noexcept 
{
    // Return false if the dimensions don't match.
    constexpr const char* opName{"feedforward"};
    if (!matchDimensions(inputSize(), input.size(), opName)) { return false; }

    // Perform feedforward for all nodes in the dense layer, use `i` as node ID.
    for (std::size_t i{}; i < outputSize(); ++i)
    {
        // Calculate the sum of the weights and the input values.
        double sum{myBias[i]};

        for (std::size_t j{}; j < inputSize(); ++j) 
        { 
            sum += myWeights[i][j] * input[j];
        }

        // Calculate the output by applying the activation function to the sum.
        myOutput[i] = myActFunc->output(sum);
    }
    // Return true to indicate success.
    return true;
}

// -----------------------------------------------------------------------------
bool Dense::backpropagate(const std::vector<double>& outputGradients) noexcept 
{
    // Return false if the dimensions don't match.
    constexpr const char* opName{"backpropagation for output dense layer"};
    if (!matchDimensions(outputSize(), outputGradients.size(), opName)) { return false; }

    // Perform backpropagation for all nodes in the dense layer, use `i` as node ID.
    for (std::size_t i{}; i < outputSize(); ++i)
    {
        // Calculate the raw error.
        const double error{outputGradients[i] - myOutput[i]};

        // Calculate error by applying the activation function derivative to the output.
        myError[i] = error * myActFunc->delta(myOutput[i]);
    }

    // Compute input gradients.
    initMatrix(myInputGradients);

    for (std::size_t i{}; i < inputSize(); ++i)
    {
        for (std::size_t j{}; j < outputSize(); ++j)
        {
            myInputGradients[i] += myError[j] * myWeights[j][i];
        }
    }

    // Return true to indicate success.
    return true;
}

// -----------------------------------------------------------------------------
bool Dense::optimize(const std::vector<double>& input, const double learningRate) noexcept 
{
    // Return false if the dimensions don't match or the learning rate is invalid.
    constexpr const char* opName{"optimization in dense layer"};
    if (!matchDimensions(inputSize(), input.size(), opName) 
        || (!checkLearningRate(learningRate, opName))) { return false; }

    // Perform optimization for all nodes in the dense layer, use `i` as node ID.
    for (std::size_t i{}; i < outputSize(); ++i)
    {
        // Adjust the bias with the calculated error and the learningRate.
        myBias[i] += myError[i] * learningRate;

        // Use `j` as weight ID.
        for (std::size_t j{}; j < inputSize(); ++j)
        {      
            // Adjust the weights with the calculated error, the learningRate, and the input.
            myWeights[i][j] += myError[i] * learningRate * input[j];
        }
    }
    // Return true to indicate success.
    return true;
}

// -----------------------------------------------------------------------------
void Dense::checkParameters(const std::size_t inputSize, const std::size_t outputSize)
{
    // Throw exception if node count or the weight count is 0.
    if (0U == outputSize)
    {
        throw std::invalid_argument("Node count cannot be 0!");
    }
    else if (0U == inputSize)
    {
        throw std::invalid_argument("Weight count cannot be 0!");
    }
}

// -----------------------------------------------------------------------------
void Dense::initialize(const std::size_t inputSize, const std::size_t outputSize,
                      const act_func::Type actFunc)
{
    // Initialize the matrices.
    initMatrix(myInputGradients, inputSize);
    initMatrix(myBias, outputSize);
    initMatrix(myWeights, outputSize, inputSize);
    initMatrix(myOutput, outputSize);
    initMatrix(myError, outputSize);

    // Fill the bias and weight matrices with random values.
    for (std::size_t i{}; i < outputSize; ++i)
    {
        myBias[i] = randomStartVal();

        for (std::size_t j{}; j < inputSize; ++j)
        {
            myWeights[i][j] = randomStartVal();
        }
    }
    // Initialize the activation function.
    factory::Factory factory{};
    myActFunc = factory.actFunc(actFunc);
}
} // namespace ml::dense_layer