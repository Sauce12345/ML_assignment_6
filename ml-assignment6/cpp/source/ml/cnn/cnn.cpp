/**
 * @brief Convolutional neural network (CNN) implementation details.
 */
#include <algorithm>
#include <iostream>

#include "ml/cnn/cnn.h"
#include "ml/factory/interface.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::cnn
{
// -----------------------------------------------------------------------------
Cnn::Cnn(factory::Interface& factory, const std::size_t convInput, const std::size_t convKernel, 
         const act_func::Type convFunc, const std::size_t poolSize, 
         const std::size_t denseOutput, const act_func::Type denseFunc)
    : myConvLayers{}
    , myDenseLayers{}
    , myFlattenLayer{nullptr}
    , myFactory{factory}
{
    // Initialize the convolutional layers.
    myConvLayers.emplace_back(factory.convLayer(convInput, convKernel, convFunc));
    myConvLayers.emplace_back(factory.maxPoolLayer(convOutputSize(), poolSize));

    // Initialize the flatten layer.
    myFlattenLayer = factory.flattenLayer(convOutputSize());

    // Initialize the dense layer.
    const std::size_t denseInput{myFlattenLayer->outputSize()};
    myDenseLayers.emplace_back(factory.denseLayer(denseInput, denseOutput, denseFunc));
}

// -----------------------------------------------------------------------------
Cnn::~Cnn() noexcept = default;

// -----------------------------------------------------------------------------
std::size_t Cnn::inputSize() const noexcept 
{ 
    return myConvLayers[0U]->inputSize(); 
}

// -----------------------------------------------------------------------------
std::size_t Cnn::outputSize() const noexcept 
{
    const std::size_t last{myDenseLayers.size() - 1U};
    return myDenseLayers[last]->outputSize();
}

// -----------------------------------------------------------------------------
const Matrix1d& Cnn::predict(const Matrix2d& input) noexcept 
{
    feedforward(input);
    return output();
}

// -----------------------------------------------------------------------------
void Cnn::addDenseLayer(const std::size_t outputSize, const act_func::Type actFunc)
{
    myDenseLayers.emplace_back(myFactory.denseLayer(this->outputSize(), outputSize, actFunc));
}

// -----------------------------------------------------------------------------
bool Cnn::train(const Matrix3d& trainIn, const Matrix2d& trainOut, const std::size_t epochCount,
                const double learningRate)
{
    // Check the input arguments, return false on failure.
    if ((0.0 >= learningRate) || (1.0 < learningRate))
    {
        std::cerr << "Failed to train CNN: invalid learning rate " << learningRate << "!\n";
        return false;   
    }
    else if (0U == epochCount)
    {
        std::cerr << "Failed to train CNN: invalid epoch count " << epochCount << "!\n";
        return false;  
    }

    const std::size_t setCount{std::min(trainIn.size(), trainOut.size())};

    if (0U == setCount)
    {
        std::cerr << "Failed to train CNN: invalid set count " << setCount << "!\n";
        return false;
    }

    // Create a training order list.
    TrainOrderList trainOrder{createTrainOrderList(setCount)};

    // Train the network the specified number of epochs.
    for (std::size_t i{}; i < epochCount; ++i)
    {
        // Shuffle the training order list at the start of each epoch.
        shuffleTrainOrderList(trainOrder);

        // Iterate through the training sets, return false on failure.
        for (auto& j : trainOrder)
        {
            const Matrix2d& input{trainIn[j]};
            const Matrix1d& output{trainOut[j]};

            const bool success{feedforward(input) && backpropagate(output) && optimize(learningRate)};
            if (!success) { return false; }
        }
    }
    // Return true on success.
    return true;
}

// -----------------------------------------------------------------------------
const Matrix1d& Cnn::output() const noexcept
{
    const std::size_t last{myDenseLayers.size() - 1U};
    return myDenseLayers[last]->output();
}

// -----------------------------------------------------------------------------
const Matrix2d& Cnn::convOutput() const noexcept
{
    const std::size_t last{myConvLayers.size() - 1U};
    return myConvLayers[last]->output();
}

// -----------------------------------------------------------------------------
std::size_t Cnn::convOutputSize() const noexcept
{
    const std::size_t last{myConvLayers.size() - 1U};
    return myConvLayers[last]->outputSize();
}

// -----------------------------------------------------------------------------
bool Cnn::feedforward(const Matrix2d& input) noexcept
{
    bool success{true};

    // Run feedforward operation in the convolutional layers, return false on failure.
    {
        success = myConvLayers[0U]->feedforward(input);
    
        for (std::size_t i{1U}; i < myConvLayers.size(); ++i)
        {
            auto& prevLayer{*(myConvLayers[i - 1U])};
            success &= myConvLayers[i]->feedforward(prevLayer.output());
        }
        if (!success) { return false; }
    }

    // Flatten the output from the convolutional layers, return false on failure.
    {
        success = myFlattenLayer->feedforward(convOutput());
        if (!success) { return false; }
    }

    // Run feedforward operation in the dense layers, return false on failure.
    {
        success = myDenseLayers[0U]->feedforward(myFlattenLayer->output());

        for (std::size_t i{1U}; i < myDenseLayers.size(); ++i)
        {
            auto& prevLayer{*(myDenseLayers[i - 1U])};
            success &= myDenseLayers[i]->feedforward(prevLayer.output());
        }
    }
    // Return true on success.
    return success;
}

// -----------------------------------------------------------------------------
bool Cnn::backpropagate(const Matrix1d& output) noexcept
{
    bool success{true};

    // Backpropagate through the dense layers, return false on failure.
    {
        const std::size_t last{myDenseLayers.size() - 1U};
        success = myDenseLayers[last]->backpropagate(output);
        
        for (std::size_t i{last}; i > 0U; --i)
        {
            success &= myDenseLayers[i - 1U]->backpropagate(myDenseLayers[i]->inputGradients());
        }
        if (!success) { return false; }
    }

    // Unflatten the input gradients from the first dense layer, return false on failure.
    {
        success = myFlattenLayer->backpropagate(myDenseLayers[0U]->inputGradients());
        if (!success) { return false; }
    }

    // Backpropagate through the convolutional layers, return false on failure.
    {
        const std::size_t last{myConvLayers.size() - 1U};
        success = myConvLayers[last]->backpropagate(myFlattenLayer->inputGradients());

        for (std::size_t i{last}; i > 0U; --i)
        {
            success &= myConvLayers[i - 1U]->backpropagate(myConvLayers[i]->inputGradients());
        }
    }
    // Return true on success.
    return success;
}

// -----------------------------------------------------------------------------
bool Cnn::optimize(const double learningRate) noexcept
{
    bool success{true};

    // Optimize the convolutional layers, return false on failure.
    {
        success = myConvLayers[0U]->optimize(learningRate);
    
        for (std::size_t i{1U}; i < myConvLayers.size(); ++i)
        {
            success &= myConvLayers[i]->optimize(learningRate);
        }
        if (!success) { return false; }
    }

    // Optimize the dense layers, return false on failure.
    {
        success = myDenseLayers[0U]->optimize(myFlattenLayer->output(), learningRate);

        for (std::size_t i{1U}; i < myDenseLayers.size(); ++i)
        {
            auto& prevLayer{*(myDenseLayers[i - 1U])};
            success &= myDenseLayers[i]->optimize(prevLayer.output(), learningRate);
        }
    }
    // Return true on success.
    return success;
}
} // namespace ml::cnn