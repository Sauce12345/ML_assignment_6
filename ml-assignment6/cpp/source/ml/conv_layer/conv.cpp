//! @note Bra överlag, men algoritmerna för feedforward, backpropagation samt optimization saknas.
//!       Fick du uppladdat de rätta/den senaste versionen av din kod?
#include <sstream>
#include <stdexcept>

#include "ml/act_func/type.h"
#include "ml/conv_layer/conv.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::conv_layer
{
Conv::Conv(const std::size_t inputSize, const std::size_t kernelSize, 
                const act_func::Type actFunc)
    : myInputPadded{}
    , myInputGradientsPadded{}
    , myInputGradients{}
    , myKernel{}
    , myKernelGradients{}
    , myOutput{}
    , myBias{randomStartVal()}
    , myBiasGradient{}
{
    // Throw exception if the kernel size is outside range [1, 11] or larger than the input size.
    if ((kMinKernelSize > kernelSize) || (kMaxKernelSize < kernelSize))
    {
        std::stringstream msg{};
        msg << "Invalid kernel size " << kernelSize << ": kernel size must be in range ["
            << kMinKernelSize << ", " << kMaxKernelSize << "]!\n";
        throw std::invalid_argument(msg.str());
    }
    else if (inputSize < kernelSize)
    {
        throw std::invalid_argument(
            "Failed to create convolutional layer: kernel size cannot be greater than input size!");
    }

    const std::size_t padOffset{kernelSize / 2U};

    const std::size_t paddedSize{inputSize + 2U * padOffset};

    // Initialize the matrices with zeros.
    initMatrix(myInputGradients, inputSize);
    initMatrix(myKernel, kernelSize);
    initMatrix(myOutput, inputSize);
    initMatrix(myInputPadded, paddedSize);
    initMatrix(myInputGradientsPadded, paddedSize);
    initMatrix(myKernelGradients, kernelSize);

    for (std::size_t ki{}; ki < myKernel.size(); ++ki)
        for (std::size_t kj{}; kj < myKernel.size(); ++kj)
            myKernel[ki][kj] = randomStartVal();

    // Ignore activation function in this implementation.
    (void) (actFunc);
}


std::size_t Conv::inputSize() const noexcept { return myInputGradients.size(); }


std::size_t Conv::outputSize() const noexcept { return myOutput.size(); }

/**
 * @brief Get the output of the layer.
 * 
 * @return Matrix holding the output of the layer.
 */
const Matrix2d& Conv::output() const noexcept { return myOutput; }

/**
 * @brief Get the input gradients of the layer.
 * 
 * @return Matrix holding the input gradients of the layer.
 */
const Matrix2d& Conv::inputGradients() const noexcept { return myInputGradients; }

/**
 * @brief Perform feedforward operation.
 * 
 * @param[in] input Matrix holding input data.
 * 
 * @return True on success, false on failure.
 */
bool Conv::feedforward(const Matrix2d& input) noexcept
{
    constexpr const char* opName{"feedforward in convolutional layer"};
    if (!matchDimensions(myOutput.size(), input.size(), opName)
        || !isMatrixSquare(input, opName))
    {
        return false;
    }

    padInput(input);

    for (std::size_t i{}; i < myOutput.size(); ++i)
    {
        for (std::size_t j{}; j < myOutput.size(); ++j)
        {
            double sum{myBias};

            for (std::size_t ki{}; ki < myKernel.size(); ++ki)
            {
                for (std::size_t kj{}; kj < myKernel.size(); ++kj)
                {
                    sum += myInputPadded[i + ki][j + kj] * myKernel[ki][kj];
                }
            }

            myOutput[i][j] = reluOutput(sum);  // ev. aktiveringsfunktion här
        }
    }

    return true;
}

/**
 * @brief Perform backpropagation.
 * 
 * @param[in] outputGradients Matrix holding gradients from the next layer.
 * 
 * @return True on success, false on failure.
 */
bool Conv::backpropagate(const Matrix2d& outputGradients) noexcept
{
    constexpr const char* opName{"backpropagation in convolutional layer"};
    if (!matchDimensions(myOutput.size(), outputGradients.size(), opName)
        || !isMatrixSquare(outputGradients, opName))
    {
        return false;
    }

    initMatrix(myInputGradientsPadded);
    initMatrix(myInputGradients);
    initMatrix(myKernelGradients);
    myBiasGradient = 0.0;

    for (std::size_t i{}; i < myOutput.size(); ++i)
    {
        for (std::size_t j{}; j < myOutput.size(); ++j)
        {
            const double delta{outputGradients[i][j]};

            myBiasGradient += delta;

            for (std::size_t ki{}; ki < myKernel.size(); ++ki)
            {
                for (std::size_t kj{}; kj < myKernel.size(); ++kj)
                {
                    myKernelGradients[ki][kj] += myInputPadded[i + ki][j + kj] * delta;

                    myInputGradientsPadded[i + ki][j + kj] += myKernel[ki][kj] * delta;
                }
            }
        }
    }

    extractInputGradients();
    return true;
}

/**
 * @brief Perform optimization.
 * 
 * @param[in] learningRate Learning rate to use.
 * 
 * @return True on success, false on failure.
 */
bool Conv::optimize(const double learningRate) noexcept 
{
    constexpr const char* opName{"optimization in convolutional layer"};
    if (!checkLearningRate(learningRate, opName))
    {
        return false;
    }

    myBias -= myBiasGradient * learningRate;

    for (std::size_t ki{}; ki < myKernel.size(); ++ki)
    {
        for (std::size_t kj{}; kj < myKernel.size(); ++kj)
        {
            myKernel[ki][kj] -= myKernelGradients[ki][kj] * learningRate;
        }
    }

    return true;
}

 /**
 * @brief Pad input with zeros.
 * 
 * @param[in] input Input data.
 */
void Conv::padInput(const Matrix2d& input) noexcept
{
    // Compute the pad offset (the number of zeros in each direction).
    const std::size_t padOffset{myKernel.size() / 2U};

    // Ensure that the padded input matrix is filled with zeros only.
    initMatrix(myInputPadded);

    // Copy the input values to the corresponding padded matrix.
    for (std::size_t i{}; i < myOutput.size(); ++i)
    {
        for (std::size_t j{}; j < myOutput.size(); ++j)
        {
            myInputPadded[i + padOffset][j + padOffset] = input[i][j];
        }
    }
}

/**
 * @brief Extract input gradients.
 */
void Conv::extractInputGradients() noexcept
{
    // Compute the pad offset (the number of zeros in each direction).
    const std::size_t padOffset{myKernel.size() / 2U};

    for (std::size_t i{}; i < myOutput.size(); ++i)
    {
        for (std::size_t j{}; j < myOutput.size(); ++j)
        {
            myInputGradients[i][j] = myInputGradientsPadded[i + padOffset][j + padOffset];
        }
    }
}
} // namespace ml::conv_layer
