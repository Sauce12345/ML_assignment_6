/**
 * @brief Stub layer stub.
 */
#pragma once

#include <stdexcept>

#include "ml/flatten_layer/interface.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml::flatten_layer
{
/**s
 * @brief Stub layer stub.
 * 
 *        This class is non-copyable and non-movable.
 */
class Stub final : public Interface
{
public:
    /**
     * @brief Constructor.
     * 
     * @param[in] inputSize Input size. Must be greater than 0.
     */
    explicit Stub(const std::size_t inputSize)
        : myInputGradients{}
        , myOutput{}
    {
        // Check the input size, throw an exception if invalid.
        if (0U == inputSize)
        {
            throw std::invalid_argument("Input size cannot be 0!");
        }

        // Initialize layer matrices.
        const std::size_t outputSize{inputSize * inputSize};
        initMatrix(myInputGradients, inputSize);
        initMatrix(myOutput, outputSize);
    }

    /** 
     * @brief Destructor. 
     */
    ~Stub() noexcept override = default;

    /**
     * @brief Get the input size of the layer.
     * 
     * @return The input size of the layer.
     */
    std::size_t inputSize() const noexcept override { return myInputGradients.size(); }

    /**
     * @brief Get the output size of the layer.
     * 
     * @return The output size of the layer.
     */
    std::size_t outputSize() const noexcept override { return myOutput.size(); }

    /**
     * @brief Get the input gradients of the layer.
     * 
     * @return Matrix holding the input gradients of the layer.
     */
    const Matrix2d& inputGradients() const noexcept override { return myInputGradients; }

    /**
     * @brief Get the flattened output of the layer.
     * 
     * @return Matrix holding the output of the layer.
     */
    const Matrix1d& output() const noexcept override { return myOutput; }

    /**
     * @brief Stub the input from 2D to 1D.
     * 
     * @param[in] input Matrix holding input data.
     * 
     * @return True on success, false on failure.
     */
    bool feedforward(const Matrix2d& input) noexcept override
    {
        // Return true if the input matches the expected (unpadded) output size.
        constexpr const char* opName{"feedforward in flatten layer"};
        return matchDimensions(myInputGradients.size(), input.size(), opName)
            && isMatrixSquare(input, opName);
    }

     /**
     * @brief Unflatten the output gradients from 1D to 2D.
     * 
     * @param[in] outputGradients Matrix holding output gradients.
     * 
     * @return True on success, false on failure.
     */
    bool backpropagate(const Matrix1d& outputGradients) noexcept override
    {
         // Return true if the output dimensions match.
        constexpr const char* opName{"backpropagation in flatten layer"};
        return matchDimensions(myOutput.size(), outputGradients.size(), opName);
    }

    Stub()                       = delete; // No default constructor.
    Stub(const Stub&)            = delete; // No copy constructor.
    Stub(Stub&&)                 = delete; // No move constructor.
    Stub& operator=(const Stub&) = delete; // No copy assignment.
    Stub& operator=(Stub&&)      = delete; // No move assignment.

private:
    /** Input gradients. */
    Matrix2d myInputGradients;

    /** Stubed output. */
    Matrix1d myOutput;
};
} // namespace ml::flatten_layer
