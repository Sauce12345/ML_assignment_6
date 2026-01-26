/**
 * @brief Convolutional layer interface.
 */
#pragma once

#include <memory>

#include "ml/types.h"
#include "ml/conv_layer/interface.h"

namespace ml::conv_layer
{
/** 
 * @brief Convolutional layer interface.
 */
class Conv final: public Interface
{
public:
    /**
     * @brief Constructor.
     * 
     * @param[in] inputSize Input size. Must be greater than 0.
     * @param[in] kernelSize Kernel size. Must be greater than 0 and smaller than the input size.
     * @param[in] actFunc Activation function to use (default = none).
     */
    explicit Conv(const std::size_t inputSize, const std::size_t kernelSize, 
                  const act_func::Type actFunc = act_func::Type::None)

    /** 
     * @brief Destructor. 
     */
    ~Conv() noexcept override = default;

    /**
     * @brief Get the input size of the layer.
     * 
     * @return The input size of the layer.
     */
    std::size_t inputSize() const noexcept override;

    /**
     * @brief Get the output size of the layer.
     * 
     * @return The output size of the layer.
     */
    std::size_t outputSize() const noexcept override;

    /**
     * @brief Get the output of the layer.
     * 
     * @return Matrix holding the output of the layer.
     */
    const Matrix2d& output() const noexcept override;

    /**
     * @brief Get the input gradients of the layer.
     * 
     * @return Matrix holding the input gradients of the layer.
     */
    const Matrix2d& inputGradients() const noexcept override;

    /**
     * @brief Perform feedforward operation.
     * 
     * @param[in] input Matrix holding input data.
     * 
     * @return True on success, false on failure.
     */
    bool feedforward(const Matrix2d& input) noexcept override;

    /**
     * @brief Perform backpropagation.
     * 
     * @param[in] outputGradients Matrix holding gradients from the next layer.
     * 
     * @return True on success, false on failure.
     */
    bool backpropagate(const Matrix2d& outputGradients) noexcept override;

    /**
     * @brief Perform optimization.
     * 
     * @param[in] learningRate Learning rate to use.
     * 
     * @return True on success, false on failure.
     */
    bool optimize(double learningRate) noexcept override;
};
} // namespace ml::conv_layer
