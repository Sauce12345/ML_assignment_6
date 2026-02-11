//! @note Denna headerfil ser mycket bra ut, snyggt!

/**
 * @brief Convolutional layer interface.
 */
#pragma once

#include <memory>

//! @note Sträva efter att sortera headerfilerna alfabetiskt.
#include "ml/types.h"
#include "ml/act_func/type.h"
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
                  const act_func::Type actFunc = act_func::Type::None);

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

    Conv()                       = delete; // No default constructor.
    Conv(const Conv&)            = delete; // No copy constructor.
    Conv(Conv&&)                 = delete; // No move constructor.
    Conv& operator=(const Conv&) = delete; // No copy assignment.
    Conv& operator=(Conv&&)      = delete; // No move assignment.

private:
    /**
     * @brief Pad input with zeros.
     * 
     * @param[in] input Input data.
     */
    void padInput(const Matrix2d& input) noexcept;

    /**
     * @brief Extract input gradients.
     */
    void extractInputGradients() noexcept;

    /** Minimum valid kernel size. */
    static constexpr std::size_t kMinKernelSize{1U};

    /** Minimum valid kernel size. */
    static constexpr std::size_t kMaxKernelSize{11U};

    /** Input matrix (padded with zeros). */
    Matrix2d myInputPadded;

    /** Input gradient matrix (padded with zeros). */
    Matrix2d myInputGradientsPadded;

    /** Input gradient matrix (without padding). */
    Matrix2d myInputGradients;

    /** Kernel matrix (holding weights). */
    Matrix2d myKernel;

    /** Kernel gradient matrix. */
    Matrix2d myKernelGradients;

    /** Output matrix. */
    Matrix2d myOutput;

    /** Bias value. */
    double myBias;

    /** Bias gradient. */
    double myBiasGradient;
};
} // namespace ml::conv_layer
