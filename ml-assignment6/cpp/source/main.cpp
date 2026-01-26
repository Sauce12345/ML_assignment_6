/**
 * @brief Training and prediction of a CNN (Convolutional Neural Network).
 */
#include "ml/cnn/cnn.h"
#include "ml/factory/factory.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace
{
/** Use stub implementations if the STUB compiler flag is set. */
#ifdef STUB
constexpr bool UseStubs{true};
#else
constexpr bool UseStubs{false};
#endif

/**
 * @brief Predict and print with the given CNN (Convolutional Neural Network).
 * 
 * @param[in] cnn The CNN with which to predict.
 * @param[in] inputs Input sets to predict with.
 */
void predictAndPrint(ml::cnn::Interface& cnn, const ml::Matrix3d& inputs) noexcept
{
    // Terminate function if no input sets are available.
    if (inputs.empty()) { return; }
    const auto* last{&inputs[inputs.size() - 1U]};

    std::cout << "--------------------------------------------------------------------------------\n";

    // Perform prediction with each input set, print the predicted output in the terminal.
    for (const auto& input : inputs)
    {
        std::cout << "Input:\n";
        ml::printMatrix(input);
        
        std::cout << "\nPrediction:\n";
        ml::printMatrix(cnn.predict(input));
        
        // Add a blank line before the next print.
        if (&input != last)  {std::cout << "\n";}
    }
    std::cout << "--------------------------------------------------------------------------------\n\n";
}
} // namespace

/**
 * @brief Create and train a CNN (Convolutional Neural Network). Print predictions on success.
 * 
 * @return 0 on success, error code -1 on failure.
 */
int main()
{
    // CNN parameters.
    constexpr std::size_t inputSize{4U};
    constexpr std::size_t kernelSize{2U};
    constexpr ml::act_func::Type convFunc{ml::act_func::Type::Relu};
    constexpr std::size_t poolSize{2U};
    constexpr std::size_t denseOutput{1U};
    constexpr ml::act_func::Type denseFunc{ml::act_func::Type::Tanh};

    // Training parameters.
    constexpr std::size_t epochCount{20000U};
    constexpr double learningRate{0.01};

    // Input data for training (digits 0 - 1).
    const ml::Matrix3d inputs{
        {{1, 1, 1, 1},
         {1, 0, 0, 1},
         {1, 0, 0, 1},
         {1, 1, 1, 1}},
        {{0, 1, 0, 0},
         {0, 1, 0, 0},
         {0, 1, 0, 0},
         {0, 1, 0, 0}},
    };
    // Output data for training (the corresponding numbers).
    const ml::Matrix2d outputs{{0}, {1}};

    // Create a machine learning factory.
    auto factory{ml::factory::create(UseStubs)};

    // Create a CNN.
    ml::cnn::Cnn cnn{*factory, inputSize, kernelSize, convFunc, poolSize, denseOutput, denseFunc};

    // Train the network.
    const bool success{cnn.train(inputs, outputs, epochCount, learningRate)};

    // If training was successful, predict with the input matrices.
    if (success) { predictAndPrint(cnn, inputs); }
    else { std::cout << "Training failed!\n"; }

    // Return 0 on success, -1 on failure.
    return success ? 0 : -1;
}
