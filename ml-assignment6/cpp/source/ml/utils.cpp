/**
 * @brief Machine learning utility functions.
 */
#include <iomanip>
#include <iostream>

#include "ml/random/generator.h"
#include "ml/types.h"
#include "ml/utils.h"

namespace ml
{
// -----------------------------------------------------------------------------
void initMatrix(Matrix1d& matrix) noexcept
{ 
    // Fill the matrix with zeros.
    for (auto& num : matrix) { num = 0.0; }
}

// -----------------------------------------------------------------------------
void initMatrix(Matrix2d& matrix) noexcept
{ 
    // Fill the matrix with zeros.
    for (auto& row : matrix)
    {
        for (auto& column : row) { column = 0.0; }
    }
}

// -----------------------------------------------------------------------------
void initMatrix(Matrix1d& matrix, const std::size_t size)
{ 
    // Resize the matrix if necessary.
    matrix.resize(size);

    // Fill the matrix with zeros.
    initMatrix(matrix);
}

// -----------------------------------------------------------------------------
void initMatrix(Matrix2d& matrix, const std::size_t size)
{ 
    // Resize the matrix f necessary.
    matrix.resize(size, Matrix1d(size));

    // Fill the matrix with zeros.
    initMatrix(matrix);
}

// -----------------------------------------------------------------------------
void initMatrix(Matrix2d& matrix, const std::size_t rowCount, const std::size_t colCount)
{
    // Resize the matrix f necessary.
    matrix.resize(rowCount, Matrix1d(colCount));

    // Fill the matrix with zeros.
    initMatrix(matrix);
}

// -----------------------------------------------------------------------------
bool isMatrixSquare(const Matrix2d& matrix, const char* opName) noexcept
{
    // Check each row in the matrix.
    for (std::size_t i{}; i < matrix.size(); ++i)
    {
        // Print an error message and return false if the matrix isn't square.
        if (matrix[i].size() != matrix.size()) 
        { 
            if (nullptr != opName)
            {
                std::cerr << "Cannot perform " << opName << " due to matrix not being square!\n";
            }
            else { std::cout << "Matrix is not square!\n"; }
            return false; 
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
bool isMatrixSquare(const Matrix2d& matrix, const std::size_t row, const char* opName) noexcept
{
    // Check the specified row in the matrix if valid. Print an error message if not.
    if (matrix.size() <= row)
    {
        std::cerr << "Invalid row " << row << " in matrix of size " << matrix.size() << "!\n";
        return false;
    }

    // Print an error message and return false if the matrix isn't square.
    if (matrix[row].size() != matrix.size()) 
    { 
        if (nullptr != opName)
        {
            std::cerr << "Cannot perform " << opName << " due to matrix not being square!\n";
        }
        else { std::cout << "Matrix is not square!\n"; }
        return false; 
    }
    return true;
}

// -----------------------------------------------------------------------------
void printMatrix(const ml::Matrix2d& matrix, std::ostream& ostream,
                 const std::size_t precision) noexcept
{
    // Set output formatting for floating point numbers.
    ostream << std::fixed << std::setprecision(precision) << "[";
    const auto* lastColumn{matrix.empty() ? nullptr : &matrix[matrix.size() - 1U]};

    // Iterate over each row in the matrix.
    for (const auto& column : matrix)
    {
        ostream << "[";
        const auto* lastRow{column.empty() ? nullptr : &column[column.size() - 1U]};

        // Print each element in the row.
        for (const auto& row : column)
        {
            ostream << row;
            if (&row < lastRow) { ostream << ", "; }
        }
        // Print row separator or closing bracket.
        if (&column < lastColumn) { ostream << "],\n"; }
        else { ostream << "]"; }
    }
    ostream << "]\n";
}

// -----------------------------------------------------------------------------
void printMatrix(const ml::Matrix1d& matrix, std::ostream& ostream,
                 const std::size_t precision) noexcept
{
    // Set output formatting for floating point numbers.
    ostream << std::fixed << std::setprecision(precision) << "[";

    const auto* lastNum{matrix.empty() ? nullptr : &matrix[matrix.size() - 1U]};

    // Iterate through the matrix.
    for (const auto& num : matrix)
    {
        ostream << num;
        if (&num < lastNum) { ostream << ", "; }
    }
    ostream << "]\n";
}

// -----------------------------------------------------------------------------
bool matchDimensions(const std::size_t expectedSize, const std::size_t actualSize, 
                     const char* opName) noexcept
{
    // Return true if the dimensions match.
    if (expectedSize == actualSize) { return true; }

    // Print an error message on mismatch.
    if (nullptr != opName)
    {
        std::cerr << "Cannot perform " << opName << " due to dimension mismatch: "
                  << " expected " << expectedSize << ", actual is " << actualSize << "!\n";
    }
    else
    {
        std::cerr << "Dimension mismatch: expected " << expectedSize 
                  << ", actual is " << actualSize << "!\n";
    }
    // Return false to indicate mismatch.
    return false;
}

// -----------------------------------------------------------------------------
bool checkLearningRate(const double learningRate, const char* opName) noexcept
{
    // Check the learning rate, return true if valid.
    // Print an error message and return false if invalid.
    if (0.0 >= learningRate)
    {
        if (nullptr != opName)
        {
            std::cerr << "Cannot perform " << opName << ": Invalid learning rate!\n";
        }
        else { std::cerr << "Invalid learning rate!\n"; }
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
double randomStartVal() noexcept
{
    constexpr double min{0.0};
    constexpr double max{1.0};

    // Return a random starting value in the range [0.0, 1.0] (inclusive).
    return random::Generator::getInstance().float64(min, max);
}

// -----------------------------------------------------------------------------
TrainOrderList createTrainOrderList(const std::size_t trainSetCount)
{
    // Create a new training order list.
    TrainOrderList list(trainSetCount);

    // Initialize the list with training set indexes in ascending order.
    for (std::size_t i{}; i < trainSetCount; ++i)
    {
        list[i] = i;
    }
    return list;
}

// -----------------------------------------------------------------------------
void shuffleTrainOrderList(TrainOrderList& list) noexcept
{
    // Shuffle the content of the training order list.
    for (std::size_t i{}; i < list.size(); ++i)
    {
        const auto r{random::Generator::getInstance().uint32(list.size())};
        const auto temp{list[i]};
        list[i] = list[r];
        list[r] = temp;
    }
}
} // namespace ml
