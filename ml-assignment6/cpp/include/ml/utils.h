/**
 * @brief Machine learning utility functions.
 */
#pragma once

#include <iostream>

#include "ml/types.h"

namespace ml
{
/**
 * @brief Initialize one-dimensional matrix with zeros.
 * 
 * @param[in] matrix The matrix to initialize.
 */
void initMatrix(Matrix1d& matrix) noexcept;

/**
 * @brief Initialize two-dimensional matrix with zeros.
 * 
 * @param[in] matrix The matrix to initialize.
 */
void initMatrix(Matrix2d& matrix) noexcept;

/**
 * @brief Initialize one-dimensional matrix with zeros.
 * 
 *        Resize the matrix if necessary.
 * 
 * @param[in] matrix The matrix to initialize.
 * @param[in] size The desired size of the matrix.
 */
void initMatrix(Matrix1d& matrix, std::size_t size);

/**
 * @brief Initialize two-dimensional matrix with zeros.
 * 
 *        Resize the matrix if necessary.
 * 
 * @param[in] matrix The matrix to initialize.
 * @param[in] size The desired size of the matrix.
 */
void initMatrix(Matrix2d& matrix, std::size_t size);

/**
 * @brief Initialize two-dimensional matrix with zeros.
 * 
 * @param[in] matrix The matrix to initialize.
 * @param[in] rowCount The desired row count of the matrix.
 * @param[in] colCount The desired column count of the matrix.
 */
void initMatrix(Matrix2d& matrix, std::size_t rowCount, std::size_t colCount);

/**
 * @brief Check whether given matrix is square. Print an error message if not.
 * 
 * @param[in] matrix The matrix to check.
 * @param[in] opName Operation name (default = none).
 * 
 * @return True if given matrix is square, false otherwise.
 */
bool isMatrixSquare(const Matrix2d& matrix, const char* opName = nullptr) noexcept;

/**
 * @brief Check whether given matrix is square. Print an error message if not.
 * 
 * @param[in] matrix The matrix to check.
 * @param[in] row The row to check.
 * @param[in] opName Operation name (default = none).
 * 
 * @return True if given matrix is square, false otherwise.
 */
bool isMatrixSquare(const Matrix2d& matrix, std::size_t row, 
                    const char* opName = nullptr) noexcept;

/**
 * @brief Print the contents of given one-dimensional matrix.
 * 
 * @param[in] matrix The matrix to print.
 * @param[in] ostream Output stream (default = terminal print).
 * @param[in] precision Decimal precision (default = 1 decimal).
 */
void printMatrix(const ml::Matrix1d& matrix, std::ostream& ostream = std::cout,
                 std::size_t precision = 1U) noexcept;

/**
 * @brief Print the contents of given two-dimensional matrix.
 * 
 * @param[in] matrix The matrix to print.
 * @param[in] ostream Output stream (default = terminal print).
 * @param[in] precision Decimal precision (default = 1 decimal).
 */
void printMatrix(const ml::Matrix2d& matrix, std::ostream& ostream = std::cout,
                 std::size_t precision = 1U) noexcept;

/**
 * @brief Match dimensions. Print an error message on mismatch.
 * 
 * @param[in] expectedSize Expected size.
 * @param[in] actualSize Actual size.
 * @param[in] opName Operation name (default = none).
 * 
 * @return True if the dimensions match, false otherwise.
 */
bool matchDimensions(std::size_t expectedSize, std::size_t actualSize, 
                     const char* opName = nullptr) noexcept;

/**
 * @brief Check learning rate. Print an error message if invalid.
 * 
 * @param[in] learningRate The learning rate to check.
 * @param[in] opName Operation name (default = none).
 * 
 * @return True if the learning rate is valid, false otherwise.
 */
bool checkLearningRate(double learningRate, const char* opName = nullptr) noexcept;

/**
 * @brief Get a randomized starting value for trainable parameters.
 * 
 * @return Random value in the range [0.0, 1.0] (inclusive).
 */
double randomStartVal() noexcept;

/**
 * @brief Create a training order list.
 * 
 * @param[in] trainSetCount The number of training sets.
 * 
 * @return The new training order list.
 */
TrainOrderList createTrainOrderList(std::size_t trainSetCount);

/**
 * @brief Shuffle training order list.
 * 
 * @param[in] list Training order list to shuffle.
 */
void shuffleTrainOrderList(TrainOrderList& list) noexcept;

} // namespace ml
