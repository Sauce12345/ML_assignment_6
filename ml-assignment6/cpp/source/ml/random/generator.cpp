/** 
 * @brief Random generator implementation details.
 */
#include <cstdint>
#include <cstdlib>
#include <ctime>

#include "ml/random/generator.h"

namespace ml::random
{
// -----------------------------------------------------------------------------
Interface& Generator::getInstance() noexcept
{
    // Create and initialize singleton random generator instance (once only).
    static Generator myInstance{};

    // Return a reference to the random generator.
    return myInstance;
}

// -----------------------------------------------------------------------------
std::uint32_t Generator::uint32(const std::uint32_t maxExclusive) const noexcept
{
    // Return a random integer in the range [0, maxExclusive - 1).
    return static_cast<std::uint32_t>(std::rand() % maxExclusive);
}

// -----------------------------------------------------------------------------
std::int32_t Generator::int32(const std::int32_t min, const std::int32_t max) const noexcept
{
    // Return min if min is larger than or equal to max.
    if (min >= max) { return min; }
    
    // Return a random integer in the range [min, max].
    return std::rand() % (max - min + 1) + min;
}

// -----------------------------------------------------------------------------
double Generator::float64(const double min, const double max) const noexcept
{
    // Return min if min is larger than or equal to max.
    if (min >= max) { return min; }

    // Return a random floating point number in the range [min, max).
    return (std::rand() / static_cast<double>(RAND_MAX)) * (max - min) + min;
}

// -----------------------------------------------------------------------------
Generator::Generator() noexcept
{
    // Initialize the random generator with the current time as seed.
    std::srand(std::time(nullptr));
}
} // namespace ml::random
