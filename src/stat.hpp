#ifndef __STAT__
#define __STAT__
#include <dqrng_distribution.h>
namespace eif {

uint64_t get_seed();

std::function<double()> gen_runif(
    dqrng::rng64_t &rng, double min = 0.0, double max = 1.0);

std::function<double()> gen_rnorm(
    dqrng::rng64_t &rng, double mean = 0.0, double sd = 1.0);

std::vector<double> runif(
    dqrng::rng64_t &rng, size_t n, double min = 0.0, double max = 1.0);

std::vector<double> rnorm(
    dqrng::rng64_t &rng, size_t n, double mean = 0.0, double sd = 1.0);

std::vector<uint32_t> replacement(
    dqrng::rng64_t &rng, uint32_t m, uint32_t n, int offset);

std::vector<uint32_t> no_replacement_shuffle(
    dqrng::rng64_t &rng, uint32_t m, uint32_t n, int offset);

template<typename SET>
std::vector<uint32_t> no_replacement_set(
    dqrng::rng64_t &rng, uint32_t m, uint32_t n, int offset);

std::vector<uint32_t> sample(
    dqrng::rng64_t &rng, uint32_t m, uint32_t n, bool replace, int offset);

std::vector<uint32_t> sample_int(dqrng::rng64_t &rng, int m, int n,
                                        bool replace = false);
}
#endif // __STAT__
