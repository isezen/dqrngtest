#include <dqrng_distribution.h>
#include <minimal_int_set.h>
#include <R_randgen.h>
#include <convert_seed.h>
#include <numeric>
#include <vector>
#include <array>
namespace eif {
  dqrng::uniform_distribution uniform{};
  dqrng::normal_distribution normal{};

  uint64_t get_seed() {
    std::vector<int> seed(2);
    seed[0] = dqrng::R_random_int();
    seed[1] = dqrng::R_random_int();
    return(dqrng::convert_seed<uint64_t>(seed[0]));
  }

  std::function<double()> gen_runif(
      dqrng::rng64_t &rng, double min = 0.0, double max = 1.0) {
    using parm_t = decltype(uniform)::param_type;
    uniform.param(parm_t(min, max));
    return(std::bind(uniform, std::ref(*rng)));
  }

  std::function<double()> gen_rnorm(
      dqrng::rng64_t &rng, double mean = 0.0, double sd = 1.0) {
    using parm_t = decltype(normal)::param_type;
    normal.param(parm_t(mean, sd));
    return(std::bind(normal, std::ref(*rng)));
  }

  std::vector<double> runif(
      dqrng::rng64_t &rng, size_t n, double min = 0.0, double max = 1.0) {
    if(max / 2. - min / 2. > (std::numeric_limits<double>::max)() / 2.) {
      std::vector<double> v = runif(rng, n, min/2., max/2.);
      std::transform(v.begin(), v.end(), v.begin(), [] (double& c){return(c * 2);});
      return(v);
    }
    auto gen = gen_runif(rng, min, max);
    std::vector<double> v(n);
    std::generate(v.begin(), v.end(), std::ref(gen));
    return(v);
  }

  std::vector<double> rnorm(
      dqrng::rng64_t &rng, size_t n, double mean = 0.0, double sd = 1.0) {
    auto gen = gen_rnorm(rng, mean, sd);
    std::vector<double> v(n);
    std::generate(v.begin(), v.end(), std::ref(gen));
    return(v);
  }

  std::vector<uint32_t> replacement(
      dqrng::rng64_t &rng, uint32_t m, uint32_t n, int offset) {
    std::vector<uint32_t> result(n);
    std::generate(result.begin(), result.end(),
                  [m, offset, rng] () {
                    return static_cast<uint32_t>(offset + (*rng)(m));
                  });
    return(result);
  }

  std::vector<uint32_t> no_replacement_shuffle(
      dqrng::rng64_t &rng, uint32_t m, uint32_t n, int offset) {
    std::vector<uint32_t> tmp(m);
    std::iota(tmp.begin(), tmp.end(), static_cast<uint32_t>(offset));
    for (uint32_t i = 0; i < n; ++i) std::swap(tmp[i], tmp[i + (*rng)(m - i)]);
    if (m == n) return(tmp);
    return(std::vector<uint32_t>(tmp.begin(), tmp.begin() + n));
  }

  template<typename SET>
  std::vector<uint32_t> no_replacement_set(
      dqrng::rng64_t &rng, uint32_t m, uint32_t n, int offset) {
    std::vector<uint32_t> result(n);
    SET elems(m, n);
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t v = (*rng)(m);
      while (!elems.insert(v)) {
        v = (*rng)(m);
      }
      result[i] = static_cast<uint32_t>(offset + v);
    }
    return(result);
  }

  std::vector<uint32_t> sample(
      dqrng::rng64_t &rng, uint32_t m, uint32_t n, bool replace, int offset) {
    if (replace || n <= 1) {
      return(replacement(rng, m, n, offset));
    } else {
      if (!(m >= n))
        throw std::string("Argument requirements not fulfilled: m >= n");
      if (m < 2 * n) {
        return(no_replacement_shuffle(rng, m, n, offset));
      } else if (m < 1000 * n) {
        return(no_replacement_set<dqrng::minimal_bit_set>(rng, m, n, offset));
      } else {
        return(no_replacement_set<dqrng::minimal_hash_set<uint32_t>>(rng, m, n, offset));
      }
    }
  }

  std::vector<uint32_t> sample_int(dqrng::rng64_t &rng, int m, int n,
                                          bool replace = false) {
    if (!(m > 0 && n >= 0))
      throw std::string("Argument requirements not fulfilled: m > 0 && n >= 0");
    return(sample(rng, uint32_t(m), uint32_t(n), replace, 0));
  }
}
