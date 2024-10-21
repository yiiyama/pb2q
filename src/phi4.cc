#include <complex>
#include <tuple>
#include <utility>
#include <vector>
#include <map>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "combinatorics.h"

namespace py = pybind11;
using complex128 = std::complex<double>;
using CSRData = std::tuple<py::array_t<complex128>, py::array_t<unsigned>, py::array_t<unsigned> >;
using Signs = std::array<int, 4>;

extern "C" {
  CSRData make_h1_matrix_1d(
    unsigned nParticles,
    std::vector<std::array<int, 1> >& momenta
  );
  // CSRData make_h1_matrix_2d(
  //   unsigned nParticles,
  //   std::vector<std::array<int, 2>& momenta
  // );
}

template<unsigned NDIM>
CSRData
make_h1_matrix(
  unsigned nParticles,
  std::vector<std::array<int, NDIM> >& momenta
)
{
  using Momentum = std::array<int, NDIM>;
  using VertexMomenta = std::array<Momentum, 4>;

  // annilation and creation op combinations
  Signs allSigns[16];
  for (unsigned iSign(0); iSign != 16; ++iSign) {
    allSigns[iSign] = {
      (iSign & 1) == 1 ? -1 : 1,
      ((iSign >> 1) & 1) == 1 ? -1 : 1,
      ((iSign >> 2) & 1) == 1 ? -1 : 1,
      ((iSign >> 3) & 1) == 1 ? -1 : 1
    };
  }

  unsigned nMax(nParticles + 1);

  // possible annihilation and creation op combinations for initial & final num particles
  auto ladderPatterns{std::map<std::pair<unsigned, unsigned>, std::vector<unsigned> >()};
  for (unsigned nBra(0); nBra != nMax; ++nBra) {
    for (unsigned nKet(0); nKet != nMax; ++nKet) {
      for (unsigned iSign(0); iSign != 16; ++iSign) {
        int nPart(nKet);
        for (int sign : allSigns[iSign]) {
          nPart += sign;
          if (nPart < 0) {
            break;
          }
        }
        if (nPart >= 0) {
          nPart -= nBra;
        }
        if (nPart == 0) {
          ladderPatterns[std::make_pair(nBra, nKet)].push_back(iSign);
        }
      }
    }
  }

  // kinematically allowed momentum combinations for each operator pattern
  auto allowedMomenta{std::array<std::vector<VertexMomenta>, 16>()};
  for (unsigned iSign(0); iSign != 16; ++iSign) {
    auto& signs{allSigns[iSign]};
    for (auto& p0 : momenta) {
      for (auto& p1 : momenta) {
        for (auto& p2 : momenta) {
          for (auto& p3 : momenta) {
            bool vanishes{true};
            for (unsigned iDim(0); iDim != NDIM; ++iDim) {
              auto total{p0[iDim] * signs[0] + p1[iDim] * signs[1] + p2[iDim] * signs[2] + p3[iDim] * signs[3]};
              if (total != 0) {
                vanishes = false;
                break;
              }
            }
            if (vanishes) {
              allowedMomenta[iSign].push_back({p0, p1, p2, p3});
            }
          }
        }
      }
    }
  }

  std::vector<unsigned> basisOffsets{};
  unsigned nBasis{0};
  basisOffsets.push_back(nBasis);
  for (unsigned nPart(0); nPart != nMax; ++nPart)
    basisOffsets.push_back(basisOffsets.back() + comb(momenta.size() - 1 + nPart, nPart));

  // Compute matrix elements
  for (unsigned nBra(0); nBra != nMax; ++nBra) {
    for (unsigned nKet(0); nKet != nMax; ++nKet) {
      auto iSignItr{ladderPatterns.find(std::make_pair(nBra, nKet))};
      if (iSignItr == ladderPatterns.end()) {
        continue;
      }

      auto braCombinations{combinationsWithReplacement<..., >()}
      for (unsigned iBraComb(0); iBraComb != basisOffsets[nBra + 1] - basisOffsets[nBra]; ++iBraComb) {
        int ptotal{0};
        for ()

      }
    }
  }

  auto data{py::array_t<complex128>(2)};
  auto indices{py::array_t<unsigned>(2)};
  auto indptr{py::array_t<unsigned>(2)};
  return {data, indices, indptr};
}

CSRData
make_h1_matrix_1d(
  unsigned nParticles,
  std::vector<std::array<int, 1> >& momenta
)
{
  return make_h1_matrix<1>(nParticles, momenta);
}

PYBIND11_MODULE(phi4, module) {
  module.doc() = "Scalar phi4 theory";
  module.def("make_h1_matrix_1d", &make_h1_matrix_1d, "Interaction Hamiltonian");
}
