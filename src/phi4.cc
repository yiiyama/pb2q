#include <complex>
#include <tuple>
#include <utility>
#include <vector>
#include <map>
#include <cmath>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "combinatorics.h"
#include "momentum.h"

namespace py = pybind11;
using complex128 = std::complex<double>;
using CSRData = std::tuple<py::array_t<complex128>, py::array_t<unsigned>, py::array_t<unsigned> >;
using Signs = std::array<int, 4>;

extern "C" {
  CSRData make_h1_matrix_1d(
    unsigned nParticles,
    std::vector<Momentum<1> >& momenta
  );
  // CSRData make_h1_matrix_2d(
  //   unsigned nParticles,
  //   std::vector<std::array<int, 2>& momenta
  // );
}

int
opSign(unsigned iPatt, unsigned iPart)
{
  return ((iPatt >> iPart) & 1) * 2 - 1;
}

template<unsigned NDIM>
CSRData
make_h1_matrix(
  unsigned nParticles,
  std::vector<Momentum<NDIM> >& momenta,
  double mass
)
{
  typedef std::array<unsigned, 4> VtxMomentumIndices;

  unsigned nMax(nParticles + 1);

  // kinematically allowed momentum combinations for each operator pattern
  std::array<std::vector<VtxMomentumIndices>, 16> allowedMomenta;
  for (unsigned iPatt(0); iPatt != 16; ++iPatt) {
    auto signs{{opSign(iPatt, 0), opSign(iPatt, 1), opSign(iPatt, 2), opSign(iPatt, 3)}};
    for (unsigned iP0(0); iP0 != momenta.size(); ++iP0) {
      for (unsigned iP1(0); iP1 != momenta.size(); ++iP1) {
        for (unsigned iP2(0); iP2 != momenta.size(); ++iP2) {
          for (unsigned iP3(0); iP3 != momenta.size(); ++iP3) {
            bool vanishes{true};
            for (unsigned iDim(0); iDim != NDIM; ++iDim) {
              int total{momenta[iP0][iDim] * signs[0]
                        + momenta[iP1][iDim] * signs[1]
                        + momenta[iP2][iDim] * signs[2]
                        + momenta[iP3][iDim] * signs[3]};
              if (total != 0) {
                vanishes = false;
                break;
              }
            }
            if (vanishes)
              allowedMomenta[iPatt].emplace_back({iP0, iP1, iP2, iP3});
          }
        }
      }
    }
  }

  std::vector<unsigned> nPartBlocks(1, 0);
  std::vector<std::vector<unsigned> > pCombs;
  for (unsigned nPart(0); nPart != nMax; ++nPart) {
    pCombs.insert(pCombs.end(), combinationsWithReplacement(momenta.size(), nPart));
    nPartBlocks.push_back(pCombs.size());
  }

  // map combination to pTotal sector
  std::map<Momentum<NDIM>, unsigned> pTotalSectors;
  std::vector<unsigned> combSectors(pCombs.size(), -1);
  for (unsigned iComb(0); iComb != pCombs.size(); ++iComb) {
    Momentum<NDIM> pTotal;
    for (unsigned iMom : pCombs[iComb])
      pTotal += momenta[iMom];
    auto itr{pTotalSectors.find(pTotal)};
    unsigned sector;
    if (itr == pTotalSectors.end()) {
      sector = pTotalSectors.size();
      pTotalSectors[pTotal] = sector;
    }
    else
      sector = itr->second;

    combSectors[iComb] = sector;
  }

  // reverse mapping sector -> combinations
  std::vector<std::vector<unsigned> > sectorized(pTotalSectors.size());
  for (unsigned iComb(0); iComb != pCombs.size(); ++iComb)
    sectorized[combSectors].push_back(iComb);

  // Compute matrix elements
  std::vector<std::pair<unsigned, unsigned>, complex128> matrixData;
  for (unsigned nBra(0); nBra != nMax; ++nBra) {
    for (unsigned nKet(0); nKet <= nBra; ++nKet) {
      std::vector<unsigned> ladderPatterns;
      for (unsigned iPatt(0); iPatt != 16; ++iPatt) {
        int nPart(nKet);
        for (unsigned iPart(0); iPart != 4; ++iPart) {
          nPart += opSign(iPatt, iPart);
          if (nPart < 0)
            break;
        }
        if (nPart < 0)
          continue;
        if (nPart - nBra == 0)
          ladderPatterns.push_back(iPatt);
      }
      if (ladderPatterns.empty())
        continue;

      for (unsigned iKet(nPartBlocks[nKet]); iKet != nPartBlocks[nKet + 1]; ++iKet) {
        std::map<Momentum<NDIM>, unsigned> pCountsKet;
        for (unsigned pIdx : pCombs[iKet])
          ++pCountsKet[pIdx];

        for (unsigned iBra : sectorized[pTotalSectors[iKet]]) {
          if (iBra > iKet)
            break;

          for (unsigned iPatt : ladderPatterns) {
            for (auto& pIndices : allowedMomenta[iPatt]) {
              auto pCounts(pCountsKet);
              double factor{1.};
              for (unsigned iPart(0); iPart != 4; ++iPart) {
                unsigned count{pCounts[pIndices[iPart]]};
                if (opSign(iPatt, iPart) == -1) {
                  if (count == 0) {
                    factor = 0.;
                    break;
                  }
                  factor *= std::sqrt(count);
                  --pCounts[pIndices[iPart]];
                }
                else {
                  factor *= std::sqrt(count + 1);
                  ++pCounts[pIndices[iPart]];
                }
              }
              if (factor == 0.)
                continue;
              for (unsigned pIdx : pCombs[iBra]) {
                if (pCounts[pIdx]-- == 0) {
                  factor = 0.;
                  break;
                }
              }
              if (factor == 0.)
                continue;

              for (unsigned pIdx : pIndices)
                factor /= std::sqrt(2. * momenta[pIdx].energy(mass));

              matrixData[std::make_pair(iBra, iKet)] += complex128(factor, 0.);
            }
          }
        }
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
