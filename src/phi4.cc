#include <complex>
#include <tuple>
#include <utility>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "combinatorics.h"
#include "momentum.h"

namespace py = pybind11;
typedef std::complex<double> complex128;
typedef std::vector<py::array_t<unsigned> > Basis;
typedef std::tuple<py::array_t<complex128>, py::array_t<unsigned>, py::array_t<unsigned> > CSRData;

extern "C" {
  std::pair<Basis, CSRData> make_h1_matrix_1d(unsigned, py::array_t<int>&, double);
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
std::pair<Basis, CSRData>
make_h1_matrix(
  unsigned nParticles,
  py::array_t<int>& momentaArray,
  double mass
)
{
  typedef std::array<unsigned, 4> VtxMomentumIndices;

  if (!((NDIM == 1 && momentaArray.ndim() == 1) || momentaArray.shape()[1] == NDIM))
    throw std::runtime_error("Wrong momentum shape");

  std::vector<Momentum<NDIM> > momenta{};
  for (unsigned iP(0); iP != momentaArray.shape()[0]; ++iP)
    momenta.emplace_back(momentaArray.data(iP));

  std::cout << "filled momenta" << std::endl;

  unsigned nMax(nParticles + 1);

  // kinematically allowed momentum combinations for each operator pattern
  std::array<std::vector<VtxMomentumIndices>, 16> allowedMomenta;
  for (unsigned iPatt(0); iPatt != 16; ++iPatt) {
    std::array<int, 4> signs{opSign(iPatt, 0), opSign(iPatt, 1), opSign(iPatt, 2), opSign(iPatt, 3)};
    for (unsigned iP0(0); iP0 != momenta.size(); ++iP0) {
      for (unsigned iP1(0); iP1 != momenta.size(); ++iP1) {
        for (unsigned iP2(0); iP2 != momenta.size(); ++iP2) {
          for (unsigned iP3(0); iP3 != momenta.size(); ++iP3) {
            bool vanishes(true);
            for (unsigned iDim(0); iDim != NDIM; ++iDim) {
              int total(momenta[iP0][iDim] * signs[0]
                        + momenta[iP1][iDim] * signs[1]
                        + momenta[iP2][iDim] * signs[2]
                        + momenta[iP3][iDim] * signs[3]);
              if (total != 0) {
                vanishes = false;
                break;
              }
            }
            if (vanishes)
              allowedMomenta[iPatt].push_back({iP0, iP1, iP2, iP3});
          }
        }
      }
    }
  }

  std::cout << "computed allowed momenta" << std::endl;

  std::vector<std::vector<std::vector<unsigned> > > pExtBlocks;
  std::vector<std::vector<unsigned> > pExts;
  for (unsigned nPart(0); nPart != nMax; ++nPart) {
    pExtBlocks.push_back(combinationsWithReplacement(momenta.size(), nPart));
    pExts.insert(pExts.end(), pExtBlocks.back().begin(), pExtBlocks.back().end());
  }

  // map combination to pTotal sector
  std::map<Momentum<NDIM>, unsigned> pTotalSectors;
  std::vector<unsigned> combSectors(pExts.size(), -1);
  for (unsigned iExt(0); iExt != pExts.size(); ++iExt) {
    Momentum<NDIM> pTotal;
    for (unsigned iMom : pExts[iExt])
      pTotal += momenta[iMom];
    auto itr(pTotalSectors.find(pTotal));
    unsigned sector;
    if (itr == pTotalSectors.end()) {
      sector = pTotalSectors.size();
      pTotalSectors[pTotal] = sector;
    }
    else
      sector = itr->second;

    combSectors[iExt] = sector;
  }

  // reverse mapping sector -> combinations
  std::vector<std::vector<unsigned> > sectorized(pTotalSectors.size());
  for (unsigned iExt(0); iExt != pExts.size(); ++iExt)
    sectorized[combSectors[iExt]].push_back(iExt);

  // Compute matrix elements
  std::map<std::pair<unsigned, unsigned>, complex128> matrixData;
  for (unsigned nBra(0); nBra != nMax; ++nBra) {
    unsigned iKetStart(0);
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

      for (unsigned iKet(iKetStart); iKet != iKetStart + pExtBlocks[nKet].size(); ++iKet) {
        std::map<Momentum<NDIM>, unsigned> pCountsKet;
        for (unsigned pIdx : pExts[iKet])
          ++pCountsKet[pIdx];

        for (unsigned iBra : sectorized[pTotalSectors[iKet]]) {
          if (iBra > iKet)
            break;

          for (unsigned iPatt : ladderPatterns) {
            for (auto& pIndices : allowedMomenta[iPatt]) {
              auto pCounts(pCountsKet);
              double factor(1.);
              for (unsigned iPart(0); iPart != 4; ++iPart) {
                unsigned count(pCounts[pIndices[iPart]]);
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
              for (unsigned pIdx : pExts[iBra]) {
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
      iKetStart += pExtBlocks[nKet].size();
    }
  }

  Basis basis{};
  for (unsigned nPart(1); nPart != nMax; ++nPart) {
    auto& block(pExtBlocks[nPart]);
    basis.push_back(py::array_t<unsigned>({std::size_t(block.size()), std::size_t(nPart)}));
    for (unsigned iExt(0); iExt != pExtBlocks[nPart].size(); ++iExt)
      std::copy(block[iExt].begin(), block[iExt].end(), basis.back().mutable_data(iExt, 0));
  }

  CSRData csrData{
    py::array_t<complex128>(matrixData.size()),
    py::array_t<unsigned>(matrixData.size()),
    py::array_t<unsigned>(pExts.size() + 1)
  };

  unsigned iElem(0);
  unsigned iRow{0};
  for (auto& melem : matrixData) {
    unsigned row(melem.first.first);
    unsigned col(melem.first.second);
    while (iRow < row)
      std::get<2>(csrData).mutable_at(iRow++) = iElem;
    std::get<0>(csrData).mutable_at(iElem) = melem.second;
    std::get<1>(csrData).mutable_at(iElem) = col;
  }

  return {basis, csrData};
}

std::pair<Basis, CSRData>
make_h1_matrix_1d(
  unsigned nParticles,
  py::array_t<int>& momenta,
  double mass
)
{
  return make_h1_matrix<1>(nParticles, momenta, mass);
}

PYBIND11_MODULE(phi4, module) {
  module.doc() = "Scalar phi4 theory";
  module.def("make_h1_matrix_1d", &make_h1_matrix_1d, "Interaction Hamiltonian");
}
