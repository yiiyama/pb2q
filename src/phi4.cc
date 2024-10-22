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
#include "pybind11/stl.h"
#include "pybind11/complex.h"
#include "combinatorics.h"
#include "momentum.h"
#include "pyarray.h"

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
  if (!((NDIM == 1 && momentaArray.ndim() == 1) || momentaArray.shape()[1] == NDIM))
    throw std::runtime_error("Wrong momentum shape");

  unsigned nMax(nParticles + 1);

  /* Convert the input momenta arrays to Momenta objects */
  std::vector<Momentum<NDIM> > momenta{};
  for (unsigned iP(0); iP != momentaArray.shape()[0]; ++iP)
    momenta.emplace_back(reinterpret_cast<int const*>(momentaArray.data(iP)));

  /* Collect the indices of kinematically allowed momentum combinations for each operator pattern */
  // We only need 8 vectors because the remaining 8 are copies of the others
  std::array<std::vector<std::array<unsigned, 4> >, 8> allowedMomenta{};
  for (auto& plist : allowedMomenta)
    plist.clear();
  for (unsigned iPatt : {0, 1, 3}) {
    std::array<int, 4> signs{opSign(iPatt, 0), opSign(iPatt, 1), opSign(iPatt, 2), opSign(iPatt, 3)};
    for (unsigned iP0(0); iP0 != momenta.size(); ++iP0) {
      for (unsigned iP1(0); iP1 != momenta.size(); ++iP1) {
        for (unsigned iP2(0); iP2 != momenta.size(); ++iP2) {
          for (unsigned iP3(0); iP3 != momenta.size(); ++iP3) {
            bool vanishes(true);
            for (unsigned iDim(0); vanishes && iDim != NDIM; ++iDim) {
              vanishes = (momenta[iP0][iDim] * signs[0]
                          + momenta[iP1][iDim] * signs[1]
                          + momenta[iP2][iDim] * signs[2]
                          + momenta[iP3][iDim] * signs[3]) == 0;
            }
            if (vanishes)
              allowedMomenta[iPatt].push_back({iP0, iP1, iP2, iP3});
          }
        }
      }
    }
  }
  // One different sign
  allowedMomenta[2].reserve(allowedMomenta[1].size());
  allowedMomenta[4].reserve(allowedMomenta[1].size());
  allowedMomenta[7].reserve(allowedMomenta[1].size());
  for (auto& indices : allowedMomenta[1]) {
    allowedMomenta[2].push_back({indices[1], indices[0], indices[2], indices[3]});
    allowedMomenta[4].push_back({indices[1], indices[2], indices[0], indices[3]});
    allowedMomenta[7].push_back({indices[1], indices[2], indices[3], indices[0]});
  }
  // Sign pairs
  allowedMomenta[5].reserve(allowedMomenta[3].size());
  allowedMomenta[6].reserve(allowedMomenta[3].size());
  for (auto& indices : allowedMomenta[3]) {
    allowedMomenta[5].push_back({indices[0], indices[2], indices[1], indices[3]});
    allowedMomenta[6].push_back({indices[2], indices[0], indices[1], indices[3]});
  }

  /* External leg creator / annihilator (ket / bra) combinations */
  std::vector<std::vector<std::vector<unsigned> > > pExtBlocks;
  std::vector<std::vector<unsigned> > pExts;
  for (unsigned nPart(0); nPart != nMax; ++nPart) {
    pExtBlocks.push_back(combinationsWithReplacement(momenta.size(), nPart));
    pExts.insert(pExts.end(), pExtBlocks.back().begin(), pExtBlocks.back().end());
  }

  /* Map the external legs to pTotal sectors */
  // Temporary map to identify different sectors
  std::map<Momentum<NDIM>, unsigned> pTotalSectors;
  // Sector assignments for each external leg combination
  std::vector<unsigned> extToSector(pExts.size(), -1);
  // Reverse mapping sector -> combinations separated by number of external particles
  std::vector<std::map<unsigned, std::vector<unsigned> > > sectorToExts(pTotalSectors.size());
  for (unsigned nPart(0), iExt(0); nPart != nMax; ++nPart) {
    for (auto& pExt : pExtBlocks[nPart]) {
      Momentum<NDIM> pTotal;
      for (unsigned iMom : pExt)
        pTotal += momenta[iMom];
      auto itr(pTotalSectors.find(pTotal));
      unsigned sector;
      if (itr == pTotalSectors.end()) {
        sector = pTotalSectors.size();
        pTotalSectors[pTotal] = sector;
        sectorToExts.emplace_back();
      }
      else
        sector = itr->second;

      extToSector[iExt] = sector;
      sectorToExts[sector][nPart].push_back(iExt);
      ++iExt;
    }
  }

  /* Compute the matrix elements */
  std::map<std::pair<unsigned, unsigned>, complex128> matrixData;
  for (unsigned nKet(0), iKetStart(0); nKet != nMax; iKetStart += pExtBlocks[nKet++].size()) {
    // Consider upper triangle only
    for (unsigned nBra(0); nBra <= nKet; ++nBra) {
      // Identify possible annihilation / creation patterns for the given nKet & nBra
      std::vector<unsigned> ladderPatterns;
      for (unsigned iPatt(0); iPatt != 16; ++iPatt) {
        int nPart(nKet);
        for (unsigned iPart(0); nPart >= 0 && iPart != 4; ++iPart)
          nPart += opSign(iPatt, iPart);
        if (nPart >= 0 && nPart - nBra == 0)
          ladderPatterns.push_back(iPatt);
      }
      if (ladderPatterns.empty())
        continue;

      // Start from iKetStart = cumsum(block size for smaller nKets)
      for (unsigned iKet(iKetStart); iKet != iKetStart + pExtBlocks[nKet].size(); ++iKet) {
        std::map<unsigned, unsigned> pCountsKet;
        for (unsigned pIdx : pExts[iKet])
          ++pCountsKet[pIdx];

        // Check iBras in the same pTotal sector
        for (unsigned iBra : sectorToExts[extToSector[iKet]][nBra]) {
          if (iBra > iKet)
            break;

          // For each a/adag pattern, loop over allowed momenta
          for (unsigned iPatt : ladderPatterns) {
            std::vector<std::array<unsigned, 4> > const* allowedP(nullptr);
            if (iPatt < 8)
              allowedP = &allowedMomenta[iPatt];
            else
              allowedP = &allowedMomenta[15 - iPatt];

            for (auto& pIndices : *allowedP) {
              // Copy the pCounts
              auto pCounts(pCountsKet);
              double factor(1.);
              for (unsigned iPart(0); factor != 0. && iPart != 4; ++iPart) {
                unsigned& count(pCounts[pIndices[iPart]]);
                if (opSign(iPatt, iPart) == -1) {
                  factor *= std::sqrt(count);
                  --count;
                }
                else {
                  factor *= std::sqrt(count + 1);
                  ++count;
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
    }
  }

  /* Convert the containers to numpy arrays */
  Basis basis{};
  basis.reserve(nMax);
  basis.emplace_back(std::vector{std::size_t(1), std::size_t(0)});
  for (unsigned nPart(1); nPart != nMax; ++nPart) {
    auto& block(pExtBlocks[nPart]);
    // Passing a 2-element vector (shape) as argument to array_t constructor
    std::vector shape{std::size_t(block.size()), std::size_t(nPart)};
    basis.emplace_back(shape);
    for (unsigned iExt(0); iExt != block.size(); ++iExt)
      std::copy(block[iExt].begin(), block[iExt].end(),
                reinterpret_cast<unsigned*>(basis.back().mutable_data(iExt, 0)));
  }

  CSRData csrData{
    py::array_t<complex128>(matrixData.size()),
    py::array_t<unsigned>(matrixData.size()),
    py::array_t<unsigned>(pExts.size() + 1)
  };

  unsigned iElem(0);
  unsigned iRow(0);
  for (auto& datum : matrixData) {
    unsigned row(datum.first.first);
    unsigned col(datum.first.second);
    while (iRow <= row)
      std::get<2>(csrData).mutable_at(iRow++) = iElem;

    std::get<0>(csrData).mutable_at(iElem) = datum.second;
    std::get<1>(csrData).mutable_at(iElem) = col;
    ++iElem;
  }
  while (iRow <= pExts.size())
    std::get<2>(csrData).mutable_at(iRow++) = iElem;

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

std::pair<Basis, CSRData>
make_h1_matrix_2d(
  unsigned nParticles,
  py::array_t<int>& momenta,
  double mass
)
{
  return make_h1_matrix<2>(nParticles, momenta, mass);
}

PYBIND11_MODULE(phi4, module) {
  module.doc() = "Scalar phi4 theory";
  module.def("make_h1_matrix_1d", &make_h1_matrix_1d, "Interaction Hamiltonian");
  module.def("make_h1_matrix_2d", &make_h1_matrix_2d, "Interaction Hamiltonian");
}
