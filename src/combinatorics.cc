#include "combinatorics.h"
#include <stdexcept>
#include <numeric>

unsigned long long
comb(unsigned nAll, unsigned nSubset)
{
  if (nSubset > nAll / 2)
    nSubset = nAll - nSubset;

  unsigned long long result{1};
  for (unsigned ival(1); ival != nSubset + 1; ++ival)
    result *= nAll - ival + 1;
  for (unsigned ival(1); ival != nSubset + 1; ++ival)
    result /= ival;
  return result;
}

std::vector<std::vector<unsigned> >
combinationsWithReplacement(unsigned poolSize, unsigned combSize)
{
  if (combSize == 0)
    return std::vector<std::vector<unsigned> >(1);

  std::vector<std::vector<unsigned> > combinations;
  combinations.reserve(comb(poolSize - 1 + combSize, combSize));

  std::vector<unsigned> combination(combSize, 0);
  combinations.push_back(combination);
  while (combination[0] != poolSize - 1) {
    auto rItr{combination.rbegin()};
    for (; rItr != combination.rend(); ++rItr) {
      if (*rItr != poolSize - 1)
        break;
    }
    unsigned index{*rItr};
    while (true) {
      *rItr = index + 1;
      if (rItr == combination.rbegin())
        break;
      --rItr;
    }
    combinations.push_back(combination);
  }

  return combinations;
}
