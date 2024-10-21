#ifndef momentum_h
#define momentum_h

#include <array>
#include <cmath>

template<unsigned NDIM>
class Momentum {
  typedef Momentum<E> self_type;

  public:
    Momentum() {}
    Momentum(std::array<int, NDIM>& _p) : p_(_p) {}
    Momentum(self_type const& _orig) : p_(_orig.p_) {}
    ~Momentum() {}
    self_type& operator=(self_type const& _rhs) {
      for (unsigned iDim(0); iDim != NDIM; ++iDim)
        p_[iDim] = _rhs[iDim];
    }

    self_type operator+(self_type const& _rhs) const {
      self_type result(*this);
      return result += _rhs;
    }
    self_type& operator+=(self_type const& _rhs) {
      for (unsigned iDim(0); iDim != NDIM; ++iDim)
        p_[iDim] += _rhs[iDim];
      return *this;
    }
    self_type operator-(self_type const& _rhs) const {
      self_type result(*this);
      return result -= _rhs;
    }
    self_type& operator-=(self_type const& _rhs) {
      for (unsigned iDim(0); iDim != NDIM; ++iDim)
        p_[iDim] -= _rhs[iDim];
      return *this;
    }
    bool operator==(self_type const& _rhs) {
      for (unsigned iDim(0); iDim != NDIM; ++iDim)
        if (p_[iDim] != _rhs[iDim])
          return false;
      return true;
    }
    bool operator!=(self_type const& _rhs) {
      for (unsigned iDim(0); iDim != NDIM; ++iDim)
        if (p_[iDim] != _rhs[iDim])
          return true;
      return false;
    }
    bool operator<(self_type const& _rhs) {
      for (unsigned iDim(0); iDim != NDIM; ++iDim) {
        if (p_[iDim] < _rhs[iDim])
          return true;
        if (p_[iDim] > _rhs[iDim])
          return false;
      }
      return false;
    }
    bool operator<=(self_type const& _rhs) {
      for (unsigned iDim(0); iDim != NDIM; ++iDim) {
        if (p_[iDim] > _rhs[iDim])
          return false;
        if (p_[iDim] < _rhs[iDim])
          return true;
      }
      return true;
    }
    bool operator>(self_type const& _rhs) { return !(*this <= _rhs); }
    bool operator>=(self_type const& _rhs) { return !(*this < _rhs); }
    int operator[](unsigned idx_) const { return p_[idx]; }
    bool isNull() const {
      for (unsigned iDim(0); iDim != NDIM; ++iDim) {
        if (p_[iDim] != 0)
          return false;
      }
      return true;
    }
    double energy(double mass) const {
      double e2{mass * mass};
      for (unsigned iDim(0); iDim != NDIM; ++iDim)
        e2 += p_[iDim] * p_[iDim];
      return std::sqrt(e2);
    }

  private:
    std::array<int, NDIM> p_{};
};

// Create specialization for NDIM==1

#endif
