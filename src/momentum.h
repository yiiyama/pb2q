#ifndef momentum_h
#define momentum_h

#include <array>
#include <cmath>
#include <iostream>
#include <algorithm>

template<unsigned NDIM>
class Momentum {
  typedef Momentum<NDIM> self_type;
  typedef std::array<int, NDIM> internal_type;

  public:
    Momentum() {}
    Momentum(std::array<int, NDIM>& _p) : p_(_p) {}
    Momentum(int const* _ptr) : p_{} { std::copy(_ptr, _ptr + NDIM, p_.begin()); }
    Momentum(self_type const& _orig) : p_(_orig.p_) {}
    ~Momentum() {}
    self_type& operator=(self_type const& _rhs) {
      for (unsigned iDim(0); iDim != NDIM; ++iDim)
        p_[iDim] = _rhs[iDim];
      return *this;
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
    bool operator==(self_type const& _rhs) const {
      for (unsigned iDim(0); iDim != NDIM; ++iDim)
        if (p_[iDim] != _rhs[iDim])
          return false;
      return true;
    }
    bool operator!=(self_type const& _rhs) const {
      for (unsigned iDim(0); iDim != NDIM; ++iDim)
        if (p_[iDim] != _rhs[iDim])
          return true;
      return false;
    }
    bool operator<(self_type const& _rhs) const {
      for (unsigned iDim(0); iDim != NDIM; ++iDim) {
        if (p_[iDim] < _rhs[iDim])
          return true;
        if (p_[iDim] > _rhs[iDim])
          return false;
      }
      return false;
    }
    bool operator<=(self_type const& _rhs) const {
      for (unsigned iDim(0); iDim != NDIM; ++iDim) {
        if (p_[iDim] > _rhs[iDim])
          return false;
        if (p_[iDim] < _rhs[iDim])
          return true;
      }
      return true;
    }
    bool operator>(self_type const& _rhs) const { return !(*this <= _rhs); }
    bool operator>=(self_type const& _rhs) const { return !(*this < _rhs); }
    int operator[](unsigned _idx) const { return p_[_idx]; }
    bool isNull() const {
      for (unsigned iDim(0); iDim != NDIM; ++iDim) {
        if (p_[iDim] != 0)
          return false;
      }
      return true;
    }
    double energy(double _mass) const {
      double e2{_mass * _mass};
      for (unsigned iDim(0); iDim != NDIM; ++iDim)
        e2 += p_[iDim] * p_[iDim];
      return std::sqrt(e2);
    }
    internal_type const& data() const { return p_; }

  private:
    internal_type p_{};
};

template<>
class Momentum<1> {
  typedef Momentum<1> self_type;
  typedef int internal_type;

  public:
    Momentum() {}
    Momentum(int _p) : p_(_p) {}
    Momentum(int const* _ptr) : p_(*_ptr) {}
    Momentum(self_type const& _orig) : p_(_orig.p_) {}
    ~Momentum() {}
    self_type& operator=(self_type const& _rhs) {
      p_ = _rhs.p_;
      return *this;
    }

    self_type operator+(self_type const& _rhs) const { return self_type(p_ + _rhs.p_); }
    self_type& operator+=(self_type const& _rhs) {
      p_ += _rhs.p_;
      return *this;
    }
    self_type operator-(self_type const& _rhs) const { return self_type(p_ - _rhs.p_); }
    self_type& operator-=(self_type const& _rhs) {
      p_ -= _rhs.p_;
      return *this;
    }
    bool operator==(self_type const& _rhs) const { return p_ == _rhs.p_; }
    bool operator!=(self_type const& _rhs) const { return p_ != _rhs.p_; }
    bool operator<(self_type const& _rhs) const { return p_ < _rhs.p_; }
    bool operator<=(self_type const& _rhs) const { return p_ <= _rhs.p_; }
    bool operator>(self_type const& _rhs) const { return p_ > _rhs.p_; }
    bool operator>=(self_type const& _rhs) const { return p_ >= _rhs.p_; }
    int operator[](unsigned idx_) const { return p_; }
    bool isNull() const { return p_ == 0; }
    double energy(double mass) const { return std::sqrt(mass * mass + p_ * p_ ); }
    internal_type data() const { return p_; }

  private:
    int p_{};
};

template<unsigned NDIM>
std::ostream& operator<<(std::ostream& os, Momentum<NDIM> const& momentum)
{
  os << "[";
  for (unsigned iDim(0); iDim != NDIM - 1; ++iDim)
    os << momentum[iDim] << ",";
  os << momentum[NDIM - 1] << "]";
  return os;
}

#endif
