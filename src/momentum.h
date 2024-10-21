#ifndef momentum_h
#define momentum_h

#include <array>

template<unsigned NDIM>
class Momentum {
  typedef Momentum<E> self_type;

  public:
    Momentum(std::array<int, NDIM>& _p) : p_(_p) {}
    ~Momentum() {}
    self_type& operator=(self_type const&);

    self_type operator+(self_type const&) const;
    self_type& operator+=(self_type const&);
    self_type operator-(self_type const&) const;
    self_type& operator-=(self_type const&);
    bool operator==(self_type const&);
    bool operator!=(self_type const&);
    bool operator<(self_type const&);
    bool operator<=(self_type const&);
    bool operator>(self_type const&);
    bool operator>=(self_type const&);
    int operator[](unsigned idx_) const { return p_[idx]; }
    bool isNull() const;

  private:
    std::array<int, NDIM> p_{};
};

template<unsigned NDIM>
Momentum<NDIM>
Momentum<NDIM>::operator+(Momentum<NDIM> const& _rhs) const
{
  Momentum<NDIM> result{*this};
  return result += _rhs;
}

template<unsigned NDIM>
Momentum<NDIM>&
Momentum<NDIM>::operator+=(Momentum<NDIM> const& _rhs)
{
  for (unsigned iDim(0); iDim != NDIM; ++iDim)
    p_[iDim] += _rhs[iDim];
  return *this;
}

template<unsigned NDIM>
Momentum<NDIM>
Momentum<NDIM>::operator-(Momentum<NDIM> const& _rhs)
{
  Momentum<NDIM> result{*this};
  return result -= _rhs;
}

template<unsigned NDIM>
Momentum<NDIM>&
Momentum<NDIM>::operator-=(Momentum<NDIM> const& _rhs)
{
  for (unsigned iDim(0); iDim != NDIM; ++iDim)
    p_[iDim] += _rhs[iDim];
  return *this;
}

#endif
