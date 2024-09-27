# pylint: disable=invalid-name
"""Reimplementation of qapply for ProductOperators."""
import logging
from sympy import Number
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sympify import sympify

from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.operator import OuterProduct, Operator
from sympy.physics.quantum.state import State, KetBase, BraBase
from sympy.physics.quantum.tensorproduct import TensorProduct

LOG = logging.getLogger('apply_op')


def apply_op(e, **options):
    """Apply product operators to states.

    See the docstring of qapply for details.
    """
    rec_depth = options.get('rec_depth', 0)
    options['rec_depth'] = rec_depth + 1

    LOG.debug('%d: apply_op(%s)', rec_depth, e)
    dagger = options.get('dagger', False)

    if e == 0:
        return S.Zero

    # This may be a bit aggressive but ensures that everything gets expanded
    # to its simplest form before trying to apply operators. This includes
    # things like (A+B+C)*|a> and A*(|a>+|b>) and all Commutators and
    # TensorProducts. The only problem with this is that if we can't apply
    # all the Operators, we have just expanded everything.
    # TODO: don't expand the scalars in front of each Mul.
    e = e.expand(commutator=True, tensorproduct=True)

    # If we just have a raw ket, return it.
    if isinstance(e, KetBase):
        return e

    # We have an Add(a, b, c, ...) and compute
    # Add(qapply(a), qapply(b), ...)
    if isinstance(e, Add):
        LOG.debug('%d: %s is Add', rec_depth, e)
        terms = []
        for arg in e.args:
            term = apply_op(arg, **options)
            LOG.debug('%d: Got term %s', rec_depth, term)
            terms.append(term)
        return Add(*terms).expand()

    # For a raw TensorProduct, call qapply on its args.
    if isinstance(e, TensorProduct):
        return e.func(*[apply_op(t, **options) for t in e.args])

    # For a Pow, call qapply on its base.
    if isinstance(e, Pow):
        return apply_op(e.base, **options) ** e.exp

    # We have a Mul where there might be actual operators to apply to kets.
    while isinstance(e, Mul):
        c_part, nc_part = e.args_cnc()
        c_mul = Mul(*c_part)
        nc_mul = Mul(*nc_part)
        LOG.debug('%d: %s is Mul, nc_mul=%s', rec_depth, e, nc_mul)
        if isinstance(nc_mul, Mul):
            result = c_mul * apply_op_Mul(nc_mul, **dict(options))
        else:
            result = c_mul * apply_op(nc_mul, **options)
        if result == e:
            if dagger:
                return Dagger(apply_op_Mul(Dagger(e), **dict(options)))
            break
        LOG.debug('%d: Updating e to %s', rec_depth, result)
        e = result

    # In all other cases (State, Operator, Pow, Commutator, InnerProduct,
    # OuterProduct) we won't ever have operators to apply to kets.
    return e


def apply_op_Mul(e, **options):
    rec_depth = options.get('rec_depth', 0)
    rec_depth_mul = options.get('rec_depth_mul', 0)
    options['rec_depth_mul'] = rec_depth_mul + 1
    LOG.debug('%d-%d: apply_op_Mul(%s)', rec_depth, rec_depth_mul, e)
    ip_doit = options.get('ip_doit', True)

    args = list(e.args)

    # If we only have 0 or 1 args, we have nothing to do and return.
    if len(args) <= 1 or not isinstance(e, Mul):
        return e

    rhs = args.pop()
    lhs = args.pop()

    # Make sure we have two non-commutative objects before proceeding.
    if sympify(rhs).is_commutative or sympify(lhs).is_commutative:
        return e

    # For a Pow with an integer exponent, apply one of them and reduce the
    # exponent by one.
    if isinstance(lhs, Pow) and lhs.exp.is_Integer:
        args.append(lhs.base ** (lhs.exp - 1))
        lhs = lhs.base

    # Pull OuterProduct apart
    if isinstance(lhs, OuterProduct):
        args.append(lhs.ket)
        lhs = lhs.bra

    # Call .doit() on Commutator/AntiCommutator.
    if isinstance(lhs, (Commutator, AntiCommutator)):
        comm = lhs.doit()
        if isinstance(comm, Add):
            return apply_op(
                e.func(*(args + [comm.args[0], rhs])) +
                e.func(*(args + [comm.args[1], rhs])),
                **options
            )
        return apply_op(e.func(*args) * comm * rhs, **options)

    # Apply tensor products of operators to states
    if (isinstance(lhs, TensorProduct)
            and all(isinstance(arg, (Operator, State, Mul, Pow)) or arg == 1 for arg in lhs.args)
            and isinstance(rhs, TensorProduct)
            and all(isinstance(arg, (Operator, State, Mul, Pow)) or arg == 1 for arg in rhs.args)
            and len(lhs.args) == len(rhs.args)):
        LOG.debug('%d-%d: Found tensor product, lhs=%s, rhs=%s', rec_depth, rec_depth_mul, lhs.args,
                  rhs.args)
        results = []
        for lhs_arg, rhs_arg in zip(lhs.args, rhs.args):
            res = apply_op(Mul(lhs_arg, rhs_arg), **options)
            if res == 0:
                LOG.debug('%d-%d: Null product of %s', rec_depth, rec_depth_mul, results)
                return S.Zero
            results.append(res)

        if all(isinstance(res, Number) for res in results):
            result = Mul(*results)
        else:
            tps = TensorProduct(*results).expand(tensorproduct=True)
            if isinstance(tps, Add):
                result = S.Zero
                for term in tps.args:
                    c_part, nc_part = term.args_cnc()
                    result += Mul(*c_part) * rhs.func(*nc_part)
            else:
                result = rhs.func(*results)
        LOG.debug('%d-%d: Factoring out TP result %s from remaining %s', rec_depth, rec_depth_mul,
                  result, args)

        return apply_op(e.func(*args) * result, **options)

    # Now try to actually apply the operator and build an inner product.
    try:
        result = lhs._apply_operator(rhs, **options)
        LOG.debug('%d-%d: Applied %s to %s -> %s', rec_depth, rec_depth_mul, lhs, rhs, result)
    except (NotImplementedError, AttributeError):
        try:
            result = rhs._apply_from_right_to(lhs, **options)
            LOG.debug('%d-%d: Right-applied %s to %s -> %s', rec_depth, rec_depth_mul, rhs, lhs,
                      result)
        except (NotImplementedError, AttributeError):
            if isinstance(lhs, BraBase) and isinstance(rhs, KetBase):
                result = InnerProduct(lhs, rhs)
                if ip_doit:
                    result = result.doit()
                LOG.debug('%d-%d: Innerproduct(%s, %s) = %s', rec_depth, rec_depth_mul, lhs, rhs,
                          result)
            else:
                result = None
                LOG.debug('%d-%d: No action between %s and %s', rec_depth, rec_depth_mul, lhs, rhs)

    # TODO: I may need to expand before returning the final result.
    if result == 0:
        return S.Zero
    if result is None:
        if len(args) == 0:
            # We had two args to begin with so args=[].
            LOG.debug('%d-%d: Returning the original expression %s', rec_depth, rec_depth_mul, e)
            return e
        LOG.debug('%d-%d: Factoring out rhs %s', rec_depth, rec_depth_mul, rhs)
        return apply_op_Mul(e.func(*(args + [lhs])), **options) * rhs
    if isinstance(result, InnerProduct):
        return result * apply_op_Mul(e.func(*args), **options)
    # result is a scalar times a Mul, Add or TensorProduct
    LOG.debug('%d-%d: Factoring out result %s from remaining %s', rec_depth, rec_depth_mul, result,
              args)
    return apply_op(e.func(*args) * result, **options)
