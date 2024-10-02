# pylint: disable=import-outside-toplevel
"""Project a FieldOperator to the physical (right-filled) subspace."""
import logging
from sympy import Add, Mul, S, Expr, Pow, sqrt
from sympy.physics.quantum import TensorProduct
from .particle import Projection, PresenceProjection, ParticleOuterProduct
from .symm import StepSymmetrizerBase

LOG = logging.getLogger('project_physical')


def project_physical(expr: Expr, **options) -> Expr:
    """Project a FieldOperator to the physical (right-filled) subspace."""
    LOG.debug('project_physical(%s)', expr)
    expr = expr.expand(tensorproduct=True, commutator=True)
    if isinstance(expr, Add):
        LOG.debug('Expr expanded as Add, recursing for each term')
        terms = []
        for arg in expr.args:
            term = project_physical(arg, **options)
            if term != 0:
                terms.append(term)

        LOG.debug('Returning sum of %s', terms)
        return Add(*terms).expand()

    if isinstance(expr, Mul):
        LOG.debug('Expr is Mul, evaluating factors from right')
        c_part, nc_part = expr.args_cnc()

        # Expand powers
        nc_part_new = []
        for op in nc_part:
            if isinstance(op, Pow):
                nc_part_new.extend([op.base] * op.exp)
            else:
                nc_part_new.append(op)

        mul_options = dict(options)
        ops = []
        ops_reeval = []
        first_npart = None
        # Iterate from the right-most operator
        for op in reversed(nc_part_new):
            op, npart_right, npart_left, nsymm = project_physical_op(op, **mul_options)
            LOG.debug('Returned %s', op)
            if op == 0:
                return S.Zero
            # If npart is not yet determined, place the op in re-evaluate list
            if npart_right is None:
                LOG.debug('We do not know the particle number subspace; placing %s in reevaluate'
                          ' list', op)
                ops_reeval.append(op)
                # Placeholder
                ops.append(None)
            else:
                # Once npart is not None it will not go back to None
                if first_npart is None:
                    LOG.debug('first_npart is %d', npart_right)
                    first_npart = npart_right
                ops.append(op)
            mul_options.update(npart=npart_left, nsymm=nsymm)

        if first_npart is None:
            LOG.debug('Particle number subspace was never determined')
            ops = ops_reeval
        else:
            LOG.debug('Reevaluating %d factors with first_npart=%d', len(ops_reeval), first_npart)
            mul_options = dict(options)
            mul_options['npart'] = first_npart
            # Re-evaluate the npart-undetermined ops from right to left
            for iop, op in enumerate(ops_reeval):
                ops[iop], npart_right, npart_left, nsymm = project_physical_op(op, **mul_options)
                mul_options.update(npart=npart_left, nsymm=nsymm)

        if LOG.getEffectiveLevel() == logging.DEBUG:
            LOG.debug('Returning product of %s * %s', Mul(*c_part), ops[::-1])
        return Mul(*(c_part + ops[::-1]))

    LOG.debug('Expr is single op')
    return project_physical_op(expr, **options)[0]


def project_physical_op(op: Expr, **options) -> tuple[Expr, int, int, int]:
    """Project a single Operator to the physical subspace."""
    npart = options.get('npart')
    nsymm = options.get('nsymm')
    symmetry = options.get('symmetry', 0)
    raise_on_subspace_violation = options.get('raise_on_subspace_violation', True)
    LOG.debug('project_physical_op(%s) (npart=%s nsymm=%s symmetry=%s)', op, npart, nsymm, symmetry)

    if symmetry != 0 and nsymm is None:
        nsymm = npart

    if npart is not None and nsymm is not None and nsymm > npart:
        raise ValueError(f'Unphysical nsymm {nsymm} greater than npart {npart}')

    if isinstance(op, StepSymmetrizerBase):
        LOG.debug('op is StepSymmetrizer (sign %d)', op._sign)
        if op._sign != symmetry:
            LOG.debug('Wrong symmetry sector, returning 0')
            return S.Zero, npart, npart, 0
        if (npart is not None and npart < op.args[0]) or nsymm is None or symmetry == 0:
            LOG.debug('No symmetry specified, returning op')
            return op, npart, npart, nsymm
        if op.args[0] > nsymm + 1:
            LOG.debug('Incompatible symmetrizer, returning op')
            return op, npart, npart, nsymm
        if op.args[0] == nsymm + 1:
            LOG.debug('Incrementing nsymm to %d', nsymm + 1)
            return op, npart, npart, nsymm + 1
        LOG.debug('Resolving symmetrizer to sqrt(%d)', op.args[0])
        return sqrt(op.args[0]), npart, npart, nsymm

    if not isinstance(op, TensorProduct):
        LOG.debug('op is not a TensorProduct')
        return op, npart, npart, nsymm

    if (max_particles := len(op.args)) == 0:
        LOG.debug('op is a null product')
        return S.Zero, npart, npart, nsymm

    if not all(isinstance(part_op, (Projection, ParticleOuterProduct)) for part_op in op.args):
        raise ValueError(f'Cannot resolve physical-space projection for {op}')

    occupancy_right = [(isinstance(part_op, PresenceProjection)
                        or (isinstance(part_op, ParticleOuterProduct)
                            and not part_op.bra.is_null_state))
                       for part_op in op.args]
    occupancy_left = [(isinstance(part_op, PresenceProjection)
                      or (isinstance(part_op, ParticleOuterProduct)
                          and not part_op.ket.is_null_state))
                      for part_op in op.args]
    LOG.debug('Occupancy right: %s left: %s', occupancy_right, occupancy_left)

    try:
        nocc_right = occupancy_right.index(False)
    except ValueError:
        nocc_right = max_particles
    if (True in occupancy_right[nocc_right:]
            or (npart is not None and nocc_right != npart)):
        LOG.debug('Operator projected out from right')
        return S.Zero, npart, npart, nsymm

    try:
        nocc_left = occupancy_left.index(False)
    except ValueError:
        nocc_left = max_particles
    if True in occupancy_left[nocc_left:]:
        if raise_on_subspace_violation:
            raise ValueError(f'Operator {op} violates right-filled subspace')
        LOG.debug('Operator projected out from left')
        return S.Zero, npart, npart, nsymm

    if symmetry != 0 and nsymm is None:
        nsymm = nocc_right

    if nsymm is not None:
        nsymm = min(nsymm, nocc_left)

    LOG.debug('Returning op=%s, nocc_right=%d, nocc_left=%d, nsymm=%s', op, nocc_right, nocc_left,
              nsymm)
    return op, nocc_right, nocc_left, nsymm
