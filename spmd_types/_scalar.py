"""
Scalar wrapper with local SPMD type annotation.

Allows users to annotate Python scalars with an exact SPMD type, so that the
type checker uses the declared type instead of the weak _Scalar sentinel.

Example::

    from torch.distributed.device_mesh import init_device_mesh
    from spmd_types import Scalar, V, P, R

    mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("tp",))
    tp = mesh.get_group("tp")

    # A rank-dependent scalar is Varying:
    typed_rank = Scalar(rank / world_size, {tp: V})

    # A partial sum across ranks:
    typed_sum = Scalar(local_sum, {tp: P})

    # Explicitly replicate (same as untyped, but opt-in):
    typed_const = Scalar(2.0, {tp: R})
"""

from __future__ import annotations

from spmd_types.types import (
    LocalSpmdType,
    normalize_local_type,
    PerMeshAxisLocalSpmdType,
    R,
    V,
)


def _is_numeric_scalar(val: object) -> bool:
    """Return True if val is a numeric scalar (int/float/complex, not bool)."""
    return isinstance(val, (int, float, complex)) and not isinstance(val, bool)


def _unwrap_scalar(val: object):
    """If val is a Scalar, return its raw value; otherwise return val unchanged."""
    if isinstance(val, Scalar):
        return val.value
    return val


def _unwrap_args(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Replace each Scalar in args/kwargs with its raw value."""
    new_args = tuple(_unwrap_scalar(a) for a in args)
    new_kwargs = {k: _unwrap_scalar(v) for k, v in kwargs.items()}
    return new_args, new_kwargs


class Scalar:
    """A Python scalar annotated with a local SPMD type.

    Wraps a numeric value (int, float, or complex -- not bool) together with
    an explicit SPMD type dict so that the type checker treats it as a typed
    operand rather than the weak ``_Scalar`` sentinel.

    Args:
        value: A numeric Python scalar.
        spmd_type: A ``LocalSpmdType`` dict mapping mesh axes to per-axis SPMD
            types (R, I, V, or P).
    """

    def __init__(self, value: int | float | complex, spmd_type: LocalSpmdType):
        if not _is_numeric_scalar(value):
            raise TypeError(
                f"Scalar value must be int, float, or complex (not bool), "
                f"got {type(value).__name__}: {value!r}"
            )
        self._local_type: LocalSpmdType = normalize_local_type(spmd_type)
        self._value = value

    @property
    def value(self) -> int | float | complex:
        """The raw numeric value."""
        return self._value

    @property
    def local_type(self) -> LocalSpmdType:
        """The SPMD type annotation."""
        return dict(self._local_type)

    def __repr__(self) -> str:
        return f"Scalar({self._value!r}, {self._local_type!r})"

    # -- Python arithmetic dunders ----------------------------------------
    #
    # These let Scalar participate in plain Python arithmetic (e.g.
    # ``Scalar(3, t) + 2``, ``Scalar(3, t) * Scalar(4, t)``).
    #
    # Per-axis type merge rules (matching the tensor type system):
    #   - Same types: result is that type
    #   - R + V -> V
    #   - I only mixes with I
    #   - Everything else: NotImplemented (error)
    #
    # Plain numeric scalars (not wrapped in Scalar) are weak -- they adopt
    # the other Scalar's type.  Scalar(..., I) is strict and only mixes
    # with other I Scalars.
    #
    # Interactions with torch.Tensor still go through __torch_function__.

    @staticmethod
    def _merge_axis_type(
        a: PerMeshAxisLocalSpmdType, b: PerMeshAxisLocalSpmdType
    ) -> PerMeshAxisLocalSpmdType | None:
        """Merge two per-axis types.  Returns None on incompatible types."""
        if a is b:
            return a
        # R + V -> V (and V + R -> V).
        rv = {a, b}
        if rv == {R, V}:
            return V
        return None

    def _merge_types(self, other_type: LocalSpmdType) -> LocalSpmdType | None:
        """Merge two LocalSpmdType dicts.  Returns None on incompatible types."""
        # Both must cover the same set of axes.
        if self._local_type.keys() != other_type.keys():
            return None
        result: LocalSpmdType = {}
        for axis in self._local_type:
            merged = self._merge_axis_type(self._local_type[axis], other_type[axis])
            if merged is None:
                return None
            result[axis] = merged
        return result

    def _binop(self, other: object, op):
        from spmd_types._state import is_type_checking

        if isinstance(other, Scalar):
            if not is_type_checking():
                return op(self._value, other._value)
            merged = self._merge_types(other._local_type)
            if merged is None:
                return NotImplemented
            return Scalar(op(self._value, other._value), merged)
        if _is_numeric_scalar(other):
            if not is_type_checking():
                return op(self._value, other)
            return Scalar(op(self._value, other), self._local_type)
        return NotImplemented

    def _rbinop(self, other: object, op):
        from spmd_types._state import is_type_checking

        if _is_numeric_scalar(other):
            if not is_type_checking():
                return op(other, self._value)
            return Scalar(op(other, self._value), self._local_type)
        return NotImplemented

    def __add__(self, other: object) -> Scalar:
        return self._binop(other, lambda a, b: a + b)

    def __radd__(self, other: object) -> Scalar:
        return self._rbinop(other, lambda a, b: a + b)

    def __sub__(self, other: object) -> Scalar:
        return self._binop(other, lambda a, b: a - b)

    def __rsub__(self, other: object) -> Scalar:
        return self._rbinop(other, lambda a, b: a - b)

    def __mul__(self, other: object) -> Scalar:
        return self._binop(other, lambda a, b: a * b)

    def __rmul__(self, other: object) -> Scalar:
        return self._rbinop(other, lambda a, b: a * b)

    def __truediv__(self, other: object) -> Scalar:
        return self._binop(other, lambda a, b: a / b)

    def __rtruediv__(self, other: object) -> Scalar:
        return self._rbinop(other, lambda a, b: a / b)

    def __floordiv__(self, other: object) -> Scalar:
        return self._binop(other, lambda a, b: a // b)

    def __rfloordiv__(self, other: object) -> Scalar:
        return self._rbinop(other, lambda a, b: a // b)

    def __mod__(self, other: object) -> Scalar:
        return self._binop(other, lambda a, b: a % b)

    def __rmod__(self, other: object) -> Scalar:
        return self._rbinop(other, lambda a, b: a % b)

    def __pow__(self, other: object) -> Scalar:
        return self._binop(other, lambda a, b: a**b)

    def __rpow__(self, other: object) -> Scalar:
        return self._rbinop(other, lambda a, b: a**b)

    def __neg__(self):
        from spmd_types._state import is_type_checking

        if not is_type_checking():
            return -self._value
        return Scalar(-self._value, self._local_type)

    def __pos__(self):
        from spmd_types._state import is_type_checking

        if not is_type_checking():
            return +self._value
        return Scalar(+self._value, self._local_type)

    def __abs__(self):
        from spmd_types._state import is_type_checking

        if not is_type_checking():
            return abs(self._value)
        return Scalar(abs(self._value), self._local_type)

    # Comparisons return plain bool (not Scalar) -- the SPMD type is
    # irrelevant for control-flow decisions.

    def __eq__(self, other: object) -> bool:
        other_val = other._value if isinstance(other, Scalar) else other
        return self._value == other_val

    def __ne__(self, other: object) -> bool:
        other_val = other._value if isinstance(other, Scalar) else other
        return self._value != other_val

    def __lt__(self, other: object) -> bool:
        other_val = other._value if isinstance(other, Scalar) else other
        return self._value < other_val

    def __le__(self, other: object) -> bool:
        other_val = other._value if isinstance(other, Scalar) else other
        return self._value <= other_val

    def __gt__(self, other: object) -> bool:
        other_val = other._value if isinstance(other, Scalar) else other
        return self._value > other_val

    def __ge__(self, other: object) -> bool:
        other_val = other._value if isinstance(other, Scalar) else other
        return self._value >= other_val

    # int/float/bool coercions for use in contexts that need a plain number.

    def __int__(self) -> int:
        return int(self._value)

    def __float__(self) -> float:
        return float(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __hash__(self) -> int:
        return hash(self._value)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Import from _state to avoid circular dependency (_checker imports _scalar).
        from spmd_types._state import is_type_checking

        if is_type_checking():
            # SpmdTypeMode is active -- let the mode handle unwrapping
            return NotImplemented
        # No type checking active: unwrap Scalars to raw values and call func
        new_args, new_kwargs = _unwrap_args(args, kwargs)
        return func(*new_args, **new_kwargs)
