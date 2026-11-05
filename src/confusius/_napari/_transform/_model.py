"""Pure state of the transform panel: the user's world-space transform and its history."""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from confusius._dims import WORLD_DIMS
from confusius.registration.affines import (
    affine_to_sitk_linear_transform,
    decompose_affine,
)

if TYPE_CHECKING:
    import SimpleITK as sitk

Frame: TypeAlias = Literal["global", "local"]
"""Axes a step acts along: fixed world axes, or axes that rotate with the volume."""

WorldAxis: TypeAlias = Literal["z", "y", "x"]
"""World axis name."""

WORLD_AXES: tuple[WorldAxis, WorldAxis, WorldAxis] = ("z", "y", "x")
"""World axis names, typed for the methods below, in `WORLD_DIMS` order."""

SHEAR_TOLERANCE = 1e-8
"""Largest shear `from_matrix` still treats as none, to absorb round-off."""


def wrap_angle(angle: float) -> float:
    """Return an angle wrapped into `[-180, 180)` degrees.

    Parameters
    ----------
    angle : float
        Angle in degrees.

    Returns
    -------
    float
        The same rotation, within half a turn either way.
    """
    return float((angle + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True, eq=False)
class TransformState:
    """World-space transform built from translation, rotation, scale and center.

    The transform is `T(x) = t + c + R @ S @ (x - c)`, in world `(z, y, x)` order. It
    leaves `T(c) = c + t` fixed, so `pivot` rather than `center` is the world point
    rotations turn about, and the one the panel displays.

    Every method returns a new state, which is what makes an undo history trivial.

    Attributes
    ----------
    translation : (3,) numpy.ndarray
        Translation `t`.
    rotation : (3, 3) numpy.ndarray
        Rotation `R`.
    scale : (3,) numpy.ndarray
        Signed per-axis scale `S`, along the volume's axes.
    center : (3,) numpy.ndarray
        Center `c`, the pre-image of `pivot`.
    affine : (4, 4) numpy.ndarray, optional
        Matrix carrying shear, which `t`, `R` and `S` cannot represent. If provided, it
        is the transform and the other fields do not contribute to it.
    turned : (3,) numpy.ndarray
        Angle turned about each world axis by rotation steps since the matrix was last
        set directly, in degrees within `[-180, 180)`. It is what the dials show, and
        a tally of steps rather than a decomposition of `rotation`.
    """

    translation: npt.NDArray[np.float64]
    rotation: npt.NDArray[np.float64]
    scale: npt.NDArray[np.float64]
    center: npt.NDArray[np.float64]
    affine: npt.NDArray[np.float64] | None = None
    turned: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    @classmethod
    def identity(cls, center: npt.ArrayLike = (0.0, 0.0, 0.0)) -> "TransformState":
        """Return the identity transform.

        Parameters
        ----------
        center : (3,) array-like, default: (0.0, 0.0, 0.0)
            Center of rotation.

        Returns
        -------
        TransformState
            The identity transform.
        """
        return cls(
            translation=np.zeros(3),
            rotation=np.eye(3),
            scale=np.ones(3),
            center=np.asarray(center, dtype=np.float64),
        )

    @classmethod
    def from_matrix(
        cls, matrix: npt.ArrayLike, center: npt.ArrayLike
    ) -> "TransformState":
        """Return the state for a homogeneous matrix.

        Parameters
        ----------
        matrix : (4, 4) array-like
            Homogeneous world-space matrix.
        center : (3,) array-like
            Center of rotation to express the state about.

        Returns
        -------
        TransformState
            State reproducing `matrix`. A matrix carrying shear is kept whole in
            `affine`, see `is_sheared`.
        """
        homogeneous = np.asarray(matrix, dtype=np.float64)
        center_array = np.asarray(center, dtype=np.float64)
        _, rotation, scale, shear = decompose_affine(homogeneous)
        if not np.allclose(shear, 0.0, atol=SHEAR_TOLERANCE):
            return replace(cls.identity(center_array), affine=homogeneous)
        mapped_center = homogeneous[:3, :3] @ center_array + homogeneous[:3, 3]
        return cls(
            translation=mapped_center - center_array,
            rotation=np.asarray(rotation, dtype=np.float64),
            scale=np.asarray(scale, dtype=np.float64),
            center=center_array,
        )

    @property
    def is_sheared(self) -> bool:
        """Return whether the transform carries shear.

        Returns
        -------
        bool
            Whether the transform carries shear, so it is only available as a matrix.
        """
        return self.affine is not None

    def matrix(self) -> npt.NDArray[np.float64]:
        """Return the transform as a homogeneous matrix.

        Returns
        -------
        (4, 4) numpy.ndarray
            Homogeneous world-space matrix.
        """
        if self.affine is not None:
            return self.affine
        linear = self.rotation @ np.diag(self.scale)
        matrix = np.eye(4)
        matrix[:3, :3] = linear
        matrix[:3, 3] = self.translation + self.center - linear @ self.center
        return matrix

    def is_identity(self) -> bool:
        """Return whether the transform leaves every point in place.

        Returns
        -------
        bool
            Whether the matrix is the identity, up to floating-point noise.
        """
        return bool(np.allclose(self.matrix(), np.eye(4)))

    def rotation_axis(
        self, axis: WorldAxis, *, frame: Frame
    ) -> npt.NDArray[np.float64]:
        """Return the world direction a rotation row turns about.

        Parameters
        ----------
        axis : {"z", "y", "x"}
            Rotation row.
        frame : {"global", "local"}
            Axes the row acts along.

        Returns
        -------
        (3,) numpy.ndarray
            Unit vector in world `(z, y, x)` order.
        """
        unit = np.eye(3)[WORLD_DIMS.index(axis)]
        return self.rotation @ unit if frame == "local" else unit

    def rotated(
        self, axis: WorldAxis, angle: float, *, frame: Frame
    ) -> "TransformState":
        """Return this state after a rotation step about the pivot.

        Parameters
        ----------
        axis : {"z", "y", "x"}
            Rotation row.
        angle : float
            Step in degrees, following the right-hand rule in ordered world `(z, y, x)`
            as [`apply_rotation`][confusius.registration.apply_rotation] does.
        frame : {"global", "local"}
            Axes the row acts along.

        Returns
        -------
        TransformState
            The rotated state.
        """
        index = WORLD_DIMS.index(axis)
        unit = np.eye(3)[index]
        step = Rotation.from_rotvec(np.deg2rad(angle) * unit).as_matrix()
        rotation = self.rotation @ step if frame == "local" else step @ self.rotation
        turned = self.turned.copy()
        turned[index] = wrap_angle(turned[index] + angle)
        return replace(self, rotation=rotation, turned=turned)

    def translated(self, delta: npt.ArrayLike, *, frame: Frame) -> "TransformState":
        """Return this state after a translation step.

        Parameters
        ----------
        delta : (3,) array-like
            Step along the frame's `(z, y, x)` axes, in data units.
        frame : {"global", "local"}
            Axes the step acts along.

        Returns
        -------
        TransformState
            The translated state.
        """
        step = np.asarray(delta, dtype=np.float64)
        if frame == "local":
            step = self.rotation @ step
        return replace(self, translation=self.translation + step)

    def zoomed(self, factors: npt.ArrayLike) -> "TransformState":
        """Return this state with new zoom factors, keeping any flip.

        Parameters
        ----------
        factors : (3,) array-like
            Zoom magnitude along the volume's `(z, y, x)` axes.

        Returns
        -------
        TransformState
            The zoomed state.

        Raises
        ------
        ValueError
            If a factor is zero or not finite.
        """
        magnitudes = np.abs(np.asarray(factors, dtype=np.float64))
        if not np.all(np.isfinite(magnitudes)) or np.any(magnitudes == 0):
            raise ValueError(
                f"zoom factors must be finite and non-zero, got {factors}."
            )
        return replace(self, scale=magnitudes * np.sign(self.scale))

    def flipped(self, axis: WorldAxis, *, frame: Frame) -> "TransformState":
        """Return this state after mirroring through the pivot.

        Parameters
        ----------
        axis : {"z", "y", "x"}
            Axis to mirror along.
        frame : {"global", "local"}
            Axes `axis` belongs to.

        Returns
        -------
        TransformState
            The mirrored state.
        """
        mirror = np.ones(3)
        mirror[WORLD_DIMS.index(axis)] = -1.0
        if frame == "local":
            return replace(self, scale=self.scale * mirror)
        # A world-axis mirror M on the left of R @ S is the same as M @ R @ M @ (M @ S),
        # where the first factor is a proper rotation and the second is a diagonal.
        flip = np.diag(mirror)
        return replace(
            self, rotation=flip @ self.rotation @ flip, scale=self.scale * mirror
        )

    def pivot(self) -> npt.NDArray[np.float64]:
        """Return the world point rotations leave fixed.

        Returns
        -------
        (3,) numpy.ndarray
            World point in `(z, y, x)` order.
        """
        matrix = self.matrix()
        return matrix[:3, :3] @ self.center + matrix[:3, 3]

    def with_pivot(self, point: npt.ArrayLike) -> "TransformState":
        """Return the same transform, expressed so that `point` is its pivot.

        Parameters
        ----------
        point : (3,) array-like
            World point in `(z, y, x)` order.

        Returns
        -------
        TransformState
            State whose `matrix` is unchanged and whose `pivot` is `point`.
        """
        target = np.asarray(point, dtype=np.float64)
        matrix = self.matrix()
        center = np.linalg.solve(matrix[:3, :3], target - matrix[:3, 3])
        return replace(self, center=center, translation=target - center)


class History:
    """Undo history of transform states.

    Parameters
    ----------
    initial : TransformState
        State to undo back to.
    """

    def __init__(self, initial: TransformState) -> None:
        self._states = [initial]
        self._cursor = 0

    @property
    def can_undo(self) -> bool:
        """Return whether there is a state to step back to.

        Returns
        -------
        bool
            Whether `undo` would return a state.
        """
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        """Return whether there is a state to step forward to.

        Returns
        -------
        bool
            Whether `redo` would return a state.
        """
        return self._cursor < len(self._states) - 1

    def push(self, state: TransformState) -> None:
        """Record `state` as the latest one, dropping anything undone.

        Parameters
        ----------
        state : TransformState
            State to record.
        """
        del self._states[self._cursor + 1 :]
        self._states.append(state)
        self._cursor += 1

    def undo(self) -> TransformState | None:
        """Step back one state.

        Returns
        -------
        TransformState or None
            Previous state, or `None` if there is none.
        """
        if self._cursor == 0:
            return None
        self._cursor -= 1
        return self._states[self._cursor]

    def redo(self) -> TransformState | None:
        """Step forward one state.

        Returns
        -------
        TransformState or None
            Next state, or `None` if there is none.
        """
        if self._cursor == len(self._states) - 1:
            return None
        self._cursor += 1
        return self._states[self._cursor]

    def clear(self, initial: TransformState) -> None:
        """Forget every state and restart from `initial`.

        Parameters
        ----------
        initial : TransformState
            State to undo back to from now on.
        """
        self._states = [initial]
        self._cursor = 0


def format_matrix(matrix: npt.ArrayLike) -> str:
    """Write a matrix as plain text, without losing precision.

    Parameters
    ----------
    matrix : (4, 4) array-like
        Matrix to write.

    Returns
    -------
    str
        One line per row, with space-separated values.
    """
    rows = np.asarray(matrix, dtype=np.float64)
    return "\n".join(" ".join(repr(float(value)) for value in row) for row in rows)


def parse_matrix(text: str) -> npt.NDArray[np.float64]:
    """Read a homogeneous matrix from plain text.

    Parameters
    ----------
    text : str
        Sixteen values, separated by whitespace or commas. Square brackets are ignored,
        so a printed NumPy array parses too.

    Returns
    -------
    (4, 4) numpy.ndarray
        The matrix.

    Raises
    ------
    ValueError
        If `text` does not hold an invertible homogeneous matrix of finite values.
    """
    tokens = text.replace(",", " ").replace("[", " ").replace("]", " ").split()
    try:
        values = np.array([float(token) for token in tokens], dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"Could not read a matrix from {text[:40]!r}.") from error
    if values.size != 16:
        raise ValueError(f"A 4×4 matrix needs 16 values, got {values.size}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("A matrix can only hold finite values.")
    matrix = values.reshape(4, 4)
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("The last row of a homogeneous matrix must be 0 0 0 1.")
    if np.isclose(np.linalg.det(matrix), 0.0):
        raise ValueError("The matrix is singular, so it cannot place a volume.")
    return matrix


def to_itk_transform(matrix: npt.ArrayLike) -> "sitk.AffineTransform":
    """Convert a world `(z, y, x)` matrix to what ITK transform files expect.

    ITK files, as 3D Slicer reads them, hold the pull transform (from output points back
    to input points) in LPS `(x, y, z)` coordinates. ConfUSIus's `(x, y, z)` is taken as
    RAS, so the first two axes flip sign.

    Parameters
    ----------
    matrix : (4, 4) array-like
        Homogeneous world-space matrix, pushing points in `(z, y, x)` order.

    Returns
    -------
    SimpleITK.AffineTransform
        Equivalent ITK transform.
    """
    lps_axes = np.zeros((4, 4))
    lps_axes[3, 3] = 1.0
    lps_axes[:3, :3] = np.diag([-1.0, -1.0, 1.0])[:, ::-1]
    push = lps_axes @ np.asarray(matrix, dtype=np.float64) @ np.linalg.inv(lps_axes)
    return affine_to_sitk_linear_transform(np.linalg.inv(push))
