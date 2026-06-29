"""Registration progress visualization."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from confusius._utils.plotting import blend_red_cyan, make_mosaic, scale_min_max
from confusius._utils.stack import find_stack_level

if TYPE_CHECKING:
    import SimpleITK as sitk
    from matplotlib.figure import Figure


_INTERPOLATION_MAP = {
    "linear": "sitkLinear",
    "nearest": "sitkNearestNeighbor",
    "bspline": "sitkBSpline",
}


def _resolve_sitk_interpolation(interpolation: str | None) -> Any:
    """Return the SimpleITK interpolator enum for a named interpolation.

    Parameters
    ----------
    interpolation : str
        One of `"linear"`, `"nearest"`, `"bspline"`.

    Returns
    -------
    SimpleITK interpolator enum
        The matching `sitk.sitk*` interpolator constant.

    Raises
    ------
    ValueError
        If `interpolation` is not one of the supported names.
    """
    import SimpleITK as sitk

    if interpolation is None:
        interpolation = "linear"
    interp_name = _INTERPOLATION_MAP.get(interpolation)
    if interp_name is None:
        supported = ", ".join(sorted(_INTERPOLATION_MAP))
        msg = (
            f"Invalid `interpolation`: {interpolation!r}. Expected one of: {supported}."
        )
        raise ValueError(msg)
    return getattr(sitk, interp_name)


def _resample_intermediate(
    registration_method: "sitk.ImageRegistrationMethod",
    moving_img: "sitk.Image",
    fixed_img: "sitk.Image",
    resample_kwargs: dict[str, Any],
) -> "sitk.Image":
    """Resample the moving image onto the fixed grid using the current transform.

    Shared by the matplotlib and napari progress plotters so the per-iteration
    resample logic stays in one place.

    Parameters
    ----------
    registration_method : SimpleITK.ImageRegistrationMethod
        The active registration method whose initial transform is used to
        resample.
    moving_img : SimpleITK.Image
        Moving image to resample.
    fixed_img : SimpleITK.Image
        Reference image defining the output grid.
    resample_kwargs : dict[str, Any]
        Keyword arguments forwarded to `sitk.Resample`. Must contain
        `"interpolation"` and `"default_value"`. May contain
        `"sitk_threads"`.

    Returns
    -------
    SimpleITK.Image
        Resampled image on the fixed grid.
    """
    import SimpleITK as sitk

    from confusius.registration._utils import set_sitk_thread_count

    interpolation = resample_kwargs.get("interpolation", "linear")
    sitk_interp = _resolve_sitk_interpolation(interpolation)
    fill_value = resample_kwargs.get("default_value", 0.0)
    sitk_threads = resample_kwargs.get("sitk_threads", -1)

    transform = registration_method.GetInitialTransform()
    with set_sitk_thread_count(sitk_threads):
        return sitk.Resample(
            moving_img,
            fixed_img,
            transform,
            sitk_interp,
            fill_value,
            moving_img.GetPixelID(),
        )


class RegistrationProgress(Protocol):
    """Duck-typed contract for an iteration progress reporter.

    Implementations are called from the registration thread (SimpleITK's
    iteration/end callbacks). They must be safe to call from a non-GUI thread;
    any GUI side effects must be marshalled via Qt signals or similar.
    """

    def update(self) -> None:
        """Called at every optimizer iteration event."""
        ...

    def close(self) -> None:
        """Called once at the registration end event."""
        ...


class MatplotlibRegistrationProgressPlotter:
    """Plot registration progress in real time.

    Displays an optimizer metric curve, a composite fixed/moving overlay, or
    both, updated at every iteration. Works in both a Jupyter notebook and an
    interactive matplotlib backend (e.g. Qt).

    Parameters
    ----------
    registration_method : SimpleITK.ImageRegistrationMethod
        The registration method whose progress to monitor.
    fixed_img : SimpleITK.Image
        The fixed (reference) image, used to resample the composite view.
    moving_img : SimpleITK.Image
        The moving image, used to resample the composite view.
    plot_metric : bool, default: True
        Whether to display the optimizer metric over iterations.
    plot_composite : bool, default: True
        Whether to display a blended fixed/moving composite at each iteration.
        Requires an additional `sitk.Resample` call per iteration.
    resample_kwargs : dict, optional
        Extra keyword arguments forwarded to the internal resample call at each
        iteration.
    """

    def __init__(
        self,
        registration_method: "sitk.ImageRegistrationMethod",
        fixed_img: "sitk.Image",
        moving_img: "sitk.Image",
        *,
        plot_metric: bool = True,
        plot_composite: bool = True,
        resample_kwargs: dict[str, Any] | None = None,
    ) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        self._method = registration_method
        self._fixed_img = fixed_img
        self._moving_img = moving_img
        self._plot_metric = plot_metric
        self._plot_composite = plot_composite
        self._metric_values: list[float] = []

        _kw: dict[str, Any] = dict(resample_kwargs or {})
        if "default_value" not in _kw:
            import SimpleITK as sitk

            _kw["default_value"] = float(sitk.GetArrayFromImage(moving_img).min())
        self._resample_kwargs = _kw

        # Detect Jupyter notebook environment. A plain IPython terminal shell
        # also has get_ipython() != None, so we check the kernel class name to
        # distinguish: ZMQInteractiveShell means a Jupyter kernel (notebook or
        # JupyterLab); TerminalInteractiveShell means a plain ipython session,
        # which should use the same interactive-window path as a regular script.
        try:
            from IPython.core.getipython import get_ipython

            _ip = get_ipython()
            self._notebook = (
                _ip is not None and type(_ip).__name__ == "ZMQInteractiveShell"
            )
        except ImportError:
            self._notebook = False

        # Build figure layout.
        n_panels = int(plot_metric) + int(plot_composite)

        if not self._notebook:
            # Warn the user if the active backend is non-interactive. Library code
            # should not switch backends — that is the caller's responsibility.
            _non_interactive = {"agg", "pdf", "ps", "svg"}
            if matplotlib.get_backend().lower() in _non_interactive:
                warnings.warn(
                    f"The active matplotlib backend '{matplotlib.get_backend()}' is "
                    "non-interactive; the registration progress window will not be "
                    "visible. Set an interactive backend before calling "
                    "register_volume, e.g.: import matplotlib; "
                    "matplotlib.use('Qt5Agg')",
                    stacklevel=find_stack_level(),
                )

            # Enable interactive mode so the figure window opens immediately and
            # plt.pause() drives the GUI event loop on each update.
            plt.ion()

        self._fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(5 * n_panels, 4),
            squeeze=False,
            facecolor="black",
        )
        axes = axes.ravel()
        panel = iter(axes)

        if plot_metric:
            ax = next(panel)
            ax.set_facecolor("black")
            (self._metric_line,) = ax.plot([], [], color="red")
            ax.set_xlabel("Iteration", color="white")
            ax.set_ylabel("Metric value", color="white")
            ax.set_title("Registration metric", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("white")
            self._metric_ax = ax

        if plot_composite:
            ax = next(panel)
            ax.set_facecolor("black")
            # Placeholder image; real data filled on first update.
            self._composite_ax = ax
            self._composite_im = None
            ax.set_title("Fixed (red) / moving (cyan)", color="white")
            ax.axis("off")

        self._fig.tight_layout()

    def update(self) -> None:
        """Update the plot with the current iteration's data.

        Called at every `sitkIterationEvent`.
        """
        if self._plot_metric:
            self._metric_values.append(self._method.GetMetricValue())
            n = len(self._metric_values)
            self._metric_line.set_data(range(n), self._metric_values)
            self._metric_ax.relim()
            self._metric_ax.autoscale_view()

        if self._plot_composite:
            resampled = _resample_intermediate(
                self._method,
                self._moving_img,
                self._fixed_img,
                self._resample_kwargs,
            )

            import SimpleITK as sitk

            fixed_arr = np.asarray(sitk.GetArrayFromImage(self._fixed_img).T)
            moving_arr = np.asarray(sitk.GetArrayFromImage(resampled).T)

            if fixed_arr.ndim == 3:
                rgb = make_mosaic(
                    np.moveaxis(fixed_arr, 0, 0),
                    np.moveaxis(moving_arr, 0, 0),
                )
            else:
                rgb = blend_red_cyan(
                    scale_min_max(fixed_arr),
                    scale_min_max(moving_arr),
                )

            if self._composite_im is None:
                self._composite_im = self._composite_ax.imshow(
                    rgb, origin="upper", aspect="equal"
                )
            else:
                self._composite_im.set_data(rgb)

        self._render()

    def close(self) -> None:
        """Finalize the plot when registration ends.

        Called at `sitkEndEvent`.
        """
        # Trigger one last render to show the final state.
        self._render()

        if self._notebook:
            import matplotlib.pyplot as plt

            plt.close(self._fig)

    def _render(self) -> None:
        """Push pending drawing commands to the screen or notebook output."""
        if self._notebook:
            from IPython.display import display

            display(self._fig, clear=True)
        else:
            # Use canvas-level rendering to scope updates to this figure only.
            # plt.pause() drives the event loop for ALL open figures, which can
            # crash when figures from a previous registration run are still open.
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    @property
    def metric_values(self) -> list[float]:
        """Optimizer metric value recorded at each iteration.

        Returns
        -------
        list of float
            Copy of the internal metric value buffer.
        """
        return list(self._metric_values)

    @property
    def figure(self) -> "Figure":
        """The matplotlib figure used for plotting.

        Returns
        -------
        matplotlib.figure.Figure
            The figure instance owned by this monitor.
        """
        return self._fig
