"""ConfUSIus command-line interface."""


def main() -> None:
    """Launch napari with the ConfUSIus plugin open."""
    import napari

    from confusius._napari import ConfUSIusWidget

    viewer = napari.Viewer()
    viewer.window.add_dock_widget(ConfUSIusWidget(viewer), name="ConfUSIus")
    napari.run()
