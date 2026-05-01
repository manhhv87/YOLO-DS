"""Entry point for the AVI inspection UI.

Run with::

    cd ui
    python main.py

The window_camera.py module already initialises the Qt application and
opens the main window when invoked as ``__main__``; we delegate to it so
that ``main.py`` can be the single, documented entry point of the UI app.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``from core.x import y`` work regardless of where the script was launched.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main() -> None:
    from PyQt5 import QtWidgets

    from window_camera import AOIWindow

    app = QtWidgets.QApplication(sys.argv)
    win = AOIWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
