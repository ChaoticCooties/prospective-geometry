"""Prospective geometry study package.

Every scientific entrypoint in this package must call
:func:`pgeom.firewall.guard_path` on each input path it resolves. The study is a
clean-room rebuild; the archived Renegade experiments are read-only scientific
history and must never become study data.
"""

__version__ = "0.1.0"
