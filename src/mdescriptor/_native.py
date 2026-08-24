"""Build-time placeholder for the private native extension.

The compiled ``mdescriptor._native`` module takes precedence over this file
in wheels and editable installs.  Keeping a source placeholder gives an
unbuilt checkout a precise error instead of accidentally reviving a removed
extension name.
"""

raise ImportError(
    "MDescriptor's native module is not built; install the project with `python -m pip install -e .`."
)
