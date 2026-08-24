# Project-local DPA4 inference core

The Python files in this directory are a project-local port of the official
DPA4 array-API inference implementation. The port keeps the upstream
`LGPL-3.0-or-later` license notices and removes the upstream runtime import
boundary; model loading and graph construction are implemented directly inside
MDescriptor.

The upstream implementation is used only as the source of the port. Running
MDescriptor does not import or require the upstream training package.
