# SPDX-License-Identifier: LGPL-3.0-or-later

from mdescriptor.descriptors.model_backed.dpa4._vendor.official_core.dpmodel.utils.neighbor_stat import (
    NeighborStat,
)
from mdescriptor.descriptors.model_backed.dpa4._vendor.official_core.utils.update_sel import (
    BaseUpdateSel,
)


class UpdateSel(BaseUpdateSel):
    r"""Neighbor-selection update computing :math:`n_{sel}` from statistics."""

    @property
    def neighbor_stat(self) -> type[NeighborStat]:
        return NeighborStat
