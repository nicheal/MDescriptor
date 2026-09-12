"""Element symbol lookup used by model-backed descriptors."""

from __future__ import annotations

from typing import Any

_ELEMENT_SYMBOLS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
_ATOMIC_SYMBOLS = {index + 1: symbol for index, symbol in enumerate(_ELEMENT_SYMBOLS)}


def _symbols_to_atype(type_mapper: Any, numbers: Any) -> Any:
    """Map atomic numbers through a checkpoint type map with stable errors."""

    symbols: list[str] = []
    for number in numbers.tolist():
        try:
            symbols.append(_ATOMIC_SYMBOLS[int(number)])
        except KeyError as exc:
            raise ValueError(
                f"atomic number {number} is absent from the checkpoint type_map"
            ) from exc
    try:
        return type_mapper.symbols_to_atype(symbols)
    except KeyError as exc:
        raise ValueError(
            f"element {exc.args[0]!r} is absent from the checkpoint type_map"
        ) from exc


__all__ = [
    "_ATOMIC_SYMBOLS",
]
