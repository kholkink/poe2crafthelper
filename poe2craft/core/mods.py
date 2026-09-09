"""Core mod / base-item model for PoE2 crafting.

Weight resolution follows the GGPK convention used by PoE1 (and, per the data
dumps, PoE2): a mod carries an *ordered* list of (tag, weight) pairs.  The first
entry whose tag is present on the item wins; a `default` entry acts as the
catch-all.  A resolved weight of 0 means the mod cannot spawn on that base.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence


class GenType(str, Enum):
    PREFIX = "prefix"
    SUFFIX = "suffix"
    IMPLICIT = "implicit"
    CORRUPTED = "corrupted"
    DESECRATED = "desecrated"
    ENCHANT = "enchant"
    UNIQUE = "unique"

    @property
    def is_affix(self) -> bool:
        return self in (GenType.PREFIX, GenType.SUFFIX)


@dataclass(frozen=True, slots=True)
class StatRange:
    stat_id: str
    lo: int
    hi: int

    def mid(self) -> float:
        return (self.lo + self.hi) / 2.0


@dataclass(frozen=True, slots=True)
class Mod:
    """One modifier row (one tier of one mod family)."""
    id: str                       # e.g. "LocalIncreasedPhysicalDamagePercent5"
    group: str                    # family; two mods of one group cannot coexist
    gen_type: GenType
    required_level: int
    tier: int                     # 1 = best; used for display + target specs
    name: str = ""
    domain: str = "item"
    stats: tuple[StatRange, ...] = ()
    # ordered (tag, weight); "default" is the catch-all
    spawn_weights: tuple[tuple[str, int], ...] = ()
    # the mod's OWN tags -- what essences/omens/desecration filter on
    tags: frozenset[str] = frozenset()

    def weight_for(self, item_tags: frozenset[str]) -> int:
        for tag, w in self.spawn_weights:
            if tag == "default" or tag in item_tags:
                return w
        return 0


@dataclass(frozen=True, slots=True)
class BaseItem:
    id: str
    name: str
    item_class: str               # "Bow", "Body Armour", "Ring", ...
    tags: frozenset[str]          # drives weight resolution
    drop_level: int = 1
    implicits: tuple[str, ...] = ()
    max_prefixes: int = 3
    max_suffixes: int = 3


@dataclass
class ModPool:
    """The eligible mod pool for one (base, ilvl) pair, pre-resolved."""
    base: BaseItem
    ilvl: int
    prefixes: list[tuple[Mod, int]] = field(default_factory=list)   # (mod, weight)
    suffixes: list[tuple[Mod, int]] = field(default_factory=list)

    @classmethod
    def build(cls, base: BaseItem, ilvl: int, mods: Iterable[Mod]) -> "ModPool":
        pool = cls(base=base, ilvl=ilvl)
        for m in mods:
            if not m.gen_type.is_affix or m.domain != "item":
                continue
            if m.required_level > ilvl:
                continue
            w = m.weight_for(base.tags)
            if w <= 0:
                continue
            (pool.prefixes if m.gen_type is GenType.PREFIX else pool.suffixes).append((m, w))
        pool.prefixes.sort(key=lambda t: t[0].id)
        pool.suffixes.sort(key=lambda t: t[0].id)
        return pool

    def side(self, gen: GenType) -> list[tuple[Mod, int]]:
        return self.prefixes if gen is GenType.PREFIX else self.suffixes

    def total_weight(self, gen: GenType, blocked_groups: frozenset[str] = frozenset()) -> int:
        return sum(w for m, w in self.side(gen) if m.group not in blocked_groups)

    def weight_of(self, mods: Sequence[Mod], blocked_groups: frozenset[str] = frozenset()) -> int:
        return sum(m.weight_for(self.base.tags) for m in mods if m.group not in blocked_groups)
