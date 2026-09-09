"""Abstract item state.

The exact state of an item (which of ~120 mods sit in which of 6 slots, at which
tier) is astronomically large.  The key reduction: for a *given target spec*, the
only thing that matters about a mod is its relation to the spec.  Every mod in
the pool is classified once into:

  * SLOT i      -- it satisfies requirement i of the target
  * BLOCKER i   -- same mod-group as requirement i but does NOT satisfy it
                   (it occupies the group, so requirement i can no longer roll)
  * JUNK        -- irrelevant; only its side (prefix/suffix) and its count matter

Junk mods are treated as exchangeable.  That is an *exact* aggregation as long as
junk-vs-junk group collisions are ignored; the engine can fall back to explicit
mod-set enumeration when that approximation is not acceptable (see docs).

Three small bookkeeping fields make the 0.5 mechanics expressible:
  fractured  -- indices locked by a Fracturing Orb; JUNK_PRE / JUNK_SUF mark a
                fractured junk mod on that side (at most one fracture per item)
  crafted    -- requirement index whose mod came from an essence (max 1 crafted
                mod per item since 0.5.0), or NONE
  desecrated -- requirement index of the desecrated mod, JUNK_PRE / JUNK_SUF for a
                junk desecrated mod, or NONE (an item can be desecrated only once
                while it carries a desecrated mod)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

NONE = -1
JUNK_PRE = -2
JUNK_SUF = -3


class Rarity(IntEnum):
    NORMAL = 0
    MAGIC = 1
    RARE = 2

    @property
    def max_prefixes(self) -> int:
        return (0, 1, 3)[self]

    @property
    def max_suffixes(self) -> int:
        return (0, 1, 3)[self]

    @property
    def max_affixes(self) -> int:
        return (0, 2, 6)[self]


class SlotState(IntEnum):
    ABSENT = 0    # nothing of this group on the item -- requirement can still roll
    BLOCKED = 1   # wrong-tier / wrong mod of the same group occupies the group
    HIT = 2       # requirement satisfied


@dataclass(frozen=True, slots=True)
class ItemState:
    """Abstract, hashable item state used as the MDP state."""
    rarity: Rarity
    slots: tuple[SlotState, ...]   # one entry per target requirement
    junk_pre: int = 0
    junk_suf: int = 0
    corrupted: bool = False
    fractured: tuple[int, ...] = ()     # requirement indices (or JUNK_PRE/JUNK_SUF) locked by fracture
    crafted: int = NONE                 # requirement index holding the essence (crafted) mod
    desecrated: int = NONE              # requirement index / JUNK_PRE / JUNK_SUF of the desecrated mod

    # ---- derived -------------------------------------------------------
    def n_pre(self, sides: tuple[bool, ...]) -> int:
        """sides[i] is True when requirement i is a prefix."""
        occupied = sum(1 for i, s in enumerate(self.slots) if s != SlotState.ABSENT and sides[i])
        return occupied + self.junk_pre

    def n_suf(self, sides: tuple[bool, ...]) -> int:
        occupied = sum(1 for i, s in enumerate(self.slots) if s != SlotState.ABSENT and not sides[i])
        return occupied + self.junk_suf

    def n_affix(self, sides: tuple[bool, ...]) -> int:
        return self.n_pre(sides) + self.n_suf(sides)

    def is_goal(self) -> bool:
        return all(s == SlotState.HIT for s in self.slots)

    @property
    def is_fractured(self) -> bool:
        return len(self.fractured) > 0

    def with_slot(self, i: int, v: SlotState) -> "ItemState":
        s = list(self.slots)
        s[i] = v
        return self.replace(slots=tuple(s))

    def replace(self, **kw) -> "ItemState":
        d = dict(rarity=self.rarity, slots=self.slots, junk_pre=self.junk_pre,
                 junk_suf=self.junk_suf, corrupted=self.corrupted,
                 fractured=self.fractured, crafted=self.crafted, desecrated=self.desecrated)
        d.update(kw)
        return ItemState(**d)

    def key(self) -> tuple:
        return (int(self.rarity), self.slots, self.junk_pre, self.junk_suf,
                self.corrupted, self.fractured, self.crafted, self.desecrated)

    def __str__(self) -> str:
        sym = {SlotState.ABSENT: ".", SlotState.BLOCKED: "x", SlotState.HIT: "#"}
        s = (f"{self.rarity.name[0]}[{''.join(sym[x] for x in self.slots)}]"
             f"+{self.junk_pre}p{self.junk_suf}s")
        if self.fractured:
            s += "F" + ",".join(_tag(i) for i in self.fractured)
        if self.crafted != NONE:
            s += f"E{_tag(self.crafted)}"
        if self.desecrated != NONE:
            s += f"D{_tag(self.desecrated)}"
        if self.corrupted:
            s += "C"
        return s


def _tag(i: int) -> str:
    return {JUNK_PRE: "jp", JUNK_SUF: "js", NONE: "-"}.get(i, str(i))


def enumerate_states(n_req: int, sides: tuple[bool, ...],
                     max_junk_pre: int = 3, max_junk_suf: int = 3,
                     rarities: tuple[Rarity, ...] = (Rarity.NORMAL, Rarity.MAGIC, Rarity.RARE),
                     ) -> Iterator[ItemState]:
    """All legal abstract states without bookkeeping fields -- used to size the solver."""
    from itertools import product
    for rar in rarities:
        for slots in product(SlotState, repeat=n_req):
            for jp in range(max_junk_pre + 1):
                for js in range(max_junk_suf + 1):
                    st = ItemState(rar, slots, jp, js)
                    if st.n_pre(sides) > rar.max_prefixes:
                        continue
                    if st.n_suf(sides) > rar.max_suffixes:
                        continue
                    yield st
