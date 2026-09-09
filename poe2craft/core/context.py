"""Pre-resolved weights for one (base, ilvl, target-spec) triple.

Everything the transition kernels need is folded into a handful of numbers so
that a single crafting step costs O(#requirements), not O(#mods).

A context can be *filtered* by minimum modifier level (Greater / Perfect orbs,
Ancient bones): the filtered view has the same requirements but weights summed
only over mods with required_level >= L.  Filtered views are cached per level.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .mods import GenType, Mod, ModPool
from .state import ItemState, SlotState


@dataclass(frozen=True, slots=True)
class Requirement:
    """One line of the target spec, e.g. 'T1-T2 % increased Physical Damage'."""
    name: str
    gen_type: GenType
    groups: frozenset[str]                    # mod groups that can satisfy it
    min_tier: int = 1                         # accept tier <= min_tier ... worst
    max_tier: int = 1
    def accepts(self, m: Mod) -> bool:
        return m.group in self.groups and self.min_tier <= m.tier <= self.max_tier


LevelDist = dict[int, float]                  # required_level -> probability


def _dist(rows: list[tuple[Mod, int]]) -> LevelDist:
    tot = sum(w for _, w in rows)
    if tot <= 0:
        return {}
    d: dict[int, float] = defaultdict(float)
    for m, w in rows:
        d[m.required_level] += w / tot
    return dict(d)


@dataclass
class CraftContext:
    """Immutable, pre-computed weight tables for the solver."""
    pool: ModPool
    reqs: tuple[Requirement, ...]

    sides: tuple[bool, ...] = ()              # True == prefix
    w_hit: tuple[int, ...] = ()               # weight that satisfies req i
    w_block: tuple[int, ...] = ()             # weight of same-group non-satisfying mods
    w_junk: dict = field(default_factory=dict)      # GenType -> total junk weight
    n_junk_groups: dict = field(default_factory=dict)
    junk_decay: bool = True                   # correct junk weight as junk accumulates
    min_level: int = 0                        # this view only counts mods with level >= min_level

    # level distributions (for Omen of Whittling), computed on the unfiltered pool
    lvl_hit: tuple[LevelDist, ...] = ()
    lvl_block: tuple[LevelDist, ...] = ()
    lvl_junk: dict = field(default_factory=dict)    # GenType -> LevelDist

    # Abyss-only desecrated mods eligible for this base (offered only at the Well of Souls)
    special_pool: dict = field(default_factory=dict)          # GenType -> [(Mod, weight)]
    w_special_desecrated: dict = field(default_factory=dict)  # GenType -> total weight
    w_hit_special: tuple[int, ...] = ()                       # special weight satisfying req i
    _filtered: dict = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, pool: ModPool, reqs: tuple[Requirement, ...], junk_decay: bool = True,
              min_level: int = 0, special_pool: dict | None = None) -> "CraftContext":
        special_pool = {g: list(v) for g, v in (special_pool or {}).items()}
        sides, w_hit, w_block, lvl_hit, lvl_block, w_hit_sp = [], [], [], [], [], []
        req_groups: set[str] = set()
        for r in reqs:
            req_groups |= r.groups
            w_hit_sp.append(sum(w for m, w in special_pool.get(r.gen_type, [])
                                if r.accepts(m) and m.required_level >= min_level))
            side = [(m, w) for m, w in pool.side(r.gen_type) if m.required_level >= min_level]
            hits = [(m, w) for m, w in side if r.accepts(m)]
            blks = [(m, w) for m, w in side if m.group in r.groups and not r.accepts(m)]
            sides.append(r.gen_type is GenType.PREFIX)
            w_hit.append(sum(w for _, w in hits))
            w_block.append(sum(w for _, w in blks))
            lvl_hit.append(_dist(hits))
            lvl_block.append(_dist(blks))

        w_junk, n_groups, lvl_junk = {}, {}, {}
        for gen in (GenType.PREFIX, GenType.SUFFIX):
            side = [(m, w) for m, w in pool.side(gen) if m.required_level >= min_level]
            junk = [(m, w) for m, w in side if m.group not in req_groups]
            w_junk[gen] = sum(w for _, w in junk)
            n_groups[gen] = len({m.group for m, _ in junk}) or 1
            lvl_junk[gen] = _dist(junk)
        return cls(pool=pool, reqs=reqs, sides=tuple(sides), w_hit=tuple(w_hit),
                   w_block=tuple(w_block), w_junk=w_junk, n_junk_groups=n_groups,
                   junk_decay=junk_decay, min_level=min_level,
                   lvl_hit=tuple(lvl_hit), lvl_block=tuple(lvl_block), lvl_junk=lvl_junk,
                   special_pool=special_pool,
                   w_special_desecrated={g: sum(w for m, w in rows if m.required_level >= min_level)
                                         for g, rows in special_pool.items()},
                   w_hit_special=tuple(w_hit_sp))

    def filtered(self, min_level: int) -> "CraftContext":
        """Same spec, only mods with required_level >= min_level (Greater/Perfect orbs)."""
        if min_level <= self.min_level:
            return self
        if min_level not in self._filtered:
            self._filtered[min_level] = CraftContext.build(
                self.pool, self.reqs, self.junk_decay, min_level, self.special_pool)
        return self._filtered[min_level]

    # ---- available weight in a given state ------------------------------
    def junk_weight(self, gen: GenType, state: ItemState) -> float:
        w = float(self.w_junk[gen])
        if not self.junk_decay:
            return w
        k = state.junk_pre if gen is GenType.PREFIX else state.junk_suf
        g = self.n_junk_groups[gen]
        # each junk mod already present locks out (on average) one junk group
        return w * max(0.0, 1.0 - k / g)

    def open_weight(self, gen: GenType, state: ItemState) -> float:
        """Total weight of mods that could still be added on this side."""
        total = self.junk_weight(gen, state)
        is_pre = gen is GenType.PREFIX
        for i, r in enumerate(self.reqs):
            if self.sides[i] != is_pre:
                continue
            if state.slots[i] == SlotState.ABSENT:
                total += self.w_hit[i] + self.w_block[i]
        return total

    def side_has_room(self, gen: GenType, state: ItemState) -> bool:
        if gen is GenType.PREFIX:
            return state.n_pre(self.sides) < state.rarity.max_prefixes
        return state.n_suf(self.sides) < state.rarity.max_suffixes

    def gen_of(self, i: int) -> GenType:
        return GenType.PREFIX if self.sides[i] else GenType.SUFFIX
