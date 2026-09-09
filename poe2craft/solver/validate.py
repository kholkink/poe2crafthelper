"""Ground-truth Monte-Carlo validator.

The solver runs on an ABSTRACT state (per-requirement status + junk counts +
fracture/crafted/desecrated bookkeeping), which assumes junk mods are
exchangeable and, for Omen of Whittling, that mod levels are independent draws
from their class distribution.  Those assumptions make the problem tractable, so
they have to be checked rather than believed.

This module simulates the *concrete* item -- real mod ids in real slots, real
weighted sampling, real mod-group exclusion, real modifier levels -- while
following the abstract policy.  If the abstraction is sound, the simulated mean
cost matches V(start).  A systematic gap is a bug or an unmodelled interaction.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..core.actions import Action
from ..core.context import CraftContext
from ..core.mods import GenType, Mod
from ..core.state import JUNK_PRE, JUNK_SUF, NONE, ItemState, Rarity, SlotState
from . import ssp

PREFIX, SUFFIX = GenType.PREFIX, GenType.SUFFIX


@dataclass
class ConcreteItem:
    rarity: Rarity = Rarity.NORMAL
    prefixes: list[Mod] = field(default_factory=list)
    suffixes: list[Mod] = field(default_factory=list)
    fractured: set[int] = field(default_factory=set)      # id(mod)
    crafted: int | None = None                            # id(mod) of the essence mod
    desecrated: int | None = None                         # id(mod) of the desecrated mod

    def side(self, gen: GenType) -> list[Mod]:
        return self.prefixes if gen is PREFIX else self.suffixes

    def mods(self) -> list[tuple[GenType, Mod]]:
        return [(PREFIX, m) for m in self.prefixes] + [(SUFFIX, m) for m in self.suffixes]

    def groups(self) -> set[str]:
        return {m.group for m in self.prefixes + self.suffixes}

    def n_affix(self) -> int:
        return len(self.prefixes) + len(self.suffixes)


def to_abstract(item: ConcreteItem, ctx: CraftContext) -> ItemState:
    slots, used = [], {}
    for i, r in enumerate(ctx.reqs):
        gen = ctx.gen_of(i)
        st = SlotState.ABSENT
        for m in item.side(gen):
            if m.group in r.groups:
                st = SlotState.HIT if r.accepts(m) else SlotState.BLOCKED
                used[id(m)] = i
                break
        slots.append(st)
    jp = sum(1 for m in item.prefixes if id(m) not in used)
    js = sum(1 for m in item.suffixes if id(m) not in used)

    def tag(mid: int) -> int:
        if mid in used:
            return used[mid]
        return JUNK_PRE if any(id(m) == mid for m in item.prefixes) else JUNK_SUF

    fractured = tuple(sorted(tag(mid) for mid in item.fractured))
    crafted = tag(item.crafted) if item.crafted is not None else NONE
    desecrated = tag(item.desecrated) if item.desecrated is not None else NONE
    return ItemState(item.rarity, tuple(slots), jp, js, fractured=fractured,
                     crafted=crafted, desecrated=desecrated)


# ---------------------------------------------------------------------------
# concrete operations
# ---------------------------------------------------------------------------
def _weighted_choice(rng: random.Random, cands: list) -> object:
    """cands: [(payload, weight)]"""
    tot = sum(w for _, w in cands)
    x = rng.random() * tot
    acc = 0.0
    for payload, w in cands:
        acc += w
        if x <= acc:
            return payload
    return cands[-1][0]


def _add_concrete(item: ConcreteItem, ctx: CraftContext, rng: random.Random,
                  force_side: GenType | None = None, force_req: int | None = None,
                  min_level: int = 0, ok: bool = True) -> Mod | None:
    """Add one real mod, sampled by true spawn weight with group exclusion."""
    present = item.groups()
    caps = {PREFIX: item.rarity.max_prefixes, SUFFIX: item.rarity.max_suffixes}
    sides = [force_side] if force_side else [PREFIX, SUFFIX]
    if force_req is not None:
        sides = [ctx.gen_of(force_req)]
    cands: list[tuple[tuple[Mod, GenType], float]] = []
    for gen in sides:
        if len(item.side(gen)) >= caps[gen]:
            continue
        for m, w in ctx.pool.side(gen):
            if m.group in present or m.required_level < min_level:
                continue
            if force_req is not None:
                r = ctx.reqs[force_req]
                if m.group not in r.groups:
                    continue
                if r.accepts(m) != ok:
                    continue
            cands.append(((m, gen), w))
    if not cands:
        return None
    m, gen = _weighted_choice(rng, cands)
    item.side(gen).append(m)
    return m


def _removable(item: ConcreteItem, force_side: GenType | None = None) -> list[tuple[GenType, int]]:
    out = []
    for gen in (PREFIX, SUFFIX):
        if force_side is not None and gen is not force_side:
            continue
        for i, m in enumerate(item.side(gen)):
            if id(m) not in item.fractured:
                out.append((gen, i))
    return out


def _pop(item: ConcreteItem, gen: GenType, idx: int) -> Mod:
    m = item.side(gen).pop(idx)
    if item.crafted == id(m):
        item.crafted = None
    if item.desecrated == id(m):
        item.desecrated = None
    return m


def _remove_concrete(item: ConcreteItem, rng: random.Random,
                     force_side: GenType | None = None) -> Mod | None:
    pool = _removable(item, force_side)
    if not pool:
        return None
    gen, idx = rng.choice(pool)
    return _pop(item, gen, idx)


def _whittle_concrete(item: ConcreteItem, rng: random.Random) -> Mod | None:
    pool = _removable(item)
    if not pool:
        return None
    lo = min(item.side(g)[i].required_level for g, i in pool)
    lowest = [(g, i) for g, i in pool if item.side(g)[i].required_level == lo]
    gen, idx = rng.choice(lowest)
    return _pop(item, gen, idx)


def _fracture_concrete(item: ConcreteItem, rng: random.Random) -> None:
    cands = [m for _, m in item.mods() if id(m) != item.desecrated]
    if cands:
        item.fractured.add(id(rng.choice(cands)))


def _desecrate_concrete(item: ConcreteItem, ctx: CraftContext, rng: random.Random,
                        bone_level: int, force_side: GenType | None, offers: int) -> None:
    caps = {PREFIX: item.rarity.max_prefixes, SUFFIX: item.rarity.max_suffixes}
    sides = [force_side] if force_side else [PREFIX, SUFFIX]
    open_sides = [g for g in sides if len(item.side(g)) < caps[g]]
    if open_sides:
        gen = rng.choice(open_sides)
    else:
        removed = _remove_concrete(item, rng, force_side)
        if removed is None:
            return
        gen = PREFIX if removed in item.prefixes or len(item.prefixes) < caps[PREFIX] else SUFFIX
        if force_side is not None:
            gen = force_side
    present = item.groups()
    cands: list[tuple[Mod, float]] = [(m, w) for m, w in ctx.pool.side(gen)
                                      if m.group not in present and m.required_level >= bone_level]
    cands += [(m, w) for m, w in ctx.special_pool.get(gen, [])
              if m.group not in present and m.required_level >= bone_level]
    if not cands:
        return
    offered = [_weighted_choice(rng, cands) for _ in range(offers)]
    chosen = None
    for m in offered:                                   # best: satisfies an open requirement
        for i, r in enumerate(ctx.reqs):
            if r.gen_type is gen and r.accepts(m) and not any(x.group in r.groups for x in item.side(gen)):
                chosen = m
                break
        if chosen is not None:
            break
    if chosen is None:                                  # else: a junk that blocks nothing
        req_groups = {g for r in ctx.reqs for g in r.groups}
        harmless = [m for m in offered if m.group not in req_groups]
        chosen = rng.choice(harmless) if harmless else offered[0]
    item.side(gen).append(chosen)
    item.desecrated = id(chosen)


def apply_action(item: ConcreteItem, act: Action, ctx: CraftContext,
                 rng: random.Random, cfg) -> ConcreteItem:
    """Replay an abstract action against the concrete item."""
    n = act.name
    if act.is_reset:
        return ConcreteItem()

    if n.startswith("desecrate:"):
        parts = n.split("+")
        bone = parts[0].split(":", 1)[1]
        side = PREFIX if "necromancy-sinistral" in n else SUFFIX if "necromancy-dextral" in n else None
        offers = cfg.desecrate_offers * (2 if "abyssal-echoes" in n else 1)
        _desecrate_concrete(item, ctx, rng, cfg.bone_levels.get(bone, 0), side, offers)
        return item

    if "essence" in n:
        idx = int(n.rsplit("req", 1)[1])
        if n.startswith("perfect-essence"):
            side = PREFIX if "omen:sinistral" in n else SUFFIX if "omen:dextral" in n else None
            _remove_concrete(item, rng, side)
            m = _add_concrete(item, ctx, rng, force_req=idx, ok=cfg.perfect_essence_ok.get(idx, True))
        else:
            item.rarity = cfg.essence_upgrades_to
            m = _add_concrete(item, ctx, rng, force_req=idx, ok=cfg.essence_ok.get(idx, True))
        if m is not None:
            item.crafted = id(m)
        return item

    op, *mods = n.split("+")
    level = 0
    if "greater" in mods:
        level = cfg.greater_level_magic if op in ("transmute", "augment") else cfg.greater_level
    if "perfect" in mods:
        level = cfg.perfect_level_magic if op in ("transmute", "augment") else cfg.perfect_level
    omen = next((m.split(":", 1)[1] for m in mods if m.startswith("omen:")), None)
    side = PREFIX if omen and omen.startswith("sinistral") else SUFFIX if omen and omen.startswith("dextral") else None

    if op == "transmute":
        item.rarity = Rarity.MAGIC
        for _ in range(cfg.transmute_mods):
            _add_concrete(item, ctx, rng, min_level=level)
    elif op == "augment":
        _add_concrete(item, ctx, rng, min_level=level)
    elif op == "regal":
        item.rarity = Rarity.RARE
        for _ in range(cfg.regal_mods):
            _add_concrete(item, ctx, rng, min_level=level)
    elif op == "alchemy":
        item.rarity = Rarity.RARE
        for _ in range(cfg.alchemy_mods):
            _add_concrete(item, ctx, rng)
    elif op == "exalt":
        if omen == "greater":
            for _ in range(cfg.greater_exalt_mods):
                _add_concrete(item, ctx, rng)
        else:
            _add_concrete(item, ctx, rng, force_side=side, min_level=level)
    elif op == "annul":
        _remove_concrete(item, rng, side)
    elif op == "chaos":
        if omen == "whittling":
            _whittle_concrete(item, rng)
        else:
            _remove_concrete(item, rng, side)
        _add_concrete(item, ctx, rng, min_level=level)
    elif op == "fracture":
        _fracture_concrete(item, rng)
    else:
        raise ValueError(f"validator does not know action {n!r}")
    return item


def run(ctx: CraftContext, sol: ssp.Solution, cfg, n_runs: int = 500,
        seed: int = 0, max_steps: int = 200000) -> dict:
    """Simulate the optimal policy on concrete items; report the mean true cost."""
    if sol.start != ssp.blank_state(len(ctx.reqs)):
        raise ValueError("Monte-Carlo validation supports only a blank white-base start")
    rng = random.Random(seed)
    costs, steps_used, truncated = [], [], 0
    for _ in range(n_runs):
        item, spent, steps = ConcreteItem(), 0.0, 0
        while steps < max_steps:
            st = to_abstract(item, ctx)
            if st.is_goal():
                break
            act = sol.policy.get(st)
            if act is None:                       # off-policy state: restart
                item, spent = ConcreteItem(), spent + 1.0
                steps += 1
                continue
            spent += act.cost
            item = apply_action(item, act, ctx, rng, cfg)
            steps += 1
        else:
            truncated += 1
        costs.append(spent); steps_used.append(steps)
    mean = sum(costs) / len(costs)
    var = sum((c - mean) ** 2 for c in costs) / max(len(costs) - 1, 1)
    se = (var / len(costs)) ** 0.5
    costs_sorted = sorted(costs)
    return {"mean": mean, "stderr": se, "n": n_runs, "truncated": truncated,
            "median": costs_sorted[len(costs) // 2],
            "p90": costs_sorted[int(len(costs) * 0.9)],
            "mean_steps": sum(steps_used) / len(steps_used)}
