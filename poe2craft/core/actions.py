"""Crafting actions as transition kernels over ItemState.

Every currency / essence / omen / bone combination is expressed as a function
    ItemState -> {ItemState: probability}
built from a few primitives: add_affix, remove_affix (uniform / by side /
lowest-level), fracture, desecrate, set_rarity.  Numbers that GGG can (and does)
change between patches live in MechanicsConfig, never inline.

Minimum-modifier-level currency (Greater / Perfect orbs, Ancient bones) reuses
the same kernels on a level-filtered view of the context (CraftContext.filtered).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

from .context import CraftContext
from .mods import GenType
from .state import JUNK_PRE, JUNK_SUF, NONE, ItemState, Rarity, SlotState

Dist = dict[ItemState, float]
PREFIX, SUFFIX = GenType.PREFIX, GenType.SUFFIX


def _norm(d: Dist) -> Dist:
    tot = sum(d.values())
    return {k: v / tot for k, v in d.items()} if tot > 0 else {}


def bind(d: Dist, fn: Callable[[ItemState], Dist]) -> Dist:
    """Monadic chaining: apply a kernel to every outcome of a distribution."""
    out: Dist = defaultdict(float)
    for s, p in d.items():
        for s2, p2 in fn(s).items():
            out[s2] += p * p2
    return dict(out)


def repeat(fn: Callable[[ItemState], Dist], n: int) -> Callable[[ItemState], Dist]:
    def k(s: ItemState) -> Dist:
        d: Dist = {s: 1.0}
        for _ in range(n):
            d = bind(d, fn)
        return d
    return k


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def add_affix(ctx: CraftContext, s: ItemState,
              force_side: Optional[GenType] = None,
              force_req: Optional[int] = None, force_ok: bool = True) -> Dist:
    """Add one random affix, weighted by spawn weight, respecting group locks.

    force_side -- omens such as Sinistral/Dextral Exaltation
    force_req  -- essence/desecration guaranteeing requirement i; force_ok=False
                  means the guaranteed mod is the wrong tier (BLOCKED)
    """
    if force_req is not None:
        i = force_req
        gen = ctx.gen_of(i)
        if not ctx.side_has_room(gen, s) or s.slots[i] != SlotState.ABSENT:
            return {s: 1.0}
        return {s.with_slot(i, SlotState.HIT if force_ok else SlotState.BLOCKED): 1.0}

    sides = [force_side] if force_side else [PREFIX, SUFFIX]
    avail = []
    for gen in sides:
        if ctx.side_has_room(gen, s):
            w = ctx.open_weight(gen, s)
            if w > 0:
                avail.append((gen, w))
    if not avail:
        return {s: 1.0}                       # no-op: currency is wasted
    tot_side = sum(w for _, w in avail)

    out: Dist = defaultdict(float)
    for gen, w_side in avail:
        p_side = w_side / tot_side
        is_pre = gen is PREFIX
        for i, _ in enumerate(ctx.reqs):
            if ctx.sides[i] != is_pre or s.slots[i] != SlotState.ABSENT:
                continue
            if ctx.w_hit[i]:
                out[s.with_slot(i, SlotState.HIT)] += p_side * ctx.w_hit[i] / w_side
            if ctx.w_block[i]:
                out[s.with_slot(i, SlotState.BLOCKED)] += p_side * ctx.w_block[i] / w_side
        jw = ctx.junk_weight(gen, s)
        if jw > 0:
            nxt = s.replace(junk_pre=s.junk_pre + 1) if is_pre else s.replace(junk_suf=s.junk_suf + 1)
            out[nxt] += p_side * jw / w_side
    return _norm(dict(out))


def _removable_junk(s: ItemState, gen: GenType) -> int:
    """Junk mods on a side that are not locked by a fracture."""
    if gen is PREFIX:
        return s.junk_pre - (1 if JUNK_PRE in s.fractured else 0)
    return s.junk_suf - (1 if JUNK_SUF in s.fractured else 0)


def _removable(ctx: CraftContext, s: ItemState) -> list[tuple[str, object]]:
    """Removable mods as ('req', i) or ('junk', GenType) entries (fractured ones are immune)."""
    items: list[tuple[str, object]] = []
    for i, st in enumerate(s.slots):
        if st != SlotState.ABSENT and i not in s.fractured:
            items.append(("req", i))
    items += [("junk", PREFIX)] * _removable_junk(s, PREFIX)
    items += [("junk", SUFFIX)] * _removable_junk(s, SUFFIX)
    return items


def _after_remove(s: ItemState, kind: str, val) -> Dist:
    """Outcome of removing one candidate, with crafted/desecrated bookkeeping."""
    if kind == "req":
        i = val
        nxt = s.with_slot(i, SlotState.ABSENT)
        if s.crafted == i:
            nxt = nxt.replace(crafted=NONE)
        if s.desecrated == i:
            nxt = nxt.replace(desecrated=NONE)
        return {nxt: 1.0}
    gen = val
    is_pre = gen is PREFIX
    n = _removable_junk(s, gen)
    nxt = s.replace(junk_pre=s.junk_pre - 1) if is_pre else s.replace(junk_suf=s.junk_suf - 1)
    marker = JUNK_PRE if is_pre else JUNK_SUF
    if s.desecrated == marker and n > 0:
        p = 1.0 / n                            # the desecrated junk is one of the n removable
        cleared = nxt.replace(desecrated=NONE)
        return {cleared: 1.0} if p >= 1.0 else {cleared: p, nxt: 1.0 - p}
    return {nxt: 1.0}


def remove_affix(ctx: CraftContext, s: ItemState,
                 force_side: Optional[GenType] = None,
                 only_blocked: bool = False,
                 only_junk: bool = False) -> Dist:
    """Remove one affix uniformly at random (annulment is unweighted)."""
    cands = _removable(ctx, s)
    if force_side:
        is_pre = force_side is PREFIX
        cands = [c for c in cands
                 if (c[0] == "req" and ctx.sides[c[1]] == is_pre)
                 or (c[0] == "junk" and c[1] is force_side)]
    if only_blocked:
        cands = [c for c in cands if c[0] == "req" and s.slots[c[1]] == SlotState.BLOCKED]
    if only_junk:
        cands = [c for c in cands if c[0] == "junk"]
    if not cands:
        return {s: 1.0}
    out: Dist = defaultdict(float)
    p = 1.0 / len(cands)
    for kind, val in cands:
        for s2, p2 in _after_remove(s, kind, val).items():
            out[s2] += p * p2
    return dict(out)


def _level_dist(ctx: CraftContext, s: ItemState, kind: str, val) -> dict[int, float]:
    if kind == "req":
        return ctx.lvl_hit[val] if s.slots[val] == SlotState.HIT else ctx.lvl_block[val]
    return ctx.lvl_junk[val]


def whittle_remove(ctx: CraftContext, s: ItemState) -> Dist:
    """Omen of Whittling: remove the LOWEST-level modifier.

    Mod levels are not part of the abstract state, so each present mod gets the
    level distribution of its class (hit / blocker / junk, from the unfiltered pool)
    and the mods are treated as independent.  Ties are split by normalisation.
    APPROXIMATION: junk added by Greater/Perfect orbs is higher-level than the
    unfiltered junk distribution says, so the risk of whittling a wanted mod is
    underestimated on such items.  The Monte-Carlo validator uses true levels.
    """
    cands = _removable(ctx, s)
    if not cands:
        return {s: 1.0}
    dists = [_level_dist(ctx, s, k, v) for k, v in cands]
    weights = []
    for k, dk in enumerate(dists):
        acc = 0.0
        for lvl, p in dk.items():
            prod = p
            for j, dj in enumerate(dists):
                if j == k or not dj:
                    continue
                prod *= sum(pj for lj, pj in dj.items() if lj >= lvl)
            acc += prod
        weights.append(acc)
    tot = sum(weights)
    if tot <= 0:
        return remove_affix(ctx, s)
    out: Dist = defaultdict(float)
    for (kind, val), w in zip(cands, weights):
        if w <= 0:
            continue
        for s2, p2 in _after_remove(s, kind, val).items():
            out[s2] += w / tot * p2
    return dict(out)


def fracture(ctx: CraftContext, s: ItemState) -> Dist:
    """Fracturing Orb: lock one random present mod (desecrated mods cannot be fractured)."""
    cands: list[tuple[str, object]] = []
    for i, st in enumerate(s.slots):
        if st != SlotState.ABSENT and s.desecrated != i:
            cands.append(("req", i))
    cands += [("junk", PREFIX)] * (s.junk_pre - (1 if s.desecrated == JUNK_PRE else 0))
    cands += [("junk", SUFFIX)] * (s.junk_suf - (1 if s.desecrated == JUNK_SUF else 0))
    if not cands:
        return {s: 1.0}
    out: Dist = defaultdict(float)
    p = 1.0 / len(cands)
    for kind, val in cands:
        if kind == "req":
            out[s.replace(fractured=(val,))] += p
        else:
            out[s.replace(fractured=((JUNK_PRE if val is PREFIX else JUNK_SUF),))] += p
    return dict(out)


def desecrate(ctx: CraftContext, s: ItemState, bone_level: int = 0,
              force_side: Optional[GenType] = None, offers: int = 3) -> Dist:
    """Bone + Well of Souls: add a desecrated mod, choosing the best of `offers` options.

    The offer pool for a side = the regular pool filtered by the bone's minimum
    modifier level + the Abyss-only desecrated mods (w_special_desecrated).  We
    take the offer that satisfies an open requirement whenever one exists,
    otherwise a junk mod.  If the item is full a random mod is removed first.
    """
    fctx = ctx.filtered(bone_level)
    sides = [force_side] if force_side else [PREFIX, SUFFIX]
    branches: list[tuple[ItemState, GenType, float]] = []
    open_sides = [g for g in sides if ctx.side_has_room(g, s)]
    if open_sides:
        for g in open_sides:
            branches.append((s, g, 1.0 / len(open_sides)))
    else:
        # full: a random modifier is removed first, the desecrated mod takes its place
        rem = remove_affix(ctx, s, force_side=force_side)
        for s2, p in rem.items():
            if s2 == s:
                continue
            g = PREFIX if s2.n_pre(ctx.sides) < s.n_pre(ctx.sides) else SUFFIX
            branches.append((s2, g, p))
        if not branches:
            return {s: 1.0}

    out: Dist = defaultdict(float)
    for s2, gen, pb in branches:
        w_total = fctx.open_weight(gen, s2) + fctx.w_special_desecrated.get(gen, 0.0)
        is_pre = gen is PREFIX
        ps = {}
        if w_total > 0:
            for i, _ in enumerate(ctx.reqs):
                wi = fctx.w_hit[i] + fctx.w_hit_special[i]
                if ctx.sides[i] == is_pre and s2.slots[i] == SlotState.ABSENT and wi > 0:
                    ps[i] = wi / w_total
        psum = sum(ps.values())
        p_miss = (1.0 - psum) ** offers if psum < 1.0 else 0.0
        for i, pi in ps.items():
            out[s2.with_slot(i, SlotState.HIT).replace(desecrated=i)] += pb * (1.0 - p_miss) * pi / psum
        junk = (s2.replace(junk_pre=s2.junk_pre + 1, desecrated=JUNK_PRE) if is_pre
                else s2.replace(junk_suf=s2.junk_suf + 1, desecrated=JUNK_SUF))
        out[junk] += pb * p_miss
    return _norm(dict(out))


def strip(s: ItemState) -> ItemState:
    """Back to a blank Normal item (what re-buying a white base gives you)."""
    return ItemState(Rarity.NORMAL, tuple(SlotState.ABSENT for _ in s.slots), 0, 0)


# --------------------------------------------------------------------------
# mechanics configuration -- patch-dependent numbers live here
# --------------------------------------------------------------------------
@dataclass
class MechanicsConfig:
    transmute_mods: int = 1
    alchemy_mods: int = 4
    regal_mods: int = 1
    exalt_mods: int = 1
    greater_exalt_mods: int = 2             # Omen of Greater Exaltation
    chaos_removes: int = 1
    chaos_adds: int = 1
    essence_upgrades_to: Rarity = Rarity.RARE
    essence_from: Rarity = Rarity.MAGIC
    # minimum modifier level of Greater / Perfect currency (0.5.x, poe2wiki)
    greater_level: int = 35                 # Greater Regal / Exalted / Chaos
    greater_level_magic: int = 44           # Greater Transmutation / Augmentation
    perfect_level: int = 50                 # Perfect Regal / Exalted / Chaos
    perfect_level_magic: int = 70           # Perfect Transmutation / Augmentation
    bone_levels: dict = field(default_factory=lambda: {"preserved": 0, "ancient": 40})
    desecrate_offers: int = 3
    # requirement index -> does the essence's guaranteed mod satisfy the tier range?
    essence_ok: dict = field(default_factory=dict)
    perfect_essence_ok: dict = field(default_factory=dict)


@dataclass
class Action:
    name: str
    cost: float                                   # in the chosen numeraire
    kernel: Callable[[CraftContext, ItemState], Dist]
    applicable: Callable[[CraftContext, ItemState], bool]
    tags: frozenset[str] = frozenset()

    @property
    def is_reset(self) -> bool:
        return "reset" in self.tags

    def transitions(self, ctx: CraftContext, s: ItemState) -> Dist:
        return self.kernel(ctx, s)


def _rarity_up(s: ItemState, r: Rarity) -> ItemState:
    return s.replace(rarity=r)


ALL_MECHANICS = frozenset({"basic", "omen_exalt", "omen_annul", "essence", "greater", "perfect",
                           "omen_chaos", "whittling", "fracture", "perfect_essence", "desecrate"})


def standard_actions(cfg: MechanicsConfig, prices: dict[str, float], n_reqs: int,
                     start: ItemState | None = None,
                     mechanics: frozenset[str] | set[str] = ALL_MECHANICS) -> list[Action]:
    """Build the action set.  `prices` maps action-price keys -> cost in the numeraire.

    Stock currency has fallback prices (so demos run without a market snapshot);
    every 0.5-toolbox action is included only when its price key is present.
    `start` is where RESET lands (default: a blank white base).
    """
    P = lambda n, d=None: prices.get(n, d)
    A: list[Action] = []
    T = lambda *t: frozenset(t)

    def add(name, cost, kernel, applicable, *tags):
        if cost is None:
            return
        A.append(Action(name, cost, kernel, applicable, T(*tags)))

    notc = lambda s: not s.corrupted
    ok_ess = lambda i: cfg.essence_ok.get(i, True)
    ok_pess = lambda i: cfg.perfect_essence_ok.get(i, True)

    # --- basic orbs ---------------------------------------------------------
    def transmute_k(L):
        return lambda c, s: bind({_rarity_up(s, Rarity.MAGIC): 1.0},
                                 repeat(lambda x: add_affix(c.filtered(L), x), cfg.transmute_mods))
    def augment_k(L):
        return lambda c, s: add_affix(c.filtered(L), s)
    def regal_k(L):
        return lambda c, s: bind({_rarity_up(s, Rarity.RARE): 1.0},
                                 repeat(lambda x: add_affix(c.filtered(L), x), cfg.regal_mods))
    def exalt_k(L):
        return lambda c, s: add_affix(c.filtered(L), s)
    def chaos_k(L, force_side=None, whittle=False):
        rem = (lambda c, s: whittle_remove(c, s)) if whittle else \
              (lambda c, s: remove_affix(c, s, force_side=force_side))
        return lambda c, s: bind(rem(c, s), lambda x: add_affix(c.filtered(L), x))

    is_normal = lambda c, s: s.rarity is Rarity.NORMAL and notc(s)
    is_magic_room = lambda c, s: s.rarity is Rarity.MAGIC and s.n_affix(c.sides) < 2 and notc(s)
    is_magic = lambda c, s: s.rarity is Rarity.MAGIC and notc(s)
    is_rare_room = lambda c, s: s.rarity is Rarity.RARE and s.n_affix(c.sides) < 6 and notc(s)
    is_rare_mods = lambda c, s: s.rarity is Rarity.RARE and len(_removable(c, s)) > 0 and notc(s)

    add("transmute", P("transmute", 0.01), transmute_k(0), is_normal, "basic")
    add("augment", P("augment", 0.02), augment_k(0), is_magic_room, "basic")
    add("regal", P("regal", 0.5), regal_k(0), is_magic, "basic")
    add("alchemy", P("alchemy", 0.3),
        lambda c, s: bind({_rarity_up(s, Rarity.RARE): 1.0},
                          repeat(lambda x: add_affix(c, x), cfg.alchemy_mods)), is_normal, "basic")
    add("exalt", P("exalt", 1.0), exalt_k(0), is_rare_room, "basic")
    add("annul", P("annul", 1.5), lambda c, s: remove_affix(c, s),
        lambda c, s: len(_removable(c, s)) > 0 and notc(s), "basic")
    add("chaos", P("chaos", 0.8), chaos_k(0), is_rare_mods, "basic")

    # --- greater / perfect variants (minimum modifier level) -------------------
    for tier, Lm, Lr in (("greater", cfg.greater_level_magic, cfg.greater_level),
                         ("perfect", cfg.perfect_level_magic, cfg.perfect_level)):
        add(f"transmute+{tier}", P(f"transmute_{tier}"), transmute_k(Lm), is_normal, tier)
        add(f"augment+{tier}", P(f"augment_{tier}"), augment_k(Lm), is_magic_room, tier)
        add(f"regal+{tier}", P(f"regal_{tier}"), regal_k(Lr), is_magic, tier)
        add(f"exalt+{tier}", P(f"exalt_{tier}"), exalt_k(Lr), is_rare_room, tier)
        add(f"chaos+{tier}", P(f"chaos_{tier}"), chaos_k(Lr), is_rare_mods, tier)

    # --- omens on exalt / annul / chaos -----------------------------------------
    ex = P("exalt", 1.0)
    add("exalt+omen:sinistral", ex + P("omen_sinistral_exalt", 8.0),
        lambda c, s: add_affix(c, s, force_side=PREFIX),
        lambda c, s: s.rarity is Rarity.RARE and c.side_has_room(PREFIX, s) and notc(s), "omen_exalt")
    add("exalt+omen:dextral", ex + P("omen_dextral_exalt", 8.0),
        lambda c, s: add_affix(c, s, force_side=SUFFIX),
        lambda c, s: s.rarity is Rarity.RARE and c.side_has_room(SUFFIX, s) and notc(s), "omen_exalt")
    add("exalt+omen:greater", ex + P("omen_greater_exalt", 15.0),
        lambda c, s: repeat(lambda x: add_affix(c, x), cfg.greater_exalt_mods)(s),
        lambda c, s: s.rarity is Rarity.RARE and s.n_affix(c.sides) < 5 and notc(s), "omen_exalt")
    an = P("annul", 1.5)
    add("annul+omen:sinistral", an + P("omen_sinistral_annul", 6.0),
        lambda c, s: remove_affix(c, s, force_side=PREFIX),
        lambda c, s: _removable_side(c, s, PREFIX) and notc(s), "omen_annul")
    add("annul+omen:dextral", an + P("omen_dextral_annul", 6.0),
        lambda c, s: remove_affix(c, s, force_side=SUFFIX),
        lambda c, s: _removable_side(c, s, SUFFIX) and notc(s), "omen_annul")
    ch = P("chaos", 0.8)
    if P("omen_sinistral_erasure") is not None:
        add("chaos+omen:sinistral-erasure", ch + P("omen_sinistral_erasure"), chaos_k(0, PREFIX),
            lambda c, s: s.rarity is Rarity.RARE and _removable_side(c, s, PREFIX) and notc(s), "omen_chaos")
    if P("omen_dextral_erasure") is not None:
        add("chaos+omen:dextral-erasure", ch + P("omen_dextral_erasure"), chaos_k(0, SUFFIX),
            lambda c, s: s.rarity is Rarity.RARE and _removable_side(c, s, SUFFIX) and notc(s), "omen_chaos")
    if P("omen_whittling") is not None:
        add("chaos+omen:whittling", ch + P("omen_whittling"), chaos_k(0, whittle=True), is_rare_mods, "whittling")

    # --- essences (magic -> rare, guaranteed mod; counts as the item's one crafted mod)
    for i in range(n_reqs):
        add(f"essence->req{i}", P(f"essence_{i}"),
            (lambda idx: lambda c, s: _with_crafted(
                bind({_rarity_up(s, cfg.essence_upgrades_to): 1.0},
                     lambda x: add_affix(c, x, force_req=idx, force_ok=ok_ess(idx))), s, idx))(i),
            (lambda idx: lambda c, s: s.rarity is cfg.essence_from and s.slots[idx] == SlotState.ABSENT
                                      and s.crafted == NONE and notc(s))(i), "essence")
        # perfect essence: on a rare, remove a random mod (optionally one side) then add
        for variant, side, okey in (("", None, None),
                                    ("+omen:sinistral", PREFIX, "omen_sinistral_crystallisation"),
                                    ("+omen:dextral", SUFFIX, "omen_dextral_crystallisation")):
            base = P(f"perfect_essence_{i}")
            extra = 0.0 if okey is None else P(okey)
            if base is None or extra is None:
                continue
            add(f"perfect-essence{variant}->req{i}", base + extra,
                (lambda idx, sd: lambda c, s: _perfect_essence(c, s, idx, ok_pess(idx), sd))(i, side),
                (lambda idx: lambda c, s: s.rarity is Rarity.RARE and s.slots[idx] == SlotState.ABSENT
                                          and s.crafted == NONE and notc(s))(i), "perfect_essence")

    # --- fracturing orb -------------------------------------------------------------
    if P("fracturing") is not None:
        add("fracture", P("fracturing"), lambda c, s: fracture(c, s),
            lambda c, s: s.rarity is Rarity.RARE and s.n_affix(c.sides) >= 4 and not s.is_fractured
                         and notc(s), "fracture")

    # --- desecration (bone + Well of Souls) -------------------------------------------
    for bone, L in cfg.bone_levels.items():
        bp = P(f"bone_{bone}")
        if bp is None:
            continue
        for variant, side, okey in (("", None, None),
                                    ("+omen:necromancy-sinistral", PREFIX, "omen_sinistral_necromancy"),
                                    ("+omen:necromancy-dextral", SUFFIX, "omen_dextral_necromancy")):
            extra = 0.0 if okey is None else P(okey)
            if extra is None:
                continue
            for echo, ek in (("", None), ("+omen:abyssal-echoes", "omen_abyssal_echoes")):
                e_extra = 0.0 if ek is None else P(ek)
                if e_extra is None:
                    continue
                offers = cfg.desecrate_offers * (2 if ek else 1)
                add(f"desecrate:{bone}{variant}{echo}", bp + extra + e_extra,
                    (lambda LL, sd, of: lambda c, s: desecrate(c, s, LL, sd, of))(L, side, offers),
                    lambda c, s: s.rarity is Rarity.RARE and s.desecrated == NONE and notc(s), "desecrate")

    # --- reset: buy another copy of the starting base ------------------------------------
    # Throwing away the start state to buy the start state again is a no-op that
    # costs currency; excluding it removes a degenerate self-loop.
    land = start if start is not None else None
    add("RESET(buy base)", P("base", 0.1),
        lambda c, s: {(land if land is not None else strip(s)): 1.0},
        lambda c, s: s != (land if land is not None else strip(s)), "reset")

    keep = set(mechanics) | {"reset"}
    return [a for a in A if a.tags & keep]


def _removable_side(ctx: CraftContext, s: ItemState, gen: GenType) -> bool:
    is_pre = gen is PREFIX
    if any(st != SlotState.ABSENT and i not in s.fractured and ctx.sides[i] == is_pre
           for i, st in enumerate(s.slots)):
        return True
    return _removable_junk(s, gen) > 0


def _with_crafted(d: Dist, s: ItemState, idx: int) -> Dist:
    """Mark the essence mod as the item's crafted mod wherever it was actually added."""
    out: Dist = defaultdict(float)
    for s2, p in d.items():
        if s2.slots[idx] != SlotState.ABSENT and s.slots[idx] == SlotState.ABSENT:
            s2 = s2.replace(crafted=idx)
        out[s2] += p
    return dict(out)


def _perfect_essence(ctx: CraftContext, s: ItemState, i: int, ok: bool,
                     force_side: Optional[GenType]) -> Dist:
    """Perfect essence on a rare: remove a random mod (or one from `force_side` with a
    Crystallisation omen), then add the guaranteed mod if its side has room."""
    out: Dist = defaultdict(float)
    gen = ctx.gen_of(i)
    for s2, p in remove_affix(ctx, s, force_side=force_side).items():
        if ctx.side_has_room(gen, s2) and s2.slots[i] == SlotState.ABSENT:
            s3 = s2.with_slot(i, SlotState.HIT if ok else SlotState.BLOCKED).replace(crafted=i)
        else:
            s3 = s2                                   # no room on that side: essence wasted
        out[s3] += p
    return dict(out)
