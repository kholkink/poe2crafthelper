"""Stochastic-Shortest-Path solver for crafting.

    V(goal) = 0
    V(s)    = min_a [ cost(a) + sum_s' P(s'|s,a) V(s') ]

The hard part is the RESET action ("bin it, buy a new white base"), which makes
V(s_blank) appear on the right-hand side of its own equation.  Plain value
iteration crawls, because the contraction factor is the probability of *not*
finishing a cycle, which for a real craft is ~0.99.

The fix is to split the fixed point out as a scalar.  Let

    f(v0) = value of the blank state in the MDP where RESET is a TERMINAL action
            costing (c_base + v0)

f is monotone non-decreasing and concave in v0, with slope equal to the
probability that the optimal policy ever resets -- strictly below 1 for any
proper policy.  The true optimum is the unique root of  f(v0) - v0 = 0, found by
bisection.  Each inner solve gets a *tight finite* upper bound for free, since
RESET is available everywhere: V(s) <= c_base + v0.  Initialising there makes
the inner Gauss-Seidel sweeps converge monotonically downward in a handful of
passes instead of thousands.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..core.actions import Action
from ..core.context import CraftContext
from ..core.state import ItemState, Rarity, SlotState

TERMINAL = "__TERMINAL__"


@dataclass
class Solution:
    V: dict[ItemState, float]
    policy: dict[ItemState, Action]
    start: ItemState
    n_states: int
    outer_iters: int
    inner_sweeps: int
    trans: dict = field(default_factory=dict)

    @property
    def expected_cost(self) -> float:
        return self.V[self.start]


def blank_state(n_reqs: int) -> ItemState:
    return ItemState(Rarity.NORMAL, tuple(SlotState.ABSENT for _ in range(n_reqs)), 0, 0)


def reachable(ctx: CraftContext, actions: list[Action], start: ItemState) -> list[ItemState]:
    return prepare(ctx, actions, start)[0]


def prepare(ctx: CraftContext, actions: list[Action], start: ItemState):
    """BFS over the reachable states, computing every (state, action) kernel exactly once.

    Returns (states in discovery order, trans) where
        trans[s] = [(action, ((s2, p), ...)), ...]   for non-goal s
    A currency that provably cannot change anything is dropped from trans (never
    worth buying); RESET is always kept.
    """
    seen, order, q = {start}, [start], deque([start])
    trans: dict[ItemState, list[tuple[Action, tuple[tuple[ItemState, float], ...]]]] = {}
    while q:
        s = q.popleft()
        if s.is_goal():
            continue
        opts = []
        for a in actions:
            if not a.applicable(ctx, s):
                continue
            d = a.transitions(ctx, s)
            if not a.is_reset and len(d) == 1 and next(iter(d)) == s:
                continue
            opts.append((a, tuple(d.items())))
            for s2 in d:
                if s2 not in seen:
                    seen.add(s2); order.append(s2); q.append(s2)
        trans[s] = opts
    return order, trans


def _expand(ctx: CraftContext, actions: list[Action], states: list[ItemState]):
    """Kept for callers that already hold a state list; prefer prepare()."""
    trans = {}
    for s in states:
        if s.is_goal():
            continue
        opts = []
        for a in actions:
            if not a.applicable(ctx, s):
                continue
            d = a.transitions(ctx, s)
            if not a.is_reset and len(d) == 1 and next(iter(d)) == s:
                continue
            opts.append((a, tuple(d.items())))
        trans[s] = opts
    return trans


def _order_states(ctx: CraftContext, states: list[ItemState]) -> list[ItemState]:
    """Sweep richest-item-first: most actions add affixes, so value flows back
    from near-goal states in a single pass when visited in this order."""
    def rank(s: ItemState) -> tuple:
        hits = sum(1 for x in s.slots if x == SlotState.HIT)
        return (-hits, -s.n_affix(ctx.sides), -int(s.rarity))
    return sorted(states, key=rank)


def _inner_solve(ctx: CraftContext, trans, order, start, v0: float, base_cost: float,
                 eps: float, max_sweeps: int, V0: dict | None = None):
    """Solve the MDP with RESET treated as terminal at cost (base_cost + v0).

    V0 warm-starts from the previous outer iterate; VI for a proper SSP converges
    from any finite initialisation, and reusing the neighbouring solution cuts the
    sweep count by an order of magnitude.
    """
    UB = base_cost + v0
    if V0 is None:
        V = {s: (0.0 if s.is_goal() else UB) for s in order}
    else:
        V = {s: (0.0 if s.is_goal() else min(V0.get(s, UB), UB)) for s in order}
    policy: dict[ItemState, Action] = {}
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        residual = 0.0
        for s in order:
            if s.is_goal():
                continue
            best, best_a = math.inf, None
            for a, d in trans[s]:
                if a.is_reset:
                    q = a.cost + v0
                else:
                    q = a.cost
                    for s2, p in d:
                        q += p * V[s2]
                if q < best:
                    best, best_a = q, a
            if best_a is None:
                continue
            if abs(best - V[s]) > residual:
                residual = abs(best - V[s])
            V[s] = best
            policy[s] = best_a
        if residual < eps:
            break
    return V, policy, sweeps


def solve(ctx: CraftContext, actions: list[Action], start: ItemState | None = None,
          eps: float = 1e-7, max_sweeps: int = 400, outer_iters: int = 40,
          hi: float = 1e7) -> Solution:
    """Find V* and pi* via the scalar reset fixed point.

    Outer loop is a bracketed secant (Illinois) search for the root of
    g(v0) = f(v0) - v0, warm-starting each inner solve from the last one.
    """
    start = start or blank_state(len(ctx.reqs))
    states = reachable(ctx, actions, start)
    trans = _expand(ctx, actions, states)
    order = _order_states(ctx, states)
    reset_cost = next((a.cost for a in actions if a.is_reset), 0.0)

    warm: dict | None = None
    def f(v0: float) -> float:
        nonlocal warm
        V, pol, sw = _inner_solve(ctx, trans, order, start, v0, reset_cost, eps, max_sweeps, warm)
        warm = V
        f.last = (V, pol, sw)
        return V[start]

    lo = 0.0
    g_lo = f(lo) - lo                    # >= 0
    g_hi = f(hi) - hi                    # <= 0 for hi large enough
    it = 0
    total_sweeps = f.last[2]
    side = 0
    for it in range(1, outer_iters + 1):
        if abs(g_hi - g_lo) < 1e-18:
            break
        mid = lo - g_lo * (hi - lo) / (g_hi - g_lo)     # secant / false position
        mid = min(max(mid, lo), hi)
        g_mid = f(mid) - mid
        total_sweeps += f.last[2]
        if g_mid > 0:
            lo, g_lo = mid, g_mid
            if side == -1:
                g_hi *= 0.5                              # Illinois: unstick the lagging end
            side = -1
        else:
            hi, g_hi = mid, g_mid
            if side == 1:
                g_lo *= 0.5
            side = 1
        if abs(g_mid) < eps * max(1.0, abs(mid)) or (hi - lo) < eps * max(1.0, hi):
            break

    V, policy, sw = f.last
    return Solution(V=V, policy=policy, start=start, n_states=len(states),
                    outer_iters=it, inner_sweeps=total_sweeps + sw, trans=trans)


def _kernel(sol: Solution, ctx: CraftContext, s: ItemState, a: Action):
    for act, d in sol.trans.get(s, ()):
        if act is a:
            return d
    return tuple(a.transitions(ctx, s).items())


def _forward_mass(ctx: CraftContext, sol: Solution, tol: float = 1e-13, max_iter: int = 20000):
    """Push unit mass from the start through ONE attempt (reset treated as absorbing).

    Returns (state_visits, action_counts, p_reset) for that single attempt.  Every attempt
    is an i.i.d. restart from the blank state, so lifetime totals are the per-attempt
    totals divided by (1 - p_reset) -- the same geometric identity the solver uses.
    Iterating the reset loop directly would need ~1/(1-p_reset) sweeps (thousands for
    a real craft); this needs only as many as the longest plausible attempt.
    """
    visits: dict[ItemState, float] = defaultdict(float)
    counts: dict[str, float] = defaultdict(float)
    p_reset = 0.0
    frontier: dict[ItemState, float] = {sol.start: 1.0}
    for _ in range(max_iter):
        if not frontier:
            break
        nxt: dict[ItemState, float] = defaultdict(float)
        for s, m in frontier.items():
            if s.is_goal():
                continue
            a = sol.policy.get(s)
            if a is None:
                continue
            visits[s] += m
            counts[a.name] += m
            if a.is_reset:
                p_reset += m
                continue
            for s2, p in _kernel(sol, ctx, s, a):
                nxt[s2] += m * p
        frontier = {k: v for k, v in nxt.items() if v > tol}
    return visits, counts, p_reset


def expected_action_counts(ctx: CraftContext, sol: Solution, **_) -> dict[str, float]:
    """Expected number of times each action fires under the optimal policy (lifetime)."""
    _, counts, p_reset = _forward_mass(ctx, sol)
    scale = 1.0 / max(1e-15, 1.0 - p_reset)
    return {k: v * scale for k, v in counts.items()}


def expected_state_visits(ctx: CraftContext, sol: Solution) -> dict[ItemState, float]:
    """Expected number of visits to each state under the optimal policy (lifetime)."""
    visits, _, p_reset = _forward_mass(ctx, sol)
    scale = 1.0 / max(1e-15, 1.0 - p_reset)
    return {k: v * scale for k, v in visits.items()}


def extract_plan(ctx: CraftContext, sol: Solution, max_depth: int = 40) -> list[str]:
    """Walk the most-likely trajectory under the optimal policy."""
    lines, s, depth = [], sol.start, 0
    seen = set()
    while not s.is_goal() and depth < max_depth:
        a = sol.policy.get(s)
        if a is None:
            lines.append(f"{s}  -> (no action)")
            break
        d = _kernel(sol, ctx, s, a)
        p_progress = sum(p for s2, p in d if sol.V.get(s2, math.inf) < sol.V[s] - 1e-12)
        lines.append(f"{str(s):<24} EV={sol.V[s]:9.2f}  {a.name:<24}"
                     f" cost={a.cost:5.2f}  P(improve)={p_progress:6.1%}")
        if a.is_reset:
            s = sol.start
        else:
            moves = [(s2, p) for s2, p in d if s2 != s]
            s = max(moves or list(d), key=lambda kv: kv[1])[0]
        if (s, depth % 2) in seen and depth > 6:
            lines.append(f"{str(s):<24} ... (policy loops here; see expected counts)")
            break
        seen.add((s, depth % 2))
        depth += 1
    if s.is_goal():
        lines.append(f"{str(s):<24} EV={sol.V[s]:9.2f}  GOAL")
    return lines


# ==========================================================================
# Exact solver: policy iteration with a closed-form reset fixed point.
#
# Bisecting on V(s_blank) works but is slow and fragile: f(v0) is piecewise
# linear with flat stretches, so a secant stalls, and every probe at a large v0
# forces the inner VI to grind through low-escape-probability annul/chaos cycles.
#
# The clean route exploits the fact that RESET always lands on exactly one state.
# For a FIXED policy pi, absorb the chain at {goal, reset} and compute
#     a(s) = E[cost until absorption]         b(s) = P(absorbed via reset)
# Then every value is affine in the single unknown V0 = V(s_blank):
#     V(s)  = a(s) + b(s) * V0
#     V0    = a(0) + b(0) * V0     =>     V0 = a(0) / (1 - b(0))
# which is the textbook c/(1-p) geometric identity, generalised to an arbitrary
# policy.  No outer search, and the answer is exact for that policy.
# ==========================================================================
def _evaluate_idx(work, pol, n, eps=1e-11, max_sweeps=5000):
    """Exact policy evaluation on index arrays.

    work: state indices to sweep (non-goal, in sweep order)
    pol[i] = None (no action) | (cost, None) for reset | (cost, [(j, p), ...])
    Returns (a, b, sweeps): V(i) = a[i] + b[i] * V0 where V0 = V(start).
    """
    a = [0.0] * n
    b = [0.0] * n
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        delta = 0.0
        for i in work:
            e = pol[i]
            if e is None:
                na, nb = 0.0, 1.0
            elif e[1] is None:
                na, nb = e[0], 1.0
            else:
                na = e[0]; nb = 0.0
                for j, p in e[1]:
                    na += p * a[j]
                    nb += p * b[j]
            d1 = na - a[i]; d2 = nb - b[i]
            if d1 < 0: d1 = -d1
            if d2 < 0: d2 = -d2
            if d1 > delta: delta = d1
            if d2 > delta: delta = d2
            a[i] = na; b[i] = nb
        if delta < eps:
            break
    return a, b, sweeps


def solve_exact(ctx: CraftContext, actions: list[Action], start: ItemState | None = None,
                max_iters: int = 60, eval_eps: float = 1e-11, verbose: bool = False,
                prepared=None, bootstrap_sweeps: int = 25) -> Solution:
    """Policy iteration with the closed-form reset fixed point.  Exact.

    States are mapped to integers once; every sweep then runs on plain lists
    (hashing ItemState objects dominated the runtime otherwise).
    """
    start = start or blank_state(len(ctx.reqs))
    states, trans = prepared if prepared is not None else prepare(ctx, actions, start)
    order = _order_states(ctx, states)
    idx = {s: i for i, s in enumerate(order)}
    n = len(order)
    goal = [s.is_goal() for s in order]
    start_i = idx[start]
    # per state: list of (action_no, cost, is_reset, [(j, p), ...])
    opts: list[list] = [[] for _ in range(n)]
    for s, lst in trans.items():
        i = idx[s]
        opts[i] = [(a, a.cost, a.is_reset, [(idx[s2], p) for s2, p in d]) for a, d in lst]
    work = [i for i in range(n) if not goal[i]]

    # bootstrap: optimistic VI sweeps to get a usable initial policy
    BIG = 1e15
    V = [0.0 if goal[i] else BIG for i in range(n)]
    policy: list = [None] * n
    for _ in range(bootstrap_sweeps):
        for i in work:
            best, best_a = math.inf, None
            for a, cost, is_reset, d in opts[i]:
                q = cost
                for j, p in d:
                    q += p * V[j]
                if q < best:
                    best, best_a = q, a
            if best_a is not None:
                if best < V[i]:
                    V[i] = best
                policy[i] = best_a

    def pol_table():
        tbl: list = [None] * n
        for i in work:
            act = policy[i]
            if act is None:
                continue
            for a, cost, is_reset, d in opts[i]:
                if a is act:
                    tbl[i] = (cost, None) if is_reset else (cost, d)
                    break
        return tbl

    sweeps = 0
    it = 0
    for it in range(1, max_iters + 1):
        a_, b_, sw = _evaluate_idx(work, pol_table(), n, eps=eval_eps)
        sweeps += sw
        b0 = b_[start_i]
        V0 = math.inf if b0 >= 1.0 - 1e-14 else a_[start_i] / (1.0 - b0)
        if V0 != math.inf:
            V = [0.0 if goal[i] else a_[i] + b_[i] * V0 for i in range(n)]
        changed = 0
        for i in work:
            best, best_a = math.inf, None
            for a, cost, is_reset, d in opts[i]:
                if is_reset:
                    q = cost + V0
                else:
                    q = cost
                    for j, p in d:
                        q += p * V[j]
                if q < best - 1e-12:
                    best, best_a = q, a
            if best_a is not None and best_a is not policy[i]:
                policy[i], changed = best_a, changed + 1
        if verbose:
            print(f'    PI step {it}: V0={V0:,.2f}  policy changes={changed}', flush=True)
        if changed == 0:
            break
    a_, b_, sw = _evaluate_idx(work, pol_table(), n, eps=eval_eps)
    b0 = b_[start_i]
    V0 = math.inf if b0 >= 1.0 - 1e-14 else a_[start_i] / (1.0 - b0)
    Vd = {s: (0.0 if goal[i] else a_[i] + b_[i] * V0) for s, i in idx.items()}
    pd = {s: policy[i] for s, i in idx.items() if policy[i] is not None}
    return Solution(V=Vd, policy=pd, start=start, n_states=n,
                    outer_iters=it, inner_sweeps=sweeps + sw, trans=trans)
