import sys, time
sys.path.insert(0, '/home/joke/poe2craft')
from poe2craft.core.mods import GenType, ModPool
from poe2craft.core.context import CraftContext, Requirement
from poe2craft.core.actions import MechanicsConfig, standard_actions
from poe2craft.solver import ssp
from poe2craft.data.synthetic import BOW, build_mods

mods = build_mods()
pool = ModPool.build(BOW, ilvl=82, mods=mods)
print(f"base={BOW.name} ilvl=82  prefixes={len(pool.prefixes)} suffixes={len(pool.suffixes)}")
print(f"total prefix weight={pool.total_weight(GenType.PREFIX)}  suffix weight={pool.total_weight(GenType.SUFFIX)}")

reqs = (
    Requirement("T1-T2 % increased Physical Damage", GenType.PREFIX,
                frozenset({"LocalPhysicalDamagePercent"}), 1, 2),
    Requirement("T1-T2 Adds Physical Damage", GenType.PREFIX,
                frozenset({"LocalPhysicalDamage"}), 1, 2),
    Requirement("T1-T2 increased Attack Speed", GenType.SUFFIX,
                frozenset({"LocalIncreasedAttackSpeed"}), 1, 2),
    Requirement("T1-T3 increased Critical Hit Chance", GenType.SUFFIX,
                frozenset({"LocalCriticalStrikeChance"}), 1, 3),
)
ctx = CraftContext.build(pool, reqs)
print("\nper-requirement weights (hit / same-group blocker):")
for i, r in enumerate(reqs):
    side = "P" if ctx.sides[i] else "S"
    tot = ctx.open_weight(GenType.PREFIX if ctx.sides[i] else GenType.SUFFIX,
                          ssp.blank_state(len(reqs)))
    print(f"  [{side}] {r.name:<40} hit={ctx.w_hit[i]:>6}  block={ctx.w_block[i]:>6}"
          f"  P(hit on a fresh add)={ctx.w_hit[i]/tot:6.2%}")

prices = {"base": 0.1, "transmute": 0.01, "augment": 0.02, "regal": 0.4, "alchemy": 0.25,
          "exalt": 1.0, "annul": 2.0, "chaos": 0.6,
          "omen_sinistral_exalt": 10.0, "omen_dextral_exalt": 10.0, "omen_greater_exalt": 20.0,
          "omen_sinistral_annul": 8.0, "omen_dextral_annul": 8.0,
          "essence_0": 4.0, "essence_1": 4.0, "essence_2": 6.0, "essence_3": 6.0}
actions = standard_actions(MechanicsConfig(), prices, len(reqs))

t0 = time.time()
sol = ssp.solve(ctx, actions)
dt = time.time() - t0
print(f"\nsolved: {sol.n_states} reachable states, {sol.outer_iters} outer, {sol.inner_sweeps} inner sweeps, {dt*1000:.0f} ms")
print(f"OPTIMAL EXPECTED COST = {sol.expected_cost:.2f} exalted-equivalents\n")

print("optimal plan (most-likely trajectory):")
for line in ssp.extract_plan(ctx, sol):
    print("  " + line)

print("\nexpected currency consumption under the optimal policy:")
counts = ssp.expected_action_counts(ctx, sol)
for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    a = next(x for x in actions if x.name == name)
    print(f"  {name:<26} x{n:9.2f}   = {n*a.cost:9.2f}")
print(f"  {'TOTAL':<26}  {'':9}   = {sum(n*next(x for x in actions if x.name==k).cost for k,n in counts.items()):9.2f}")
