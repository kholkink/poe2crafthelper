"""End-to-end demo on the real PoE2 mod dump."""
import sys, time
sys.path.insert(0, '/home/joke/poe2craft')
from collections import Counter
from poe2craft.core.mods import GenType, ModPool
from poe2craft.core.context import CraftContext, Requirement
from poe2craft.core.actions import MechanicsConfig, standard_actions
from poe2craft.solver import ssp
from poe2craft.data.repoe import Repoe, poe1_prior_weight, uniform_weight
from poe2craft.data.weights import CommunityWeights

ILVL = 82
r = Repoe.load('/home/joke/poe2craft/cache')
base = r.find_bases(item_class='Bow')[0]
cw = CommunityWeights.load('/home/joke/poe2craft/cache/community/weights_poe2.json', 'Bow')
missed = []
mods = r.mods_for(base, weight_model=cw.model(fallback=0, warn=missed))
print(f"WEIGHTS  community table: {len(cw.by_id)} weighted ids, {len(missed)} eligible mods had no weight (dropped)")
pool = ModPool.build(base, ilvl=ILVL, mods=mods)

print(f"BASE  {base.name}  (ilvl {ILVL})   tags={sorted(base.tags)}")
print(f"POOL  {len(pool.prefixes)} prefix rows / {len(pool.suffixes)} suffix rows"
      f"   W_pre={pool.total_weight(GenType.PREFIX)}  W_suf={pool.total_weight(GenType.SUFFIX)}")
print("\nfamilies available at this ilvl:")
for gen in (GenType.PREFIX, GenType.SUFFIX):
    fams = sorted({m.group for m, _ in pool.side(gen)})
    print(f"  {gen.value}: {', '.join(fams)}")

reqs = (
    Requirement("T1-T2  % increased Physical Damage", GenType.PREFIX,
                frozenset({"LocalPhysicalDamagePercent"}), 1, 2),
    Requirement("T1-T2  Adds Physical Damage",        GenType.PREFIX,
                frozenset({"PhysicalDamage"}), 1, 2),
    Requirement("T1-T2  increased Attack Speed",      GenType.SUFFIX,
                frozenset({"IncreasedAttackSpeed"}), 1, 2),
    Requirement("T1-T3  increased Critical Hit Chance", GenType.SUFFIX,
                frozenset({"CriticalStrikeChanceIncrease"}), 1, 3),
)
ctx = CraftContext.build(pool, reqs)
blank = ssp.blank_state(len(reqs))
print("\nTARGET SPEC")
for i, q in enumerate(reqs):
    gen = GenType.PREFIX if ctx.sides[i] else GenType.SUFFIX
    tot = ctx.open_weight(gen, blank)
    ok = "OK " if ctx.w_hit[i] > 0 else "!! family not in pool"
    print(f"  {ok} {q.name:<40} w_hit={ctx.w_hit[i]:>6} w_block={ctx.w_block[i]:>6}"
          f"  P(single add hits)={ctx.w_hit[i]/tot if tot else 0:7.3%}")

prices = {"base": 1.0, "transmute": 0.02, "augment": 0.05, "regal": 0.5, "alchemy": 0.3,
          "exalt": 1.0, "annul": 2.5, "chaos": 0.7,
          "omen_sinistral_exalt": 12.0, "omen_dextral_exalt": 12.0, "omen_greater_exalt": 25.0,
          "omen_sinistral_annul": 9.0, "omen_dextral_annul": 9.0,
          "essence_0": 5.0, "essence_1": 5.0, "essence_2": 8.0, "essence_3": 8.0}
actions = standard_actions(MechanicsConfig(), prices, len(reqs))

t0 = time.time(); sol = ssp.solve_exact(ctx, actions, verbose=True); dt = time.time() - t0
print(f"\nSOLVED  {sol.n_states} states | {sol.outer_iters} policy-iteration steps,"
      f" {sol.inner_sweeps} sweeps | {dt:.2f}s")
print(f"OPTIMAL EXPECTED COST = {sol.expected_cost:,.1f} exalted-equivalents")

print("\nOPTIMAL PLAN (most-likely trajectory)")
for line in ssp.extract_plan(ctx, sol): print("  " + line)

print("\nEXPECTED CURRENCY SPEND UNDER THE OPTIMAL POLICY")
counts = ssp.expected_action_counts(ctx, sol)
byname = {a.name: a for a in actions}
tot = 0.0
for name, n in sorted(counts.items(), key=lambda kv: -kv[1] * byname[kv[0]].cost):
    c = n * byname[name].cost; tot += c
    print(f"  {name:<26} x{n:10.2f}  = {c:10.1f}")
print(f"  {'TOTAL':<26} {'':11}  = {tot:10.1f}")

# --- independent validation on concrete items ---------------------------------
from poe2craft.solver import validate
print("\nGROUND-TRUTH VALIDATION (concrete items, real mod ids, real group collisions)")
t0 = time.time()
v = validate.run(ctx, sol, MechanicsConfig(), n_runs=400, seed=7)
print(f"  simulated mean cost : {v['mean']:,.1f}  +/- {v['stderr']:,.1f} (1 s.e., n={v['n']})")
print(f"  solver V(blank)     : {sol.expected_cost:,.1f}")
gap = v['mean'] - sol.expected_cost
z = gap / v['stderr'] if v['stderr'] else 0
print(f"  gap                 : {gap:+,.1f}  ({z:+.2f} standard errors)")
print(f"  median / p90 cost   : {v['median']:,.0f} / {v['p90']:,.0f}   mean steps: {v['mean_steps']:,.0f}")
print(f"  truncated runs      : {v['truncated']}   [{time.time()-t0:.0f}s]")
