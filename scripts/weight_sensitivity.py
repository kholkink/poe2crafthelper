"""How much does the answer depend on the weight table we cannot verify?"""
import sys, time; sys.path.insert(0,'/home/joke/poe2craft')
from poe2craft.core.mods import GenType, ModPool
from poe2craft.core.context import CraftContext, Requirement
from poe2craft.core.actions import MechanicsConfig, standard_actions
from poe2craft.solver import ssp
from poe2craft.data.repoe import Repoe, uniform_weight
from poe2craft.data.weights import CommunityWeights

r = Repoe.load('/home/joke/poe2craft/cache')
base = r.find_bases(item_class='Bow')[0]
cw = CommunityWeights.load('/home/joke/poe2craft/cache/community/weights_poe2.json','Bow')
reqs = (
 Requirement("T1-T2 phys%",   GenType.PREFIX, frozenset({"LocalPhysicalDamagePercent"}),1,2),
 Requirement("T1-T2 flat phys",GenType.PREFIX, frozenset({"PhysicalDamage"}),1,2),
 Requirement("T1-T2 atk spd", GenType.SUFFIX, frozenset({"IncreasedAttackSpeed"}),1,2),
 Requirement("T1-T3 crit",    GenType.SUFFIX, frozenset({"CriticalStrikeChanceIncrease"}),1,3),
)
prices = {"base":1.0,"transmute":0.02,"augment":0.05,"regal":0.5,"alchemy":0.3,"exalt":1.0,
 "annul":2.5,"chaos":0.7,"omen_sinistral_exalt":12.0,"omen_dextral_exalt":12.0,
 "omen_greater_exalt":25.0,"omen_sinistral_annul":9.0,"omen_dextral_annul":9.0,
 "essence_0":5.0,"essence_1":5.0,"essence_2":8.0,"essence_3":8.0}

print(f"{'weight source':<34}{'P(T1-T2 phys% per add)':>24}{'optimal EV':>14}   top action")
print("-"*92)
for label, wm in (("client dump (all eligible = 1)", uniform_weight),
                  ("community empirical estimates", cw.model(fallback=0))):
    pool = ModPool.build(base, 82, r.mods_for(base, weight_model=wm))
    ctx = CraftContext.build(pool, reqs)
    blank = ssp.blank_state(4)
    p0 = ctx.w_hit[0]/ctx.open_weight(GenType.PREFIX, blank)
    sol = ssp.solve_exact(ctx, standard_actions(MechanicsConfig(), prices, 4))
    counts = ssp.expected_action_counts(ctx, sol)
    acts = {a.name:a for a in standard_actions(MechanicsConfig(), prices, 4)}
    top = max(counts.items(), key=lambda kv: kv[1]*acts[kv[0]].cost)[0]
    print(f"{label:<34}{p0:>23.3%}{sol.expected_cost:>14,.0f}   {top}")
