"""Placeholder mod pool with PoE-shaped weights.

Used only until a real dump is wired into the loader.  Shape mirrors what the
GGPK gives you: many groups, ~8 tiers each, top tier gated by required_level and
carrying a lower spawn weight.
"""
from __future__ import annotations

from ..core.mods import BaseItem, GenType, Mod, StatRange

BOW = BaseItem(id="ExpertDualstringBow", name="Expert Dualstring Bow",
               item_class="Bow",
               tags=frozenset({"bow", "two_hand_weapon", "weapon", "ranged", "default"}),
               drop_level=77)

# group -> (gen_type, stat_id, top-tier weight, other-tier weight)
_PREFIX_GROUPS = {
    "LocalPhysicalDamagePercent":  ("increased Physical Damage", 500, 1000),
    "LocalPhysicalDamage":         ("Adds Physical Damage", 400, 1000),
    "LocalFireDamage":             ("Adds Fire Damage", 400, 1000),
    "LocalColdDamage":             ("Adds Cold Damage", 400, 1000),
    "LocalLightningDamage":        ("Adds Lightning Damage", 400, 1000),
    "LocalChaosDamage":            ("Adds Chaos Damage", 100, 250),
    "LocalAccuracyRating":         ("Accuracy Rating", 500, 1000),
    "LocalPhysicalDamageFlat":     ("Flat Phys (hybrid)", 300, 800),
    "LocalAttackSpeedHybrid":      ("Phys + Attack Speed hybrid", 200, 600),
    "LifeLeechLocal":              ("Life Leech", 250, 600),
}
_SUFFIX_GROUPS = {
    "LocalIncreasedAttackSpeed":   ("increased Attack Speed", 400, 1000),
    "LocalCriticalStrikeChance":   ("increased Critical Hit Chance", 350, 1000),
    "CriticalMultiplier":          ("increased Critical Damage Bonus", 300, 900),
    "IncreasedDexterity":          ("+# to Dexterity", 700, 1000),
    "IncreasedStrength":           ("+# to Strength", 700, 1000),
    "IncreasedIntelligence":       ("+# to Intelligence", 700, 1000),
    "FireResistance":              ("+#% Fire Resistance", 800, 1000),
    "ColdResistance":              ("+#% Cold Resistance", 800, 1000),
    "LightningResistance":         ("+#% Lightning Resistance", 800, 1000),
    "ChaosResistance":             ("+#% Chaos Resistance", 250, 500),
    "LightRadius":                 ("increased Light Radius", 1000, 1000),
    "ProjectileSpeed":             ("increased Projectile Speed", 500, 1000),
}

_TIER_LEVELS = [82, 75, 68, 60, 52, 44, 30, 1]   # tier 1 .. tier 8


def build_mods(n_tiers: int = 8) -> list[Mod]:
    mods: list[Mod] = []
    for groups, gen in ((_PREFIX_GROUPS, GenType.PREFIX), (_SUFFIX_GROUPS, GenType.SUFFIX)):
        for group, (label, w_top, w_rest) in groups.items():
            for t in range(1, n_tiers + 1):
                w = w_top if t == 1 else w_rest
                mods.append(Mod(
                    id=f"{group}{n_tiers - t + 1}", group=group, gen_type=gen,
                    required_level=_TIER_LEVELS[min(t, len(_TIER_LEVELS)) - 1],
                    tier=t, name=f"{label} (T{t})",
                    stats=(StatRange(group.lower(), 10 * (n_tiers - t + 1), 15 * (n_tiers - t + 1)),),
                    spawn_weights=(("bow", w), ("default", 0)),
                    tags=frozenset({"attack", "damage"} if gen is GenType.PREFIX else {"attack"}),
                ))
    return mods
