from __future__ import annotations

from dataclasses import dataclass

REGIONS = ["East", "West", "South", "Midwest"]
SEED_MATCHUPS = [
    (1, 16),
    (8, 9),
    (5, 12),
    (4, 13),
    (6, 11),
    (3, 14),
    (7, 10),
    (2, 15),
]


@dataclass(frozen=True)
class TeamSlot:
    school: str
    seed: int
    region: str


def assign_seeds(field):
    """Assign seeds and regions using a snake draft."""
    if len(field) < 64:
        raise ValueError("Field must have at least 64 teams")

    slots = []
    ranked = field.reset_index(drop=True)
    for seed in range(1, 17):
        seed_slice = ranked.iloc[(seed - 1) * 4 : seed * 4]
        if seed_slice.empty:
            continue
        regions = REGIONS if seed % 2 == 1 else list(reversed(REGIONS))
        for region, (_, row) in zip(regions, seed_slice.iterrows()):
            slots.append(TeamSlot(school=row["School"], seed=seed, region=region))
    return slots


def build_first_round(slots):
    """Build first round matchups per region."""
    by_region = {region: [] for region in REGIONS}
    for slot in slots:
        by_region[slot.region].append(slot)

    matchups = []
    for region, region_slots in by_region.items():
        region_slots = sorted(region_slots, key=lambda s: s.seed)
        seed_map = {slot.seed: slot for slot in region_slots}
        for seed_a, seed_b in SEED_MATCHUPS:
            team_a = seed_map.get(seed_a)
            team_b = seed_map.get(seed_b)
            if team_a and team_b:
                matchups.append((region, team_a, team_b))
    return matchups
