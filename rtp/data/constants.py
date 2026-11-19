"""Constants of the data generators.

They live here, not in ``rtp.config``, so that the data hash can cover the generator's own code
without covering the whole run configuration module: editing a measurement setting or the model
registry must not force a rebuild of the data.
"""

from __future__ import annotations

# Legacy four-option family (rtp.data.choice_items). ``evidence_level`` there is INFORMATION SUFFICIENCY
# (how much of the card is shown); the only adverse evidence is ``handicap`` (the target's rating lowered).
EVIDENCE_LEVELS = (0, 1, 2, 3, 4)
HANDICAPS = (0, 1, 2, 3, 4)
HANDICAP_MIN_EVIDENCE = 3  # handicap only applies where ratings are shown (evidence >= 3)

# Ladder family (rtp.data.ladder_items).
LADDER_STRATA = ("name", "attr", "full", "comp")  # information strata: name only / manipulated attribute only /
#                                                   full card / full card + a stated user requirement
LADDER_TYPES = ("rating", "price", "reviews")  # evidence types with their own ladders
LADDER_TIERS = ("core", "ext", "ctx")  # item files: core panel, extension, four-option context arm
