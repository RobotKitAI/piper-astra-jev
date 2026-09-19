# trowel

A small cat's-tongue trowel: a polished steel blade, 140 by 67 mm, a steel rod
rising from the blade's heel into a steel ferrule, and a beech handle leaning up
from there. The working mesh `trowel.stl` is in metres; `meta.json` has the
measurements and the frame (origin at the centre of volume, z up from the blade
towards the handle, x from the heel edge towards the tip).

The steel is ten times denser than the beech, so the tool balances under the
ferrule, 42 mm in front of the centre of volume. `meta.json` gives that centre of
mass (`centre_of_mass_m`) and how it was estimated; `piper_llm/geometry.py`
uses it for the ways the trowel can rest and for where to hold it.

Generated from calliper measurements of the real tool. Run 6 uses it.
