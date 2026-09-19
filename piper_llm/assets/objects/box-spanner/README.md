# box-spanner

A tubular box spanner, 14 by 15 mm, DIN 896 form B: a zinc-plated steel tube
141 mm long, 19 mm across in the middle, with its ends formed into hex sockets
(20.3 and 21.3 mm across the corners outside) and a cross hole near each end.
The working mesh `box-spanner.stl` is in metres; `meta.json` has the
measurements, the frame (origin at the centre of volume, z along the tube
towards the 14 mm end, x towards a hex corner) and the symmetry (a half turn
about z).

Standing on an end, it is a peg the pipe fitting's 25.5 mm bore can slide over,
with about 2 mm to spare on each side at the hex ends. Standing is not a likely
way for a dropped tube to land, so load it with `min_prob=0` to keep those
resting poses.

Generated from calliper measurements of the real part.
