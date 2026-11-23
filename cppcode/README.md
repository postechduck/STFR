# cppcode

`stfr_popularity.cpp` computes the time-decayed item popularity used by the TIDE
and CausalEPP baselines (derived from the TIDE authors' implementation).  It is
compiled on first use through `cppimport` (needs `pybind11` and a C++ compiler);
the build products (`*.so`, `.rendered.*`) are written next to the source and are
ignored by git.  STFR itself and the other methods do not use this module.

The module reads `data/<dataset>/item_interactions.csv` written by
`prep/prep_split.py` (one row per item: `id,count,t1,t2,...` with training
interaction times in ascending order).
