# Figure 3 contract

Claim: the mean PORA trajectory reaches lower cost earlier in the supplied 24 paired Mumford1 records.
Archetype: quantitative comparison, one main axis and a 120–240 s detail inset.
Backend: existing Python/matplotlib workflow only; Times New Roman retained.
Evidence: 4 initial networks × 6 paired search seeds; 48 method runs and 1968 saved incumbent checkpoints.
Center: arithmetic mean of the last completed incumbent at each second. Spread: one sample standard deviation across 24 runs, not a confidence interval.
Initial value: the initial incumbent until the first completed checkpoint. Equal-work endpoints are reported in the paper table, not inferred from 400 s.
Outputs: editable SVG and PDF, 600-dpi PNG, print-size preview, source CSVs and reproducible script.
Styling: retain approved teal/purple curves, academic-red zoom guide and 120 s marker, and closed black frames.
QA: match 120 s means to endpoint table; verify n=24, bounds, readable text and arrows at 180 mm width.
