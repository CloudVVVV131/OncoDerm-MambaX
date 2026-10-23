# R5.1 figure sources

`OncoDerm_Architecture_Figures_R5_1.pptx`: two editable slides, Figure1 and FigureS1, transcribed from the bound original model. The slides are technical illustrations, not experimental model code.

`render_statistical_figures.py`: uses the adjacent saved CSV files to reproduce Figure3 and Figure4 into `../Figures`. Requires matplotlib, pandas and Pillow; no model library or dataset is loaded. It does not estimate new confidence intervals, thresholds or performance.

`Figure_3_per_run.csv`:12 observed within-checkpoint AP input differences, source DataS1 fixed-input and DataS2 CORE tables.
`Figure_3_intervals.csv`:4 original mean effects and97.5% bounds.
`Figure_4_operating_points.csv`:8 saved original-model common-head available-input operating points; bars represent sample SD.
`Table_2C_display.csv`:display-precision four original AP comparisons, consistent with SupplementS29.

The two original data ZIPs remain byte-identical to R5. Figure2 and FiguresS2–S9 are retained outputs, not regenerated datasets or new measurements.
