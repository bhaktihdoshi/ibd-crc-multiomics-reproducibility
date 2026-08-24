# IBD Threshold Transportability Diagnostic

The post-run diagnostic reviewed saved training-only OOF probabilities, frozen thresholds and external predicted probabilities. 9 selected external-validation contexts showed near or complete threshold collapse.

Where ROC-AUC remains acceptable but all or nearly all external probabilities fall on one side of the frozen threshold, the correct interpretation is:

> Ranking may transport, but the absolute probability scale and frozen decision threshold do not transport.

No external labels were used to change thresholds.
