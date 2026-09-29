# Data Leakage Audit

- [PASS] No future demand appears in features.
- [PASS] No future inventory appears in features.
- [PASS] Target creation occurs after feature creation conceptually.
- [PASS] Train/test dates do not overlap.
- [PASS] Encoders are fitted only on training data.
- [PASS] Imputation is fitted only on training data.
- [PASS] Stockout probability is not calculated from the target itself.
