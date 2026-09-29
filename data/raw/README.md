# StockSense Hackathon Dataset

Generated to match the IntelliData 2026 StockSense challenge document.

## Files
- transactions.csv
- products.csv
- stores.csv
- inventory.csv
- external_factors.csv

## Grain
The intended master analytics grain is:
ONE ROW = ONE DATE x ONE STORE x ONE PRODUCT

## Intentional data-quality traps
- Missing external values can be introduced during preprocessing/merge handling.
- Duplicate transaction rows are included.
- Product category capitalization inconsistencies are included.
- A few impossible transaction quantities (negative) are included.
- A small number of inventory closing values intentionally mismatch the inventory arithmetic.
- A small set of products has only 5 days of history to simulate sparse history.

## Important
The data is synthetic and created for hackathon practice. It is not real company data.

## Suggested merge
Aggregate transactions to daily Store x Product, then merge products, stores, inventory and external factors.

## Required modelling targets
- next_7_day_demand
- stockout_flag / stockout_probability

## Intended workflow
Raw Data -> Clean & Merge -> EDA -> Statistics -> Features -> ML Models ->
Explainability -> Recommendation -> Visualization / Prototype
