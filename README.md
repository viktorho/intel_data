# intel_data

## Project Pipeline

Our data pipeline transforms initial requirements into a validated dataset through four distinct phases:

- [x] **Phase 1: Requirement JSON**  \
  Gather user requirements and encode them in structured JSON to define scope, sources, metrics, and update frequency.
- [x] **Phase 2: Feature Blueprint & Crawl Plan**  \
  Draft a blueprint of desired features and plan the web crawling strategy, mapping each requirement to source endpoints.
- [ ] **Phase 3: Data Harvest & Structuring** *(In progress)*  \
  Implement crawlers to collect raw data and transform it into normalized tables ready for downstream processing.
- [ ] **Phase 4: Validation & Target-Table Assembly**  \
  Validate harvested data, enforce quality constraints, and assemble the final target table for analysis.

## Testing

Run the main entry point with a prompt to exercise the pipeline:

```bash
python3 src/main.py --prompt "a dataset for calculating average price for real estate from 2020-2025 in Ho Chi Minh city, daily update, include sources from gov.vn, metric VND, no need to exclude any source"
```

