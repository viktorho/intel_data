# intel_data

## Project Pipeline
This project was born from the frustration of endless, noisy web scrapes and the wasted hours spent cleaning HTML before any real insights could emerge. By combining a declarative plan-driven crawler, configurable content filters, and lightweight extractors, our goal is to hand off perfectly structured JSON—complete with quality scores and audit logs—directly into downstream LLMs or BI tools. In short, we turn “data chaos” into “machine-ready” answers at scale, slashing manual effort and keeping token costs in check.

<div align="center">
  <img width="640" height="482" alt="image" src="https://github.com/user-attachments/assets/9561ef87-90ff-4c56-aebf-c838b9731eb3" />
</div>
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

