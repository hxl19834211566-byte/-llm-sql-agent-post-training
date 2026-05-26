# Documentation Index

Reports and runbooks for the Text-to-SQL agent project.

## Core Reports

- [Mainline report](mainline_report.md): concise final result, pipeline, and adoption rationale.
- [Final pipeline runbook](final_pipeline_runbook.md): commands for rebuilding schema v2, generating value-linking prompts, generating candidates, and running rerank v15.
- [Side routes](side_routes.md): DPO, OPD, ORPO, and GRPO experiments, scripts, results, and adoption decisions.

## Main Improvements

- [Value linking v1 report](value_linking_v1_report.md): how SQLite value hints improved the candidate oracle upper bound.
- [Value linking v1 runbook](value_linking_v1_runbook.md): commands for generating value-linking eval files and candidates.
- [Rerank repair report](rerank_repair_report.md): rerank/repair changes that closed the selected-vs-oracle gap.
- [Schema v2 logic review](schema_v2_logic_review.md): schema prompt construction and validation notes.

## Agent Demo

- [Tool-calling demo report](tool_calling_demo_report.md): offline and online demo flow for `search_schema`, `validate_sql`, and `run_sql`.

Generated PDFs and raw log dumps are kept out of the tracked documentation set.
