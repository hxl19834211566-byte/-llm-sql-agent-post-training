# Side Routes

The final SQL mainline is still:

```text
SFT v3 + schema v2 + value-linking v1 + n20 candidates + rerank v15
= 450/500 = 90.0% execution accuracy
```

The routes below are included as controlled side experiments. They are useful for comparison and for showing the training/evaluation surface, but none of them replaces the final inference-time pipeline.

## Summary

| Route | Purpose | Best eval500 result | Decision |
| --- | --- | ---: | --- |
| DPO v2 | Preference tuning with narrower high-confidence SQL pairs | 353/500 = 70.6% | Not adopted |
| DPO v3 | Preference tuning with mixed high-confidence SQL error pairs | 353/500 = 70.6% | Not adopted |
| OPD v1 | Distill execution-correct SQL from a Qwen3-8B teacher | 346/500 = 69.2% | Not adopted |
| ORPO v1 | Lightweight preference optimization from SFT v3 | 363/500 = 72.6% | Not adopted |
| GRPO v1 | RL-style candidate-generation experiment with execution reward | 413/500 = 82.6% selected, 440/500 = 88.0% oracle | Not adopted |

## DPO

Goal: teach the model to prefer executable, schema-correct SQL over mistakes produced by SFT v3.

Data shape:

```text
chosen   = gold SQL
rejected = SFT v3 generated SQL with execution/schema/semantic errors
```

Scripts:

```text
scripts/prepare_sql_dpo_v1.py
scripts/prepare_sql_dpo_v2_from_v1.py
scripts/prepare_sql_dpo_v3_from_v1.py
scripts/validate_sql_dpo_v1.py
scripts/train_sql_dpo.py
```

Result:

```text
DPO v2 = 353/500 = 70.6%
DPO v3 = 353/500 = 70.6%
```

DPO was kept as a preference-learning comparison, but it did not exceed SFT v3 greedy or the final rerank pipeline.

## OPD

Goal: continue training the SFT v3 adapter on teacher-generated SQL that passes execution matching.

Flow:

```text
Spider train prompts
-> Qwen3-8B teacher SQL generation
-> execution match filter
-> continue-train SFT v3 student
```

Scripts:

```text
scripts/prepare_opd_teacher_prompts.py
scripts/build_opd_distill_data.py
scripts/train_sql_opd_completion.py
scripts/run_opd_v1.sh
```

Result:

```text
OPD v1 = 346/500 = 69.2%
```

The teacher-correct set was clean but small, and the teacher pipeline did not beat SFT v3 on the fixed evaluation.

## ORPO

Goal: test a lighter preference-optimization route using the DPO v3 high-confidence pairs.

Scripts:

```text
scripts/train_sql_orpo.py
scripts/run_orpo_v1.sh
```

Result:

```text
ORPO v1 = 363/500 = 72.6%
```

ORPO slightly beat SFT v3 greedy on the fixed eval, but it did not beat the schema/value-linking rerank pipeline.

## GRPO

Goal: test whether execution-reward policy optimization improves candidate generation.

Reward design:

```text
main reward: execution result matches gold SQL
auxiliary rewards: executable SQL, read-only shape, schema validity, intent/value hints
```

Scripts:

```text
scripts/train_sql_grpo.py
scripts/run_grpo_v1.sh
```

Result:

```text
GRPO v1 selected = 413/500 = 82.6%
GRPO v1 oracle   = 440/500 = 88.0%
```

GRPO improved over single-candidate generation, but it did not beat the final SFT v3 value-linking n20 + rerank v15 result.
