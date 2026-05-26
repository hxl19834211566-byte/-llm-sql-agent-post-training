# Spider Schema v2 Logic Review

审查对象：`data/processed/spider_schema_v2.json`

审查时间：2026-05-11

## 结论

`spider_schema_v2.json` 对 Spider `tables.json` 的表达是严格一致的，可以作为 prompt / rerank / repair 的 schema 元数据输入使用。

但它不能被理解为“真实 SQLite DDL 的完全还原”。Spider `tables.json` 本身对一部分复合主键和少量外键记录不完整，因此 schema_v2 当前继承了这个边界：它忠实表达 Spider 元数据，而不是从 SQLite 文件反向抽取完整约束。

## 已验证内容

本地校验：

```bash
python scripts/validate_schema_v2.py \
  --schema-index data/processed/spider_schema_v2.json \
  --tables-json data/raw/spider_tables.json \
  --output logs/schema_v2_validation_local_rerun.json
```

结果：

```text
db: 166
tables: 876
columns: 4503
primary_keys: 781
foreign_keys: 795
issue_count: 166
```

本地的 166 个 issue 全部是 `missing_sqlite_path`，原因是本地缺少 `data/raw/spider/database/.../*.sqlite`，不是 schema_v2 与 `tables.json` 不一致。

额外结构检查结果：

```text
PK/FK 索引越界: 0
非法 FK pair: 0
复合主键表: 0
重复表名: 0
同表重复列名: 0
无 PK 数据库: 1, baseball_1
无 FK 数据库: 2, student_1, company_1
```

这说明当前 schema_v2 的结构字段、表名、列名、类型、PK/FK 索引映射与 Spider `tables.json` 闭包一致。

服务器端 SQLite 校验：

```bash
cd /root/project
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py310
python scripts/validate_schema_v2.py \
  --schema-index data/processed/spider_schema_v2.json \
  --tables-json data/raw/spider/tables.json \
  --output logs/schema_v2_validation.json
```

结果：

```text
db: 166
tables: 876
columns: 4503
schema_v2 primary_keys: 781
schema_v2 foreign_keys: 795
sqlite primary_keys: 949
sqlite foreign_keys: 798
issue_count: 85
```

85 个 issue 都来自 SQLite DDL 与 Spider `tables.json` 的差异：

```text
sqlite_primary_key_differs_from_tables_json: 80 db
sqlite_foreign_key_differs_from_tables_json: 5 db
```

主键差异全部是 SQLite 比 schema_v2 多，没有发现 schema_v2 额外写了不存在的主键：

```text
SQLite 多出的 PK 列总数: 168
schema_v2 额外 PK 列总数: 0
```

外键差异也全部是 SQLite 比 schema_v2 多：

```text
SQLite 多出的 FK 边总数: 5
schema_v2 额外 FK 边总数: 0
```

这说明 schema_v2 没有凭空制造约束，主要风险是继承了 Spider `tables.json` 对复合约束的欠表达。

## 主要风险

### 1. 复合主键被弱化

大量 SQLite 里的复合主键，在 Spider `tables.json` 中只保留了其中一个列。因此 schema_v2 会把这些表写成单列 `PRIMARY KEY`，而不是表级复合主键。

典型例子：

```text
college_2:
SQLite PK 多出 16 个列

hospital_1:
SQLite PK 多出 13 个列

academic:
SQLite PK 多出 7 个列
```

影响：

- 对生成 SQL 的直接执行通常影响较小，因为 Spider 预测 SQL 主要依赖列名、表名、连接边。
- 对 rerank / repair 的 join 判断有影响，因为复合关联表的“唯一性”和“中间表角色”会被低估。
- 如果后续用 schema_v2 做 schema-aware reward、SQL 静态校验、join path 打分，应补充 SQLite 级别的复合主键信息。

### 2. 少量外键缺失

SQLite 比 Spider `tables.json` 多 5 条外键：

```text
store_product:
store_product.Product_ID -> product.Product_ID

imdb:
tags.kid -> keyword.kid

baseball_1:
fielding_postseason.team_id -> player.team_id

restaurants:
LOCATION.RESTAURANT_ID -> RESTAURANT.RESTAURANT_ID

loan_1:
loan.cust_ID -> customer.Cust_ID
```

影响：

- 这些库的 join path 召回会少一条候选边。
- 对涉及这些边的问题，rerank 可能无法正确奖励真实 join。
- 这类问题适合在 repair/rerank 阶段补充，不建议直接把当前 schema_v2 当作完整数据库图。

### 3. `sqlite_sequence` 被当成普通表

有 3 个库在 Spider `tables.json` 中包含 SQLite 内部表 `sqlite_sequence`：

```text
world_1
soccer_1
store_1
```

schema_v2 会生成：

```sql
CREATE TABLE "sqlite_sequence" (
  "name" TEXT,
  "seq" TEXT
);
```

这段 SQL 作为文本 prompt 没问题，但如果拿去 SQLite 里执行，会失败：

```text
object name reserved for internal use: sqlite_sequence
```

影响：

- 作为 prompt 文本：低风险，但会引入一个无业务意义的系统表。
- 作为可执行 DDL：高风险，必须过滤。
- 作为 rerank schema 图：建议过滤，避免模型错误引用系统表。

### 4. 超长 schema

schema_v2 平均长度约 1433 字符，中位数约 896 字符，最长 9135 字符。

超过 4000 字符的库有 8 个：

```text
baseball_1: 9135
cre_Drama_Workshop_Groups: 6829
sakila_1: 4950
assets_maintenance: 4875
hospital_1: 4826
formula_1: 4394
cre_Theme_park: 4383
customer_deliveries: 4067
```

影响：

- 如果直接放入训练 prompt，长库会增加上下文负担。
- 对 SFT v4 不建议无筛选地全量替换为 schema_v2。
- 更适合先在 rerank / repair / self-check 中使用结构化字段，或对长库做 compact schema。

## 对当前项目的判断

当前 `schema_v2` 的设计目标是“在原 Spider schema 基础上增加 PK/FK 信息”，不是“从 SQLite 生成真实完整 DDL”。按这个目标，它是严谨的：

- `schema` 与 `schema_v2` 166/166 完全一致。
- `schema_version` 166/166 都是 `spider_schema_v2_pk_fk`。
- 表名、列名、列类型与 `tables.json` 一致。
- PK/FK 没有索引越界。
- schema 文本覆盖 CREATE TABLE、列、PRIMARY KEY、FOREIGN KEY。

但如果后续目标是提升 rerank / repair 的 join path 判断，当前版本仍有改进空间：

- 需要补充 SQLite 中真实复合主键。
- 需要补齐 SQLite 中多出的 5 条外键。
- 需要过滤 `sqlite_sequence`。
- 对超长 schema 做 compact 表达。

## 建议

短期不建议把 `spider_schema_v2.json` 直接作为新 SFT 训练的唯一变化点。原因是它会增加 prompt 长度，同时复合主键仍不完整，收益不一定稳定。

更合适的下一步：

1. 在 rerank / repair 中读取 `primary_keys`、`foreign_keys` 结构字段，先用于 join path 奖励、非法 join 惩罚、候选 SQL 自检。
2. 新增 `spider_schema_v2_sqlite_augmented.json` 或 `schema_graph_v1.json`，从 SQLite 补齐复合 PK 和少量 FK。
3. 过滤 `sqlite_sequence`，避免系统表进入 prompt 和 schema graph。
4. 对 8 个超长库生成 compact schema，只保留相关表、候选 SQL 涉及表、1-hop / 2-hop join 邻居。

## 保存的校验产物

```text
logs/schema_v2_validation_local.json
logs/schema_v2_validation_local_rerun.json
logs/schema_v2_validation_server.json
scripts/validate_schema_v2.py
```

