# DiffJury PR Reviewer

- 审查统一 diff，输出结构化问题、修复建议和测试建议
- GitHub `pull_request` webhook（`opened`、`reopened`、`synchronize`）
- `agentic` 运行模式：Lead、Security、Correctness/Reliability、Critic 多 Agent 协作
- SQLite 保存任务状态、执行轨迹和最终报告
- JSON API 与 Markdown 报告
- webhook HMAC-SHA256 签名校验，以及可选的 GitHub PR 评论回写
- Web 管理台、任务 Dashboard 与 Prometheus 指标
- 普通任务采用固定单轮多 Agent 协作（Lead 2 次、双 Worker 各 1 次、Critic 1 次，共 5 次 LLM 调用）；高风险任务最多允许一轮 Worker 返工
- LLM unified patch、AST/CST、沙箱前后测试对比与仅 Draft PR 的修复闭环
- PostgreSQL、Redis 生产模式
- 防腐化的知识治理：经验以条目形式保存，写入前去重把关，按 token 预算装配，按审查结果归因，定期整理和淘汰，所有变更经回放门禁后发布并可回滚
- 自研 Agent Runtime、持久化 checkpoint、执行预算与任务断点续跑
- 带 Tool Registry、参数 Schema 校验和结构化 Observation 的 Agent Loop
- 确认反馈的租户/仓库隔离记忆、检索与过期清理
- Redis Streams ACK、Worker 租约、指数退避重试和死信队列
- Webhook delivery 幂等、重放时间窗与评论 upsert
- 用户登录、RBAC、租户/仓库隔离和不可变管理审计
- 动态 Skill manifest 校验、签名校验和隔离进程沙箱
- 自动修复后的编译/测试门禁、灰度发布与影子流量
- OpenTelemetry Trace、Prometheus 指标和持久化告警

## 快速开始

项目使用 Python 3.11。先安装锁定范围内的运行依赖，并在同一个 PowerShell 窗口中配置本地管理员：

```powershell
python -m pip install -r requirements.txt

$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$env:DIFFJURY_AUTH_REQUIRED = 'true'
$env:DIFFJURY_AUTH_SECRET = [Convert]::ToBase64String($bytes)
$env:DIFFJURY_BOOTSTRAP_ADMIN_USERNAME = 'admin'
$env:DIFFJURY_BOOTSTRAP_ADMIN_PASSWORD = '<替换为至少 10 个字符的密码>'

python -m diffjury
```

不要直接使用示例占位符作为密码或密钥。环境变量只对当前 PowerShell 及其子进程生效；修改配置后需要停止并重新启动 DiffJury。服务可以在未配置模型时启动并提供健康检查，但提交审查前必须按下方“模型配置”章节配置模型。

Bootstrap 管理员只在用户名尚不存在时创建；已有同名用户的密码不会在重启时被覆盖。

服务默认监听 `127.0.0.1:8080`。启动后打开 `http://127.0.0.1:8080/`，前端会在业务 API 返回未授权状态后显示登录层。登录状态保存在当前浏览器的 `localStorage` 中；需要重新登录时可以点击退出，或清除站点数据。

API 调用需要先登录并携带 Bearer Token：

```powershell
$session = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/auth/login `
  -ContentType 'application/json' `
  -Body (@{username='admin'; password='<你的密码>'} | ConvertTo-Json)
$headers = @{Authorization="Bearer $($session.access_token)"}
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/reviews `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body (@{
    repository = 'demo/api'
    pull_request = 12
    mode = 'agentic'
    diff = "diff --git a/app.py b/app.py`n--- a/app.py`n+++ b/app.py`n@@ -1 +1,2 @@`n+password = 'secret'`n+eval(user_input)"
  } | ConvertTo-Json)
```

查询任务：

```powershell
Invoke-RestMethod -Headers $headers http://127.0.0.1:8080/v1/tasks/<task-id>
Invoke-WebRequest -Headers $headers http://127.0.0.1:8080/v1/tasks/<task-id>/report
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

## 模型配置

DeepSeek 官方 API（按 Token 计费）：

```powershell
$env:DIFFJURY_LLM_PROVIDER = 'deepseek'
$env:DIFFJURY_DEEPSEEK_API_KEY = '<deepseek-api-key>'
python -m diffjury
```

通过 OpenRouter 使用有速率限制、可用性可能变化的 DeepSeek 免费模型：

```powershell
$env:DIFFJURY_LLM_PROVIDER = 'openrouter-deepseek-free'
$env:DIFFJURY_OPENROUTER_API_KEY = '<openrouter-api-key>'
python -m diffjury
```

如果指定的免费 DeepSeek 版本下线，可将 `DIFFJURY_LLM_MODEL` 改为 OpenRouter 当前提供的其他 `:free` 模型，或把 Provider 改为 `openrouter-free` 让免费路由自动选择可用模型。

任意其他 OpenAI Chat Completions 兼容端点使用 `custom`：

```powershell
$env:DIFFJURY_LLM_PROVIDER = 'custom'
$env:DIFFJURY_LLM_BASE_URL = 'https://example.com/v1'
$env:DIFFJURY_LLM_API_KEY = '<token>'
$env:DIFFJURY_LLM_MODEL = '<model-name>'
```

密钥只通过环境变量读取，不要提交到仓库。

项目启动时会自动读取项目根目录的 `.env`，也兼容 `diffjury/.env`；系统环境变量优先于 `.env` 文件。推荐将以下内容写入根目录 `.env`（该文件已被 `.gitignore` 忽略）：

```env
DIFFJURY_LLM_PROVIDER=deepseek
DIFFJURY_DEEPSEEK_API_KEY=你的真实APIKey
```

## GitHub Webhook

项目使用“GitHub 仓库 Webhook + 公网转发 + fine-grained PAT”接收 PR 事件，不需要创建或安装 GitHub App：

```text
GitHub Pull request 事件
        │
        ▼
https://<公网域名>/webhooks/github
        │  公网转发
        ▼
http://127.0.0.1:8080/webhooks/github
        │
        ▼
DiffJury 创建异步审查任务
```

### 1. 配置 DiffJury

先生成一个 Webhook Secret，并根据需要配置 GitHub fine-grained personal access token：

```powershell
$webhookBytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($webhookBytes)
$env:DIFFJURY_GITHUB_WEBHOOK_SECRET = [Convert]::ToBase64String($webhookBytes)

# 私有仓库、PR 评论回写或自动修复需要；只审查公开仓库且不回写时可以不配置。
$env:DIFFJURY_GITHUB_TOKEN = '<GitHub fine-grained PAT>'

# 默认关闭。设为 true 后，审查完成时更新或创建 PR 评论。
$env:DIFFJURY_AUTO_POST_REVIEW = 'true'

python -m diffjury
```

Webhook Secret 用于验证 GitHub 请求头中的 HMAC-SHA256 签名，不能与登录用的 `DIFFJURY_AUTH_SECRET` 混用。Webhook 请求不携带管理台 Bearer Token；`/webhooks/github` 使用签名而不是用户登录进行认证。

fine-grained PAT 只授权需要接入的仓库，并按功能授予最小权限：

- 读取私有仓库 PR Diff：`Contents: Read`、`Pull requests: Read`；
- 回写审查评论：`Pull requests: Read and write`；
- 创建自动修复分支和提交：`Contents: Read and write`、`Pull requests: Read and write`。

只接收 Webhook 但不访问私有仓库、不回写评论且不执行自动修复时，可以不设置 PAT。密钥必须在启动 DiffJury 前设置，修改后需要重启服务。

### 2. 建立公网转发

GitHub 无法访问 `127.0.0.1`，需要把公网 HTTPS 地址转发到本地 `http://127.0.0.1:8080`。任选一种已安装的转发工具，例如：

```powershell
# Cloudflare Quick Tunnel
cloudflared tunnel --url http://127.0.0.1:8080

# 或 ngrok
ngrok http 8080
```

命令启动后会显示一个形如 `https://example.trycloudflare.com` 或 `https://example.ngrok-free.app` 的公网 HTTPS 地址。保持 DiffJury 和转发进程同时运行。临时公网地址通常会在转发工具重启后变化，变化后必须同步更新 GitHub Webhook 的 Payload URL。

上述快捷转发会把 8080 端口上的管理台和 API 一并暴露到公网，因此必须保持 `DIFFJURY_AUTH_REQUIRED=true`，并使用强管理员密码和随机 `DIFFJURY_AUTH_SECRET`。长期部署建议通过反向代理只公开 `/webhooks/github`（以及按需公开 `/health`），不要向公网暴露整个管理台。

### 3. 在 GitHub 仓库中添加 Webhook

进入目标仓库的 **Settings → Webhooks → Add webhook**，填写：

- **Payload URL**：`https://<公网域名>/webhooks/github`；
- **Content type**：`application/json`；
- **Secret**：与 `DIFFJURY_GITHUB_WEBHOOK_SECRET` 完全相同；
- **SSL verification**：保持启用；
- **Which events would you like to trigger this webhook?**：选择 **Let me select individual events**，只勾选 **Pull requests**；
- **Active**：保持勾选。

DiffJury 会处理 `opened`、`reopened` 和 `synchronize` 三种 PR 动作；其他 `pull_request` 动作会正常接收但被忽略。服务会根据 payload 中的 `diff_url` 下载 Diff，并异步创建审查任务。

### 4. 验证连接

先确认本地服务和公网地址都能访问健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod https://<公网域名>/health
```

然后新建 PR、重新打开 PR，或向 PR 推送一次提交。在 GitHub 的 **Settings → Webhooks → Recent Deliveries** 中应看到 `/webhooks/github` 返回 `202`；管理台的任务中心随后会出现对应审查任务。如果失败，优先检查公网转发进程是否仍在运行、Payload URL 是否包含 `/webhooks/github`、Secret 是否一致，以及 PAT 是否有目标仓库权限。

默认只在管理台保存结果。只有 `DIFFJURY_AUTO_POST_REVIEW=true` 时才会向 PR 回写评论。

自动修复只覆盖可确定安全的规则，例如调试输出、`shell=True` 和硬编码 Python 凭据；结果始终提交到新的 `diffjury/fix-pr-*` 分支，不直接修改源分支。

## 知识治理（防止上下文腐化）

会自我学习的 Agent 往往越用越差：每次反馈都往 prompt 或 Skill 里追加几句，重复的、过时的、互相矛盾的、空泛的内容越积越多，单次调用的上下文持续膨胀，相关规则被淹没。DiffJury 不再维护“越写越长的提示词”，而是把学到的经验当成需要治理的数据：

| 环节 | 做法 | 代码 |
|---|---|---|
| 条目化 | 每条经验是独立的 `KnowledgeItem`：角色、触发条件（`cwe:`/`path:`/`lang:`/关键词）、证据、状态、有效/有害计数、曝光与引用次数 | `knowledge.py` |
| 写入把关 | 没有证据的拒收；规则超过 60 tokens 先压缩，压缩后仍超长则拒收；完全重复的直接点赞；近似条目（TF-IDF 相似度 ≥ 0.3 的前 5 个）交给 LLM 裁决 ADD / MERGE / UPVOTE / REJECT，合并结果同样受长度上限约束；没有近似条目的新经验也要先过一遍筛查，空泛建议直接拒收；与已归档条目高度相似时恢复旧条目，不新建；新条目一律进入试用期 | `admission.py` |
| 预算装配 | 每次模型调用按 `相关度 × 效用 ÷ token 数` 贪心装入固定预算（默认每角色 600 tokens），固定条目优先；Lead 只看到最相关的 8 个 Skill；Worker 选中的 Skill 正文共享 3000 tokens 预算，单个 Skill 正文不超过 1500 tokens；装不下的条目记为 dropped | `assembler.py` |
| 效果归因 | Worker 在 finding 的 `applied_rules` 中注明用到的条目（伪造的 id 会被丢弃）；被发布的 finding 给对应条目记有效，被 Critic/Lead/门禁拒绝的记有害 0.5；模型常常用了经验却不写 id，所以已发布的 finding 若命中某条已注入条目的 `cwe:` 或关键词触发条件，也给该条目记 0.5 的隐式有效（没有内容触发条件的条目拿不到这份分）；人工标记误报记有害 2、确认有效记有效 1；计数按 30 天半衰期衰减 | `knowledge_service.py` |
| 试用期与灰度 | 新条目只在灰度任务（默认 20%）中可见，曝光够多、效用够高后才申请晋升 | `lifecycle.py`、`service.py` |
| 淘汰到冷存储 | 按“最近一次被引用”而不是“最近一次被注入”判断活跃度：多次导致误报、长期曝光却没人引用、长时间闲置的条目会被归档；每个角色的在用 token 有硬上限，超出时先淘汰试用期和低效用条目。归档条目不再进入提示词，但可以恢复 | `lifecycle.py` |
| Dream 整理 | 按相似度聚类，每次只让 LLM 处理一个小簇：合并重复条目（计数和证据一并继承）、删除被推翻或空泛的条目；没有模型时只合并几乎相同的条目，不会丢失独立的经验 | `dream.py` |
| 统一门禁 | 晋升、合并、删除都会生成候选快照，在验证集上与当前快照做配对回放。F1、高风险召回、干净样本准确率都不能出现“有把握的下降”：配对 bootstrap 95% 置信区间上界不能低于 −0.02；验证集较小时单个 finding 就能让 F1 波动几个点，所以仅凭点估计下降不拒绝，但点估计下降超过 0.10 时仍然拒绝；Dream 整理还要求平均单次调用提示词 tokens 不能增加。未通过的变更不会写入，只记一条“门禁拒绝”的发布记录 | `release_gate.py` |
| 发布与回滚 | 每次通过门禁的变更都是一个版本快照，可以一键回滚到任何非拒绝版本 | `knowledge.py`、`knowledge_service.py` |
| 记忆治理 | 情景记忆超过 90 天过期，并按仓库限制情景记忆和语义记忆的总条数 | `memory.py` |

管理台的“知识治理”页面展示条目、效用、发布记录和事件流；`/metrics` 提供 `diffjury_knowledge_items{status}`、`diffjury_knowledge_live_tokens`、`diffjury_knowledge_active_rule_tokens{role}`、准入决定计数、发布与淘汰计数，以及 `instruction_tokens_total / instructed_model_calls_total`，后两者相除就是平均单次调用的指令 tokens。

整理任务会每 50 次审查在后台自动运行一次，也可以通过 `POST /v1/knowledge/maintain` 手动触发。门禁默认从 `evaluation_data/pr_diff_100.jsonl` 的 validation 划分中按固定方式抽取 40 个样本；没有配置模型时门禁关闭，变更会直接应用，并在发布记录中标为未经门禁。

### 腐化压力测试

`scripts/run_rot_stress_test.py` 用同一串学习事件对比两种做法：一种把每条经验直接追加进 prompt、把每个流程沉淀成新 Skill；另一种走上面的完整治理流程。事件取自训练仓库 01–06，构成为：漏报 50%（其中五分之一沉淀为 Skill）、误报 20%、空泛噪声 20%、与已有经验矛盾 10%。脚本每 25 个事件测量一次提示词规模，并在检查点用独立的 holdout 仓库 09–10 回放评估审查质量：holdout 默认重复回放 3 次后合并（`--holdout-repeats`），再对两种做法做配对 bootstrap，给出差值的 95% 置信区间。中断后用同一条命令重新运行即可续跑。

```powershell
python scripts/run_rot_stress_test.py --out evaluation_results/rot_stress          # 使用 .env 中的模型，耗时数小时
python scripts/run_rot_stress_test.py --fake --events 60 --out evaluation_results/rot_smoke   # 离线冒烟测试
```

结果写入 `report.md`、`report.json`、`measurements.csv`，以及 `instruction_tokens_per_call.svg`、`live_knowledge_tokens.svg`、`holdout_f1.svg` 三张图；每个检查点的逐例结果保存在 `holdout_rows/`。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 健康检查 |
| `POST` | `/v1/auth/login` | 登录并获取租户绑定的短期 Bearer Token |
| `POST` | `/v1/reviews` | 创建同步审查任务 |
| `POST` | `/v1/reviews?async=true` | 创建异步审查任务 |
| `GET` | `/v1/tasks/{id}` | 获取状态、轨迹和报告 |
| `GET` | `/v1/tasks/{id}/report` | 获取 Markdown 报告 |
| `GET` | `/v1/tasks/{id}/feedback` | 获取该已完成任务的反馈历史 |
| `POST` | `/v1/tasks/{id}/fix` | 创建自动修复分支和提交 |
| `POST` | `/v1/tasks/{id}/feedback` | 回流误报、漏报或坏修复 |
| `POST` | `/v1/tasks/{id}/cancel` | 请求取消任务 |
| `POST` | `/v1/tasks/{id}/resume` | 从最近 checkpoint 续跑任务 |
| `POST` | `/webhooks/github` | 接收 GitHub PR webhook |
| `POST` | `/v1/skills/reload` | 动态重新加载 Skill |
| `GET/POST` | `/v1/evaluation/cases` | 查询或增加版本化评测样本 |
| `GET` | `/v1/knowledge/items?status=active,probation&kind=rule` | 查询知识条目（含效用分） |
| `POST` | `/v1/knowledge/items` | 手动提交一条经验，同样经过写入把关 |
| `GET` | `/v1/knowledge/items/{id}` | 查询单个条目及其事件 |
| `POST` | `/v1/knowledge/items/{id}/pin`、`unpin`、`archive`、`restore` | 固定、取消固定、归档或恢复条目 |
| `GET` | `/v1/knowledge/stats` | 条目数量、在用 tokens、各角色占用和预算 |
| `GET` | `/v1/knowledge/events` | 准入、合并、淘汰、晋升、归因等事件流 |
| `GET` | `/v1/knowledge/releases` | 知识发布记录与门禁结果 |
| `POST` | `/v1/knowledge/maintain` | 在后台运行一次淘汰、Dream 整理和晋升 |
| `POST` | `/v1/knowledge/releases/{version}/rollback` | 回滚到指定发布版本 |
| `GET` | `/metrics` | Prometheus 文本指标 |
| `GET` | `/api/alerts` | 查询租户告警 |
| `GET` | `/api/audit` | 查询租户审计日志 |
| `GET` | `/api/queue/dead-letters` | 查询死信任务 |
| `POST` | `/v1/queue/dead-letters/replay` | 重放死信任务 |
| `GET/POST` | `/api/deployments/knowledge`、`/v1/deployments/knowledge` | 查询或配置灰度/影子发布 |

`POST /v1/reviews` 的 `diff` 最大默认 1 MiB；单任务默认最多 8 步、120 秒。可通过环境变量调整，详见 `.env.example`。

## 架构

```text
HTTP / GitHub Webhook
        │
        ▼
 ReviewService ── TaskStore(SQLite / PostgreSQL)
        │
        ▼
        ReviewHarness (DiffJury Runtime / checkpoint / resume / budget / trace)
        │
        ├── DiffParser
        ├── Redis Streams / ACK / lease / retry / DLQ
        └── Agentic Lead/Workers
              ├── Lead：一次任务分配并判定风险；高风险时最多请求一轮返工；最终综合
              ├── Security Worker：输入/权限/敏感数据/危险调用链
              ├── Correctness/Reliability Worker：状态/异常/并发/资源/兼容性
              ├── Critic Worker：一次性盲审、反例与证据挑战
              └── Gates：格式、证据、置信度和发布门禁
        │
        ▼
 KnowledgeGovernor
        ├── AdmissionGate：证据 / 长度 / 去重 / LLM 裁决 → 试用期
        ├── ContextAssembler：相关度 × 效用 ÷ tokens，固定预算装配
        ├── Attribution：applied_rules → 有效 / 有害，按半衰期衰减
        ├── Lifecycle：晋升、淘汰到冷存储、按角色限制容量
        ├── Dream：按簇合并与清理
        └── ReleaseGate：配对回放 + bootstrap，通过后才发布，可回滚
```

## Skills 的定位与编写方式

`skills/` 下的目录是 Agent Skill 包，不是工具连接器目录。它们目前以审查领域为中心，是因为本项目的产品目标是代码审查；Skill 的职责是把某个领域的判断标准、调查步骤和输出约束注入模型，而不是自己执行 SQL、调用 GitHub 或创建网络连接。

一个 Skill 包至少包含一个 `SKILL.md`：

```text
skills/<skill-name>/
└── SKILL.md                 # YAML 元数据 + 完整运行指令
```

`SKILL.md` 的 frontmatter 由运行时读取：

- `name`：必须与目录名一致，且只使用小写字母、数字和连字符；
- `description`：只用于目录发现和路由，应该短而明确地说明何时启用；
- `allowed-tools`：该 Skill 允许 worker 使用的工具名集合。它是权限声明，不会安装或连接工具；最终权限还会与角色权限取交集；

frontmatter 后的 Markdown 正文才是 Skill 的核心。正文会在 Lead 选中 Skill 后注入对应 worker 的上下文，因此应写成可执行的审查协议，而不是几条标签式提示词。推荐至少包含：任务边界、适用/不适用范围、逐步调查流程、边界条件清单、证据与严重性标准、误报排除规则、修复与验证要求，以及稳定的输出字段。

Skill 可以附带文本资源（例如规范摘录、查询模板或测试清单）。运行时会把资源列入 Skill 包，并仅为已选中的 Skill 注册 `read_skill_resource`；资源不能通过路径穿越或符号链接逃逸出包目录。真正的工具连接由运行时的 `RepositoryToolSuite`、外部集成或插件提供，Skill 只声明“可以使用哪些已经存在的工具”。

运行时流程如下：

```text
磁盘扫描 SKILL.md
        ↓ 只暴露 name/description 给 Lead
Lead 选择相关 Skill
        ↓ 注入 instructions，并收窄 allowed-tools
Worker 使用仓库工具取证
        ↓
Finding gate 校验格式、证据、置信度和发布条件
```

因此，把 Skill 内容直接硬编码到全局上下文并不能完全等价：那会让每个任务都携带全部领域规则，增加上下文长度和相互干扰；Skill 机制可以按任务选择、限制工具权限、热加载版本。内置 Skill 会同步为知识条目（默认固定，不会被淘汰），以便统计它们的曝光和引用；从反馈中学到的 Skill 与规则一样经过写入把关、试用期、预算装配和淘汰。对于只有一两句、没有独立选择价值的规则，直接放在全局 prompt 更合适；只有当一组规则有清晰边界、需要按场景启用或需要独立演进时，才应该做成 Skill。

本仓库的内置 Skill 已按上述结构补充了完整的 Mission、Review procedure、Checklist、Evidence/severity、Remediation/verification 和 Output contract。新增 Skill 时可复制其中一个目录，再根据领域替换这些章节，并用 `POST /v1/skills/reload` 让新任务加载最新版本。
