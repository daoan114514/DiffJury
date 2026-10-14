import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


_DOTENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_dotenv(paths: Optional[Iterable[str]] = None) -> None:
    """Load local dotenv files without overriding real process environment values.

    The project-root file has priority over ``diffjury/.env``.  This allows the
    latter to remain compatible with existing local setups while keeping the
    conventional root-level ``.env`` as the recommended location.
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(package_dir)
    candidates = list(paths) if paths is not None else [
        os.path.join(project_root, ".env"),
        os.path.join(package_dir, ".env"),
    ]
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8-sig") as handle:
                lines = handle.readlines()
        except OSError:
            continue
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            if not _DOTENV_KEY.fullmatch(key):
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key, value)


load_dotenv()


def _int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError("%s must be positive" % name)
    return value


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _non_negative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError("%s must be non-negative" % name)
    return value


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    db_path: str
    max_diff_bytes: int
    max_steps: int
    timeout_seconds: int
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    github_webhook_secret: str
    github_token: str
    auto_post_review: bool
    database_url: str = ""
    redis_url: str = ""
    async_workers: int = 2
    memory_enabled: bool = True
    memory_recall_limit: int = 6
    memory_working_ttl_seconds: int = 86400
    skills_dir: str = "skills"
    github_app_id: str = ""
    github_app_slug: str = ""
    github_private_key_path: str = ""
    public_base_url: str = "http://127.0.0.1:8080"
    llm_provider: str = "local"
    deepseek_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_site_url: str = ""
    openrouter_app_name: str = "DiffJury"
    auth_required: bool = False
    auth_secret: str = ""
    bootstrap_admin_username: str = ""
    bootstrap_admin_password: str = ""
    default_tenant_id: str = "default"
    session_ttl_seconds: int = 3600
    webhook_max_age_seconds: int = 600
    queue_max_attempts: int = 3
    queue_lease_seconds: int = 60
    repair_test_command: str = ""
    repair_verify_timeout_seconds: int = 120
    otel_endpoint: str = ""
    otel_service_name: str = "diffjury"
    alert_failure_rate: float = 0.20
    alert_min_samples: int = 10
    alert_window_seconds: int = 900
    alert_webhook_url: str = ""
    alert_smtp_host: str = ""
    alert_email_to: str = ""
    agent_token_budget: int = 8000
    agent_time_budget_seconds: int = 60
    agent_context_window_tokens: int = 32768
    agent_context_input_tokens: int = 20000
    context_diff_token_budget: int = 12000
    context_observation_token_budget: int = 4000
    context_recent_observations: int = 2
    context_map_chunk_tokens: int = 3000
    enabled_agents: str = "lead,security,correctness-reliability,critic"
    llm_input_cost_per_million: float = 0.0
    llm_output_cost_per_million: float = 0.0
    knowledge_enabled: bool = True
    knowledge_rule_budget_tokens: int = 600
    knowledge_skill_budget_tokens: int = 3000
    knowledge_skill_body_cap_tokens: int = 1500
    knowledge_skill_catalog_limit: int = 8
    knowledge_rule_max_tokens: int = 60
    knowledge_similarity_threshold: float = 0.3
    knowledge_duplicate_threshold: float = 0.85
    knowledge_half_life_days: float = 30.0
    knowledge_pin_disk_skills: bool = True
    knowledge_canary_percent: int = 20
    knowledge_maintain_every_reviews: int = 50
    knowledge_dream_token_threshold: int = 4000
    knowledge_capacity_tokens_per_role: int = 2400
    knowledge_gate_dataset: str = ""
    knowledge_gate_splits: str = "validation"
    knowledge_gate_max_cases: int = 40
    knowledge_gate_margin: float = 0.02
    knowledge_gate_max_drop: float = 0.10
    knowledge_gate_workers: int = 4
    memory_episodic_ttl_days: float = 90.0
    memory_episodic_cap: int = 300
    memory_semantic_cap: int = 500

    def resolved_llm(self) -> Dict[str, object]:
        """Resolve a named provider to the existing OpenAI-compatible transport."""
        provider = self.llm_provider.strip().lower()
        if provider in {"", "local", "none"}:
            if self.llm_base_url or self.llm_api_key or self.llm_model:
                provider = "custom"
            else:
                return {}

        if provider == "deepseek":
            api_key = self.deepseek_api_key or self.llm_api_key
            if not api_key:
                raise ValueError("DeepSeek requires DIFFJURY_DEEPSEEK_API_KEY")
            return {
                "provider": "deepseek",
                "base_url": self.llm_base_url or "https://api.deepseek.com",
                "api_key": api_key,
                "model": self.llm_model or "deepseek-v4-flash",
                "headers": {},
            }

        if provider in {"openrouter-deepseek-free", "openrouter_deepseek_free"}:
            api_key = self.openrouter_api_key or self.llm_api_key
            if not api_key:
                raise ValueError("OpenRouter requires DIFFJURY_OPENROUTER_API_KEY")
            headers = {}
            if self.openrouter_site_url:
                headers["HTTP-Referer"] = self.openrouter_site_url
            if self.openrouter_app_name:
                headers["X-Title"] = self.openrouter_app_name
            return {
                "provider": "openrouter-deepseek-free",
                "base_url": self.llm_base_url or "https://openrouter.ai/api/v1",
                "api_key": api_key,
                "model": self.llm_model or "deepseek/deepseek-chat-v3-0324:free",
                "headers": headers,
            }

        if provider == "openrouter-free":
            api_key = self.openrouter_api_key or self.llm_api_key
            if not api_key:
                raise ValueError("OpenRouter requires DIFFJURY_OPENROUTER_API_KEY")
            headers = {}
            if self.openrouter_site_url:
                headers["HTTP-Referer"] = self.openrouter_site_url
            if self.openrouter_app_name:
                headers["X-Title"] = self.openrouter_app_name
            return {
                "provider": "openrouter-free",
                "base_url": self.llm_base_url or "https://openrouter.ai/api/v1",
                "api_key": api_key,
                "model": self.llm_model or "openrouter/free",
                "headers": headers,
            }

        if provider == "custom":
            if not (self.llm_base_url and self.llm_api_key and self.llm_model):
                raise ValueError(
                    "Custom LLM requires DIFFJURY_LLM_BASE_URL, "
                    "DIFFJURY_LLM_API_KEY and DIFFJURY_LLM_MODEL"
                )
            return {
                "provider": "custom",
                "base_url": self.llm_base_url,
                "api_key": self.llm_api_key,
                "model": self.llm_model,
                "headers": {},
            }
        raise ValueError("unsupported DIFFJURY_LLM_PROVIDER: %s" % self.llm_provider)

    def validate(self) -> None:
        for name in ("knowledge_similarity_threshold", "knowledge_duplicate_threshold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError("DIFFJURY_%s must be between 0 and 1" % name.upper())
        if self.knowledge_similarity_threshold > self.knowledge_duplicate_threshold:
            raise ValueError(
                "DIFFJURY_KNOWLEDGE_SIMILARITY_THRESHOLD cannot exceed "
                "DIFFJURY_KNOWLEDGE_DUPLICATE_THRESHOLD"
            )
        if not 0 <= self.knowledge_canary_percent <= 100:
            raise ValueError("DIFFJURY_KNOWLEDGE_CANARY_PERCENT must be between 0 and 100")
        if not 0.0 <= self.knowledge_gate_margin <= 1.0:
            raise ValueError("DIFFJURY_KNOWLEDGE_GATE_MARGIN must be between 0 and 1")
        if not 0.0 <= self.knowledge_gate_max_drop <= 1.0:
            raise ValueError("DIFFJURY_KNOWLEDGE_GATE_MAX_DROP must be between 0 and 1")
        if self.knowledge_skill_body_cap_tokens > self.knowledge_skill_budget_tokens:
            raise ValueError(
                "DIFFJURY_KNOWLEDGE_SKILL_BODY_CAP_TOKENS cannot exceed "
                "DIFFJURY_KNOWLEDGE_SKILL_BUDGET_TOKENS"
            )
        if self.auth_required and len(self.auth_secret.encode("utf-8")) < 32:
            raise ValueError(
                "DIFFJURY_AUTH_SECRET must contain at least 32 bytes when authentication is enabled"
            )
        if bool(self.bootstrap_admin_username) != bool(self.bootstrap_admin_password):
            raise ValueError("bootstrap admin username and password must be configured together")
        if not 0.0 <= self.alert_failure_rate <= 1.0:
            raise ValueError("DIFFJURY_ALERT_FAILURE_RATE must be between 0 and 1")
        if self.llm_input_cost_per_million < 0 or self.llm_output_cost_per_million < 0:
            raise ValueError("LLM token prices cannot be negative")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=os.getenv("DIFFJURY_HOST", "127.0.0.1"),
            port=_int("DIFFJURY_PORT", 8080),
            db_path=os.getenv("DIFFJURY_DB_PATH", "diffjury.db"),
            max_diff_bytes=_int("DIFFJURY_MAX_DIFF_BYTES", 1024 * 1024),
            max_steps=_int("DIFFJURY_MAX_STEPS", 8),
            timeout_seconds=_int("DIFFJURY_TIMEOUT_SECONDS", 120),
            llm_base_url=os.getenv("DIFFJURY_LLM_BASE_URL", "").rstrip("/"),
            llm_api_key=os.getenv("DIFFJURY_LLM_API_KEY", ""),
            llm_model=os.getenv("DIFFJURY_LLM_MODEL", ""),
            github_webhook_secret=os.getenv("DIFFJURY_GITHUB_WEBHOOK_SECRET", ""),
            github_token=os.getenv("DIFFJURY_GITHUB_TOKEN", ""),
            auto_post_review=_bool("DIFFJURY_AUTO_POST_REVIEW"),
            database_url=os.getenv("DIFFJURY_DATABASE_URL", ""),
            redis_url=os.getenv("DIFFJURY_REDIS_URL", ""),
            async_workers=_int("DIFFJURY_ASYNC_WORKERS", 2),
            memory_enabled=_bool("DIFFJURY_MEMORY_ENABLED", True),
            memory_recall_limit=_int("DIFFJURY_MEMORY_RECALL_LIMIT", 6),
            memory_working_ttl_seconds=_int(
                "DIFFJURY_MEMORY_WORKING_TTL_SECONDS", 86400
            ),
            skills_dir=os.getenv("DIFFJURY_SKILLS_DIR", "skills"),
            github_app_id=os.getenv("DIFFJURY_GITHUB_APP_ID", ""),
            github_app_slug=os.getenv("DIFFJURY_GITHUB_APP_SLUG", ""),
            github_private_key_path=os.getenv("DIFFJURY_GITHUB_PRIVATE_KEY_PATH", ""),
            public_base_url=os.getenv("DIFFJURY_PUBLIC_BASE_URL", "http://127.0.0.1:8080").rstrip("/"),
            llm_provider=os.getenv("DIFFJURY_LLM_PROVIDER", "local"),
            deepseek_api_key=os.getenv("DIFFJURY_DEEPSEEK_API_KEY", ""),
            openrouter_api_key=os.getenv("DIFFJURY_OPENROUTER_API_KEY", ""),
            openrouter_site_url=os.getenv("DIFFJURY_OPENROUTER_SITE_URL", ""),
            openrouter_app_name=os.getenv("DIFFJURY_OPENROUTER_APP_NAME", "DiffJury"),
            auth_required=_bool("DIFFJURY_AUTH_REQUIRED", False),
            auth_secret=os.getenv("DIFFJURY_AUTH_SECRET", ""),
            bootstrap_admin_username=os.getenv("DIFFJURY_BOOTSTRAP_ADMIN_USERNAME", ""),
            bootstrap_admin_password=os.getenv("DIFFJURY_BOOTSTRAP_ADMIN_PASSWORD", ""),
            default_tenant_id=os.getenv("DIFFJURY_DEFAULT_TENANT_ID", "default"),
            session_ttl_seconds=_int("DIFFJURY_SESSION_TTL_SECONDS", 3600),
            webhook_max_age_seconds=_int("DIFFJURY_WEBHOOK_MAX_AGE_SECONDS", 600),
            queue_max_attempts=_int("DIFFJURY_QUEUE_MAX_ATTEMPTS", 3),
            queue_lease_seconds=_int("DIFFJURY_QUEUE_LEASE_SECONDS", 60),
            repair_test_command=os.getenv("DIFFJURY_REPAIR_TEST_COMMAND", ""),
            repair_verify_timeout_seconds=_int("DIFFJURY_REPAIR_VERIFY_TIMEOUT_SECONDS", 120),
            otel_endpoint=os.getenv("DIFFJURY_OTEL_ENDPOINT", ""),
            otel_service_name=os.getenv("DIFFJURY_OTEL_SERVICE_NAME", "diffjury"),
            alert_failure_rate=float(os.getenv("DIFFJURY_ALERT_FAILURE_RATE", "0.20")),
            alert_min_samples=_int("DIFFJURY_ALERT_MIN_SAMPLES", 10),
            alert_window_seconds=_int("DIFFJURY_ALERT_WINDOW_SECONDS", 900),
            alert_webhook_url=os.getenv("DIFFJURY_ALERT_WEBHOOK_URL", ""),
            alert_smtp_host=os.getenv("DIFFJURY_ALERT_SMTP_HOST", ""),
            alert_email_to=os.getenv("DIFFJURY_ALERT_EMAIL_TO", ""),
            agent_token_budget=_int("DIFFJURY_AGENT_TOKEN_BUDGET", 8000),
            agent_time_budget_seconds=_int("DIFFJURY_AGENT_TIME_BUDGET_SECONDS", 60),
            agent_context_window_tokens=_int(
                "DIFFJURY_AGENT_CONTEXT_WINDOW_TOKENS", 32768
            ),
            agent_context_input_tokens=_int(
                "DIFFJURY_AGENT_CONTEXT_INPUT_TOKENS", 20000
            ),
            context_diff_token_budget=_int(
                "DIFFJURY_CONTEXT_DIFF_TOKEN_BUDGET", 12000
            ),
            context_observation_token_budget=_int(
                "DIFFJURY_CONTEXT_OBSERVATION_TOKEN_BUDGET", 4000
            ),
            context_recent_observations=_non_negative_int(
                "DIFFJURY_CONTEXT_RECENT_OBSERVATIONS", 2
            ),
            context_map_chunk_tokens=_int(
                "DIFFJURY_CONTEXT_MAP_CHUNK_TOKENS", 3000
            ),
            enabled_agents=os.getenv(
                "DIFFJURY_ENABLED_AGENTS",
                "lead,security,correctness-reliability,critic",
            ),
            llm_input_cost_per_million=float(
                os.getenv("DIFFJURY_LLM_INPUT_COST_PER_MILLION", "0")
            ),
            llm_output_cost_per_million=float(
                os.getenv("DIFFJURY_LLM_OUTPUT_COST_PER_MILLION", "0")
            ),
            knowledge_enabled=_bool("DIFFJURY_KNOWLEDGE_ENABLED", True),
            knowledge_rule_budget_tokens=_non_negative_int(
                "DIFFJURY_KNOWLEDGE_RULE_BUDGET_TOKENS", 600
            ),
            knowledge_skill_budget_tokens=_non_negative_int(
                "DIFFJURY_KNOWLEDGE_SKILL_BUDGET_TOKENS", 3000
            ),
            knowledge_skill_body_cap_tokens=_int(
                "DIFFJURY_KNOWLEDGE_SKILL_BODY_CAP_TOKENS", 1500
            ),
            knowledge_skill_catalog_limit=_int("DIFFJURY_KNOWLEDGE_SKILL_CATALOG_LIMIT", 8),
            knowledge_rule_max_tokens=_int("DIFFJURY_KNOWLEDGE_RULE_MAX_TOKENS", 60),
            knowledge_similarity_threshold=float(
                os.getenv("DIFFJURY_KNOWLEDGE_SIMILARITY_THRESHOLD", "0.3")
            ),
            knowledge_duplicate_threshold=float(
                os.getenv("DIFFJURY_KNOWLEDGE_DUPLICATE_THRESHOLD", "0.85")
            ),
            knowledge_half_life_days=float(
                os.getenv("DIFFJURY_KNOWLEDGE_HALF_LIFE_DAYS", "30")
            ),
            knowledge_pin_disk_skills=_bool("DIFFJURY_KNOWLEDGE_PIN_DISK_SKILLS", True),
            knowledge_canary_percent=_non_negative_int("DIFFJURY_KNOWLEDGE_CANARY_PERCENT", 20),
            knowledge_maintain_every_reviews=_non_negative_int(
                "DIFFJURY_KNOWLEDGE_MAINTAIN_EVERY_REVIEWS", 50
            ),
            knowledge_dream_token_threshold=_non_negative_int(
                "DIFFJURY_KNOWLEDGE_DREAM_TOKEN_THRESHOLD", 4000
            ),
            knowledge_capacity_tokens_per_role=_int(
                "DIFFJURY_KNOWLEDGE_CAPACITY_TOKENS_PER_ROLE", 2400
            ),
            knowledge_gate_dataset=os.getenv("DIFFJURY_KNOWLEDGE_GATE_DATASET", ""),
            knowledge_gate_splits=os.getenv("DIFFJURY_KNOWLEDGE_GATE_SPLITS", "validation"),
            knowledge_gate_max_cases=_non_negative_int("DIFFJURY_KNOWLEDGE_GATE_MAX_CASES", 40),
            knowledge_gate_margin=float(os.getenv("DIFFJURY_KNOWLEDGE_GATE_MARGIN", "0.02")),
            knowledge_gate_max_drop=float(os.getenv("DIFFJURY_KNOWLEDGE_GATE_MAX_DROP", "0.10")),
            knowledge_gate_workers=_int("DIFFJURY_KNOWLEDGE_GATE_WORKERS", 4),
            memory_episodic_ttl_days=float(os.getenv("DIFFJURY_MEMORY_EPISODIC_TTL_DAYS", "90")),
            memory_episodic_cap=_non_negative_int("DIFFJURY_MEMORY_EPISODIC_CAP", 300),
            memory_semantic_cap=_non_negative_int("DIFFJURY_MEMORY_SEMANTIC_CAP", 500),
        )
