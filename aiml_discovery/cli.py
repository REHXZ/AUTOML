"""aiml-discovery CLI entry point.

Usage
-----
    aiml-discovery init
    aiml-discovery run --data ./data.csv --goal "Predict churn"
    aiml-discovery run --data ./data.csv --goal "Predict churn" --provider anthropic
    aiml-discovery run --data ./data.csv --goal "Predict churn" --output ./results/
    aiml-discovery run --resume <session-id> --goal "Predict churn"
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

# ── Shared style helpers ──────────────────────────────────────────────────────

STEP_COLORS: dict[str, str] = {
    "scientist":             "cyan",
    "eda":                   "blue",
    "feature_engineering":   "yellow",
    "modeling":              "green",
    "model_tester":          "bright_green",
    "review":                "magenta",
    "fine_tuning":           "bright_yellow",
    "researcher":            "bright_blue",
    "drift":                 "red",
}


def _agent_style(agent: str, no_color: bool) -> str:
    if no_color:
        return f"[{agent}]"
    color = STEP_COLORS.get(agent.lower(), "white")
    try:
        import click as _click
        return _click.style(f"[{agent}]", fg=color, bold=True)
    except Exception:
        return f"[{agent}]"


def _print_step(step, no_color: bool) -> None:
    agent = step.agent or "autopilot"
    tag = _agent_style(agent, no_color)

    if step.thought:
        click.echo(f"  {tag} {step.thought[:160]}")
    elif step.tool_call:
        suffix = f" → {step.tool_call}" if not no_color else f" -> {step.tool_call}"
        click.echo(f"  {tag}{suffix}")
    elif getattr(step, "kind", None) == "training":
        click.echo(f"  {tag} Training complete")
    elif getattr(step, "kind", None) == "chart":
        click.echo(f"  {tag} Chart generated")


# ── CLI root ──────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(package_name="aiml-discovery")
def main() -> None:
    """aiml-discovery - autonomous AutoML in your terminal."""


# ── init ──────────────────────────────────────────────────────────────────────

_ENV_TEMPLATES: dict[str, str] = {
    "openai": (
        "# OpenAI\n"
        "OPENAI_API_KEY=sk-...\n"
        "# OPENAI_MODEL=gpt-4o\n"
    ),
    "anthropic": (
        "# Anthropic\n"
        "ANTHROPIC_API_KEY=sk-ant-...\n"
        "# ANTHROPIC_MODEL=claude-opus-4-8\n"
    ),
    "azure": (
        "# Azure OpenAI\n"
        "OPENAI_API_KEY=<your-azure-key>\n"
        "OPENAI_API_BASE=https://<resource>.openai.azure.com/\n"
        "AZURE_OPENAI_DEPLOYMENT=gpt-4o\n"
    ),
    "ollama": (
        "# Ollama (local)\n"
        "OLLAMA_BASE_URL=http://localhost:11434/v1\n"
        "# OLLAMA_MODEL=llama3\n"
    ),
}


@main.command()
@click.option(
    "--provider", "-p",
    type=click.Choice(["openai", "anthropic", "azure", "ollama"], case_sensitive=False),
    default="openai",
    show_default=True,
    help="Provider to generate .env template for.",
)
@click.option("--force", is_flag=True, help="Overwrite existing .env file.")
def init(provider: str, force: bool) -> None:
    """Create a .env template with credentials for your chosen LLM provider."""
    env_path = Path(".env")
    if env_path.exists() and not force:
        click.echo(f".env already exists. Use --force to overwrite.")
        sys.exit(1)

    template = _ENV_TEMPLATES.get(provider.lower(), _ENV_TEMPLATES["openai"])
    env_path.write_text(template, encoding="utf-8")
    click.echo(click.style("[ok]", fg="green") + f"  Created .env for {provider}.")
    click.echo("  Fill in your credentials, then run:")
    click.echo(click.style('    aiml-discovery run --data ./data.csv --goal "Your goal"', fg="cyan"))


# ── run ───────────────────────────────────────────────────────────────────────

@main.command("run")
@click.option("--data",     "-d", type=click.Path(exists=True), default=None,
              help="Path to dataset (CSV, Excel, JSON, SQLite).")
@click.option("--goal",     "-g", default="", show_default=False,
              help="Natural-language objective for the autopilot.")
@click.option("--provider",       default=None,
              type=click.Choice(["openai", "anthropic", "azure", "ollama", "custom"], case_sensitive=False),
              help="LLM provider (auto-detected from env when omitted).")
@click.option("--model",          default=None,
              help="Override the model name (e.g. gpt-4o, claude-opus-4-8).")
@click.option("--api-key",        default=None, envvar="AIML_API_KEY",
              help="API key (overrides env vars).")
@click.option("--base-url",       default=None,
              help="Custom base URL (Azure endpoint, Ollama URL, etc.).")
@click.option("--output",   "-o", default="./aiml_output", show_default=True,
              help="Directory for notebook + reports.")
@click.option("--project",        default=None,
              help="Project label (default: data file stem).")
@click.option("--resume",         default=None, metavar="SESSION_ID",
              help="Resume a previous session by ID.")
@click.option("--quiet",    "-q", is_flag=True,
              help="Suppress step-by-step output.")
@click.option("--no-color",       is_flag=True,
              help="Plain text output (for CI / log files).")
@click.option("--save/--no-save", default=True, show_default=True,
              help="Write notebook + reports to --output after completion.")
def run_cmd(
    data: str | None,
    goal: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    output: str,
    project: str | None,
    resume: str | None,
    quiet: bool,
    no_color: bool,
    save: bool,
) -> None:
    """Run the AutoML autopilot on a dataset."""

    # ── validate inputs ───────────────────────────────────────────────────────
    if not data and not resume:
        raise click.UsageError("--data is required (or --resume to continue a session).")
    if not goal:
        goal = click.prompt("Goal")

    # ── build provider config ─────────────────────────────────────────────────
    from backend.logic.providers import ProviderConfig, provider_from_env

    if provider or model or api_key or base_url:
        cfg = provider_from_env(provider)
        if model:
            cfg.model = model
        if api_key:
            cfg.api_key = api_key
        if base_url:
            cfg.base_url = base_url
    else:
        cfg = provider_from_env()

    # ── banner ────────────────────────────────────────────────────────────────
    if not quiet:
        _print_banner(data, goal, cfg, no_color)

    # ── run ───────────────────────────────────────────────────────────────────
    from aiml_discovery import Autopilot

    pilot = Autopilot(
        data=data,
        goal=goal,
        provider=cfg,
        project=project or "",
        output_dir=output,
        session_id=resume,
    )

    if not quiet:
        click.echo(f"  Session: {pilot.session_id}\n")

    ask_pending: str | None = None

    try:
        for step in pilot.run():
            kind = getattr(step, "kind", None)

            if kind == "ask_user":
                ask_pending = step.thought or "The autopilot has a question:"
                click.echo()
                click.echo(click.style("? ", fg="yellow", bold=True) + ask_pending)
                answer = click.prompt("  Your answer")
                for follow in pilot.answer(answer):
                    if not quiet:
                        _print_step(follow, no_color)
                ask_pending = None
            elif not quiet:
                _print_step(step, no_color)

    except KeyboardInterrupt:
        click.echo("\n  Interrupted — saving partial results...")
        pilot.stop()

    # ── save results ──────────────────────────────────────────────────────────
    if save:
        dest = pilot.save_results(output)
        if not quiet:
            click.echo()
            ok = click.style("[ok]", fg="green", bold=True)
            click.echo(f"  {ok}  Results saved to {dest}/")
            click.echo(f"      notebook.ipynb   — open in Jupyter")
            click.echo(f"      training_runs.json")
            click.echo(f"      summary.txt")

    # ── summary line ─────────────────────────────────────────────────────────
    if not quiet:
        best = pilot.best_model
        runs = pilot.training_runs
        click.echo()
        if best:
            click.echo(f"  Best model : {click.style(best, fg='green', bold=True)}")
        if runs:
            click.echo(f"  Models tried: {len(runs)}")


# ── sessions ──────────────────────────────────────────────────────────────────

@main.command("sessions")
@click.option("--project", "-p", default=None, help="Filter by project label.")
@click.option("--limit",   "-n", default=10, show_default=True, help="Max rows to show.")
def sessions_cmd(project: str | None, limit: int) -> None:
    """List recent autopilot sessions."""
    from backend.services.project_store import ProjectStore
    from backend.services.session_store import list_sessions

    store = ProjectStore()
    projects = store.list_projects(user_id="local")

    if project:
        projects = [p for p in projects if project.lower() in p.name.lower()]

    rows: list[tuple[str, str, str, str]] = []
    for proj in projects:
        try:
            sessions = list_sessions(store, proj.id)
        except Exception:
            sessions = []
        for s in sessions[:limit]:
            rows.append((s.session_id, proj.name, s.status, s.user_goal[:50]))

    if not rows:
        click.echo("No sessions found.")
        return

    rows = rows[:limit]
    click.echo(f"{'SESSION ID':<36}  {'PROJECT':<20}  {'STATUS':<10}  GOAL")
    click.echo("-" * 90)
    for sid, pname, status, goal_text in rows:
        color = "green" if status == "complete" else ("yellow" if status == "running" else "white")
        click.echo(f"{sid:<36}  {pname:<20}  {click.style(status, fg=color):<10}  {goal_text}")


# ── providers ─────────────────────────────────────────────────────────────────

@main.command("providers")
def providers_cmd() -> None:
    """Show which LLM providers are configured via environment variables."""
    from backend.logic.providers import PROVIDER_PRESETS, configured_providers

    active = set(configured_providers())
    click.echo("Configured providers (from env vars):\n")
    for name, preset in PROVIDER_PRESETS.items():
        label = preset.get("label", name)
        tick = click.style("[ok]", fg="green") if name in active else click.style("[--]", fg="red")
        click.echo(f"  {tick}  {label} ({name})")
    click.echo()
    if not active:
        click.echo("  No provider detected. Run `aiml-discovery init` to create a .env file.")


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_banner(data: str | None, goal: str, cfg, no_color: bool) -> None:
    from backend.logic.providers import PROVIDER_PRESETS

    label = PROVIDER_PRESETS.get(cfg.provider, {}).get("label", cfg.provider)
    model = cfg.effective_model()

    if no_color:
        click.echo("=" * 60)
        click.echo("aiml-discovery")
        click.echo(f"  data     : {data or '(resume)'}")
        click.echo(f"  goal     : {goal}")
        click.echo(f"  provider : {label}  ({model})")
        click.echo("=" * 60)
    else:
        click.echo(click.style("aiml-discovery", fg="cyan", bold=True))
        click.echo(f"  {click.style('data    ', fg='bright_black')} {data or '(resume)'}")
        click.echo(f"  {click.style('goal    ', fg='bright_black')} {goal}")
        click.echo(f"  {click.style('provider', fg='bright_black')} {label}  {click.style(model, fg='bright_black')}")
        click.echo()
