"""Generate a static HTML browser for language benchmark trial artifacts."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import tyro

from experiments.common import read_json


@dataclass
class LanguageReplayViewerArgs:
    """CLI args for static language benchmark trial browsing."""

    experiment_dir: Path
    """Path to one experiment results directory."""

    output_path: Path | None = None
    """Optional output HTML path; defaults to language/viewer/index.html."""


# Return the default static language viewer path for one experiment.
def default_language_viewer_path(experiment_dir: Path) -> Path:
    """Return the default generated language viewer HTML path."""
    return experiment_dir / "language" / "viewer" / "index.html"


# Load all saved language raw trial artifacts in deterministic trial order.
def load_language_trial_payloads(experiment_dir: Path) -> List[Dict[str, Any]]:
    """Return raw language trial payloads sorted by trial_id."""
    raw_dir = experiment_dir / "language" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"No language raw artifact directory found at {raw_dir}.")
    trial_paths = sorted(raw_dir.glob("*.json"))
    if not trial_paths:
        raise ValueError(f"No language raw trial artifacts found in {raw_dir}.")
    trials = [read_json(path) for path in trial_paths]
    return sorted(trials, key=lambda payload: str(payload.get("trial_id", "")))


# Build compact summary metrics for the loaded language trials.
def build_language_viewer_summary(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return counts and component accuracies for one language viewer payload."""
    if not trials:
        return {"num_trials": 0, "num_exact_match": 0, "exact_match_accuracy": 0.0}
    exact_matches = sum(1 for trial in trials if bool(trial.get("exact_match")))
    summary: Dict[str, Any] = {
        "num_trials": len(trials),
        "num_exact_match": exact_matches,
        "exact_match_accuracy": exact_matches / len(trials),
    }
    for field_name in (
        "intent_match",
        "object_match",
        "target_match",
        "clarification_match",
    ):
        summary[f"{field_name}_accuracy"] = sum(
            1 for trial in trials if bool(trial.get(field_name))
        ) / len(trials)
    return summary


# Serialize JSON for safe embedding inside a script tag.
def _script_json(payload: Any) -> str:
    """Return JSON text escaped for inline script-tag embedding."""
    return json.dumps(payload, sort_keys=True).replace("</", "<\\/")


# Render one standalone HTML document for the language trial viewer.
def render_language_viewer_html(
    *,
    experiment_name: str,
    trials: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> str:
    """Return a self-contained HTML document for inspecting language trials."""
    escaped_title = html.escape(experiment_name)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Language Trial Viewer - {escaped_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8f7f4;
      --panel: #ffffff;
      --text: #202124;
      --muted: #5f6368;
      --line: #d7d2c8;
      --accent: #2a7f76;
      --bad: #b3261e;
      --bad-bg: #fce8e6;
      --good: #137333;
      --good-bg: #e6f4ea;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: #fffaf0;
    }}
    h1 {{ margin: 0 0 6px; font-size: 28px; font-weight: 700; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h3 {{ margin: 0; font-size: 15px; }}
    .subtle {{ color: var(--muted); margin: 0; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      padding: 18px 32px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .card .value {{ font-size: 24px; font-weight: 700; margin-top: 3px; }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(220px, 2fr) minmax(160px, 1fr) minmax(160px, 1fr);
      gap: 12px;
      padding: 0 32px 18px;
    }}
    input, select {{
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: white;
      color: var(--text);
      font: inherit;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(330px, 42%) minmax(360px, 1fr);
      gap: 18px;
      padding: 0 32px 32px;
    }}
    .list, .detail {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 500px;
      overflow: hidden;
    }}
    .list-header, .detail-header {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #fbfaf8;
    }}
    .trial-list {{ max-height: 72vh; overflow: auto; }}
    .trial {{
      width: 100%;
      text-align: left;
      display: block;
      border: 0;
      border-bottom: 1px solid var(--line);
      background: white;
      padding: 12px 14px;
      cursor: pointer;
    }}
    .trial:hover, .trial.active {{ background: #eef7f6; }}
    .trial-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      font-weight: 700;
      font-size: 14px;
    }}
    .trial-meta {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .prompt {{ margin-top: 7px; font-size: 13px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .pass {{ background: var(--good-bg); color: var(--good); }}
    .fail {{ background: var(--bad-bg); color: var(--bad); }}
    .detail-body {{ padding: 16px; }}
    .comparison {{
      display: grid;
      grid-template-columns: 150px 1fr 1fr;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .comparison div {{ padding: 9px 10px; border-bottom: 1px solid var(--line); }}
    .comparison div:nth-child(3n+1) {{ background: #fbfaf8; font-weight: 700; }}
    .comparison div:nth-last-child(-n+3) {{ border-bottom: 0; }}
    .mismatch {{ background: var(--bad-bg); color: var(--bad); font-weight: 700; }}
    .section {{ margin-top: 18px; }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 0;
      background: #f6f5f2;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
    }}
    @media (max-width: 900px) {{
      .controls, main {{ grid-template-columns: 1fr; padding-left: 18px; padding-right: 18px; }}
      .summary, header {{ padding-left: 18px; padding-right: 18px; }}
      .trial-list {{ max-height: 45vh; }}
      .comparison {{ grid-template-columns: 112px 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Language Trial Viewer</h1>
    <p class="subtle">Experiment: {escaped_title}</p>
  </header>
  <section id="summary" class="summary"></section>
  <section class="controls" aria-label="Trial filters">
    <input id="search" type="search" placeholder="Search prompt, assistant text, object, intent, target">
    <select id="family-filter"></select>
    <select id="status-filter">
      <option value="all">All trials</option>
      <option value="fail">Exact-match failures</option>
      <option value="pass">Exact-match passes</option>
    </select>
  </section>
  <main>
    <section class="list">
      <div class="list-header">
        <h2>Trials <span id="visible-count" class="subtle"></span></h2>
      </div>
      <div id="trial-list" class="trial-list"></div>
    </section>
    <section class="detail">
      <div class="detail-header">
        <h2 id="detail-title">Trial Detail</h2>
        <p id="detail-subtitle" class="subtle"></p>
      </div>
      <div id="detail-body" class="detail-body"></div>
    </section>
  </main>
  <script>
    const TRIALS = {_script_json(trials)};
    const SUMMARY = {_script_json(summary)};
    const FIELDS = [
      ["Outcome", "expected_outcome_type", "predicted_outcome_type"],
      ["Intent", "expected_intent", "predicted_intent"],
      ["Object", "expected_object_name", "predicted_object_name"],
      ["Target", "expected_target_label", "predicted_target_label"],
      ["Clarification", "expected_clarification", "predicted_clarification"],
    ];
    let selectedTrialId = TRIALS.length ? TRIALS[0].trial_id : null;

    function valueText(value) {{
      if (value === null || value === undefined || value === "") return "(none)";
      return String(value);
    }}

    function escapeText(value) {{
      return valueText(value).replace(/[&<>"']/g, (char) => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "\\"": "&quot;", "'": "&#39;"
      }}[char]));
    }}

    function pct(value) {{
      return `${{(Number(value || 0) * 100).toFixed(1)}}%`;
    }}

    function prettyJson(value) {{
      if (value === null || value === undefined || value === "") return "(none)";
      return JSON.stringify(value, null, 2);
    }}

    function renderSummary() {{
      const cards = [
        ["Trials", SUMMARY.num_trials],
        ["Exact Match", `${{SUMMARY.num_exact_match}}/${{SUMMARY.num_trials}}`],
        ["Exact Accuracy", pct(SUMMARY.exact_match_accuracy)],
        ["Intent", pct(SUMMARY.intent_match_accuracy)],
        ["Object", pct(SUMMARY.object_match_accuracy)],
        ["Target", pct(SUMMARY.target_match_accuracy)],
        ["Clarification", pct(SUMMARY.clarification_match_accuracy)],
      ];
      document.getElementById("summary").innerHTML = cards.map(([label, value]) => `
        <div class="card"><div class="label">${{escapeText(label)}}</div><div class="value">${{escapeText(value)}}</div></div>
      `).join("");
    }}

    function renderFamilyFilter() {{
      const families = [...new Set(TRIALS.map((trial) => trial.prompt_family || "unknown"))].sort();
      document.getElementById("family-filter").innerHTML = [
        '<option value="all">All prompt families</option>',
        ...families.map((family) => `<option value="${{escapeText(family)}}">${{escapeText(family)}}</option>`),
      ].join("");
    }}

    function trialSearchText(trial) {{
      return [
        trial.trial_id, trial.prompt_id, trial.prompt_family, trial.prompt_variant,
        trial.prompt_text, trial.assistant_text, trial.active_object_context,
        trial.expected_intent, trial.predicted_intent, trial.expected_object_name,
        trial.predicted_object_name, trial.expected_target_label, trial.predicted_target_label,
        JSON.stringify(trial.expected_tool_call || null),
        JSON.stringify(trial.predicted_tool_call || null),
        JSON.stringify(trial.target_evaluation || null),
        JSON.stringify(trial.predicted_tool_trace || null),
        trial.error,
      ].map(valueText).join(" ").toLowerCase();
    }}

    function filteredTrials() {{
      const query = document.getElementById("search").value.trim().toLowerCase();
      const family = document.getElementById("family-filter").value;
      const status = document.getElementById("status-filter").value;
      return TRIALS.filter((trial) => {{
        if (family !== "all" && trial.prompt_family !== family) return false;
        if (status === "pass" && !trial.exact_match) return false;
        if (status === "fail" && trial.exact_match) return false;
        return query === "" || trialSearchText(trial).includes(query);
      }});
    }}

    function renderTrialList() {{
      const visibleTrials = filteredTrials();
      if (!visibleTrials.some((trial) => trial.trial_id === selectedTrialId)) {{
        selectedTrialId = visibleTrials.length ? visibleTrials[0].trial_id : null;
      }}
      document.getElementById("visible-count").textContent = `(${{visibleTrials.length}} visible)`;
      document.getElementById("trial-list").innerHTML = visibleTrials.map((trial) => {{
        const statusClass = trial.exact_match ? "pass" : "fail";
        const statusText = trial.exact_match ? "pass" : "fail";
        const activeClass = trial.trial_id === selectedTrialId ? " active" : "";
        return `<button class="trial${{activeClass}}" data-trial-id="${{escapeText(trial.trial_id)}}">
          <div class="trial-title">
            <span>${{escapeText(trial.trial_id)}}</span>
            <span class="badge ${{statusClass}}">${{statusText}}</span>
          </div>
          <div class="trial-meta">${{escapeText(trial.prompt_family)}} / ${{escapeText(trial.prompt_variant)}}</div>
          <div class="prompt">${{escapeText(trial.prompt_text)}}</div>
        </button>`;
      }}).join("");
      document.querySelectorAll(".trial").forEach((button) => {{
        button.addEventListener("click", () => {{
          selectedTrialId = button.dataset.trialId;
          renderTrialList();
          renderDetail();
        }});
      }});
      renderDetail();
    }}

    function renderComparison(trial) {{
      const rows = [
        "<div>Field</div><div>Expected</div><div>Predicted</div>",
        ...FIELDS.map(([label, expectedKey, predictedKey]) => {{
          const expectedValue = valueText(trial[expectedKey]);
          const predictedValue = valueText(trial[predictedKey]);
          const mismatchClass = expectedValue === predictedValue ? "" : " class=\\"mismatch\\"";
          return `<div>${{escapeText(label)}}</div><div>${{escapeText(expectedValue)}}</div><div${{mismatchClass}}>${{escapeText(predictedValue)}}</div>`;
        }}),
      ];
      return `<div class="comparison">${{rows.join("")}}</div>`;
    }}

    function renderDetail() {{
      const trial = TRIALS.find((candidate) => candidate.trial_id === selectedTrialId);
      if (!trial) {{
        document.getElementById("detail-title").textContent = "No Trial Selected";
        document.getElementById("detail-subtitle").textContent = "";
        document.getElementById("detail-body").innerHTML = "<p>No trials match the current filters.</p>";
        return;
      }}
      document.getElementById("detail-title").textContent = trial.trial_id;
      document.getElementById("detail-subtitle").textContent = `${{trial.prompt_family}} / ${{trial.prompt_variant}}`;
      const checks = [
        ["Exact", trial.exact_match],
        ["Intent", trial.intent_match],
        ["Object", trial.object_match],
        ["Target", trial.target_match],
        ["Clarification", trial.clarification_match],
      ].map(([label, ok]) => `<span class="badge ${{ok ? "pass" : "fail"}}">${{escapeText(label)}}: ${{ok ? "pass" : "fail"}}</span>`).join(" ");
      document.getElementById("detail-body").innerHTML = `
        <h3>Prompt</h3>
        <pre>${{escapeText(trial.prompt_text)}}</pre>
        <div class="section">${{checks}}</div>
        <div class="section">
          <h3>Expected vs Predicted</h3>
          ${{renderComparison(trial)}}
        </div>
        <div class="section">
          <h3>Assistant Text</h3>
          <pre>${{escapeText(trial.assistant_text)}}</pre>
        </div>
        <div class="section">
          <h3>Expected Tool Call</h3>
          <pre>${{escapeText(prettyJson(trial.expected_tool_call))}}</pre>
        </div>
        <div class="section">
          <h3>Predicted Tool Call</h3>
          <pre>${{escapeText(prettyJson(trial.predicted_tool_call))}}</pre>
        </div>
        <div class="section">
          <h3>Target Evaluation</h3>
          <pre>${{escapeText(prettyJson(trial.target_evaluation))}}</pre>
        </div>
        <div class="section">
          <h3>Predicted Tool Trace</h3>
          <pre>${{escapeText(prettyJson(trial.predicted_tool_trace || []))}}</pre>
        </div>
        <div class="section">
          <h3>Context and Metrics</h3>
          <pre>${{escapeText(JSON.stringify({{
            active_object_context: trial.active_object_context,
            backend: trial.backend,
            metrics: trial.metrics || {{}},
            error: trial.error || null,
          }}, null, 2))}}</pre>
        </div>
      `;
    }}

    function rerender() {{
      renderTrialList();
    }}

    renderSummary();
    renderFamilyFilter();
    document.getElementById("search").addEventListener("input", rerender);
    document.getElementById("family-filter").addEventListener("change", rerender);
    document.getElementById("status-filter").addEventListener("change", rerender);
    renderTrialList();
  </script>
</body>
</html>
"""


# Generate the static language viewer HTML and return its output path.
def write_language_viewer_html(experiment_dir: Path, output_path: Path | None = None) -> Path:
    """Write a static language trial viewer for one experiment directory."""
    trials = load_language_trial_payloads(experiment_dir)
    metadata_path = experiment_dir / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    output_html_path = output_path or default_language_viewer_path(experiment_dir)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(
        render_language_viewer_html(
            experiment_name=str(metadata.get("experiment_name", experiment_dir.name)),
            trials=trials,
            summary=build_language_viewer_summary(trials),
        )
    )
    return output_html_path


# Parse CLI args and generate the language trial viewer.
def main() -> None:
    """Entry point for static language benchmark trial viewer generation."""
    args = tyro.cli(LanguageReplayViewerArgs)
    output_path = write_language_viewer_html(args.experiment_dir, args.output_path)
    print(f"Wrote language trial viewer to {output_path}")


if __name__ == "__main__":
    main()
