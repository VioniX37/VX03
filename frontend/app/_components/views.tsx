"use client";

import React, { useMemo, useState } from "react";
import type { MLRecommendation, ModelMetadata, PresolveData, SolveResult, TrustCertificate } from "./types";
import { LineChart, MatrixView, TreeView, treeWord, type ChartSeries } from "./charts";
import { AsciiBar, Btn, Comment, CountUp, Leader, Panel, Spinner, fmt, toneText, useCountUp, useInView, useReveal, type Tone } from "./term";

/* ---------------------------------------------------------------- helpers */

export function statusTone(status: string): Tone {
  if (status === "optimal") return "ok";
  if (["feasible", "time_limit", "iteration_limit", "node_limit"].includes(status)) return "warn";
  return "bad";
}

export function statusLabel(status: string) {
  return fmt.words(status);
}

export function verdict(cert: TrustCertificate): { tone: Tone; label: string } {
  switch (cert.status) {
    case "OPTIMAL_CERTIFIED":
      return { tone: "ok", label: "certified optimal" };
    case "FEASIBLE_CERTIFIED":
      return { tone: "warn", label: "feasible, optimality not proven" };
    case "FAILED":
      return { tone: "bad", label: "failed verification" };
    default:
      return { tone: "dim", label: "no solution to verify" };
  }
}

const STAGE_KEYS = ["model", "presolve", "strategy", "solve", "postsolve", "certify"];
const TIMING_KEYS: Record<number, string> = { 2: "presolve", 3: "strategy", 4: "solve", 5: "postsolve", 6: "certificate" };

function stageWord(status: string): { text: string; tone: Tone } {
  if (status === "COMPLETED" || status === "CERTIFIED") return { text: "ok", tone: "ok" };
  if (status === "BYPASS") return { text: "skip", tone: "dim" };
  if (status === "FAILED") return { text: "fail", tone: "bad" };
  return { text: "n/a", tone: "warn" };
}

function objectiveDigits(v: number | null | undefined) {
  return v !== null && v !== undefined && Math.abs(v) >= 1000 ? 2 : 4;
}

/* ---------------------------------------------------------------- run log */

export function RunLog({ result, command }: { result: SolveResult; command: string }) {
  const stages = result.workflow_stages ?? [];
  const shown = useReveal(stages.length + 1, 120);
  return (
    <div className="overflow-x-auto">
      <div className="min-w-[640px] space-y-0.5 text-[13.5px]">
        <p className="whitespace-pre-wrap break-all">
          <span className="select-none text-amber">$ </span>
          <span className="text-fg">{command}</span>
        </p>
        {stages.slice(0, shown).map((st) => {
          const word = stageWord(st.status);
          const key = TIMING_KEYS[st.stage];
          const took = key && result.timings?.[key] !== undefined ? fmt.dur(result.timings[key]) : "";
          return (
            <div key={st.stage} className="animate-type grid grid-cols-[10ch_5ch_minmax(0,1fr)_10ch] gap-x-3">
              <span className="text-dim">{STAGE_KEYS[st.stage - 1] ?? st.name}</span>
              <span className={toneText[word.tone]}>{word.text}</span>
              <span className="truncate text-fg" title={st.summary}>
                {st.summary}
              </span>
              <span className="text-right text-faint">{took}</span>
            </div>
          );
        })}
        {shown > stages.length ? (
          <p className="animate-fade pt-1 text-dim">finished in {fmt.dur(result.timings?.total ?? result.runtime_seconds)}</p>
        ) : null}
      </div>
    </div>
  );
}

export function ResultStrip({ result }: { result: SolveResult }) {
  const v = verdict(result.trust_certificate);
  const stat = (label: string, value: React.ReactNode, wide = false) => (
    <div className={`min-w-0 ${wide ? "col-span-2" : ""}`}>
      <p className="text-[12px] text-dim">{label}</p>
      <p className={`mt-0.5 truncate ${wide ? "text-[22px] leading-tight text-fg" : "text-[14px]"}`}>{value}</p>
    </div>
  );
  return (
    <div className="animate-rise grid grid-cols-2 gap-x-8 gap-y-4 sm:grid-cols-4 xl:grid-cols-8">
      {stat(
        "objective",
        <CountUp value={result.objective_value} format={(x) => fmt.num(x, objectiveDigits(result.objective_value))} duration={900} />,
        true,
      )}
      {stat("status", <span className={toneText[statusTone(result.status)]}>{statusLabel(result.status)}</span>)}
      {stat("certificate", <span className={toneText[v.tone]}>{v.label}</span>, true)}
      {stat("iterations", <CountUp value={result.iterations} format={(x) => fmt.int(x)} />)}
      {result.nodes_explored > 0
        ? stat("nodes", <CountUp value={result.nodes_explored} format={(x) => fmt.int(x)} />)
        : stat("method", fmt.words(result.algorithm_key ?? result.algorithm_used))}
      {result.mip_gap !== null && result.mip_gap !== undefined
        ? stat("gap", fmt.pct(result.mip_gap * 100, 3))
        : stat("solve time", fmt.dur(result.timings?.solve ?? result.runtime_seconds))}
    </div>
  );
}

/* ------------------------------------------------------------- 1: model */

/** [▒▒▒▒████████] continuous / integer / binary, printed left to right when it scrolls into view. */
function CompositionBar({ model, total }: { model: ModelMetadata; total: number }) {
  const width = 20;
  const [ref, inView] = useInView<HTMLParagraphElement>();
  const reveal = Math.round(useCountUp(inView ? width : 0, 700) ?? 0);
  const cont = Math.round((model.num_continuous / total) * width);
  const int = Math.round((model.num_integer / total) * width);
  const bin = model.num_binary > 0 ? Math.max(0, width - cont - int) : 0;
  const cells = [
    ...Array.from({ length: cont }, () => ({ ch: "▒", cls: "text-faint" })),
    ...Array.from({ length: int }, () => ({ ch: "█", cls: "text-fg" })),
    ...Array.from({ length: bin }, () => ({ ch: "█", cls: "text-amber" })),
  ].slice(0, width);
  return (
    <p ref={ref} aria-hidden className="mb-2 overflow-hidden whitespace-pre text-[13.5px]">
      <span className="text-faint">[</span>
      {cells.slice(0, reveal).map((c, i) => (
        <span key={i} className={c.cls}>
          {c.ch}
        </span>
      ))}
      <span className="text-faint/70">{"·".repeat(Math.max(0, width - Math.min(reveal, cells.length)))}</span>
      <span className="text-faint">]</span>
    </p>
  );
}

export function ModelView({ model }: { model: ModelMetadata }) {
  const total = Math.max(model.num_variables, 1);
  const ruiz = model.ruiz_stats;
  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
      <Panel
        title="constraint matrix"
        right={`${fmt.int(model.num_constraints)} x ${fmt.int(model.num_variables)} · ${fmt.pct(model.density * 100)} filled`}
      >
        <Comment className="mb-4">each row is a constraint and each column a variable. hover a cell to read its coefficient.</Comment>
        <MatrixView model={model} />
        {model.variables?.length || model.constraints?.length ? (
          <details className="group mt-5 border-t border-line pt-3">
            <summary className="cursor-pointer list-none text-[13.5px] text-dim hover:text-fg">
              <span className="text-amber group-open:hidden">+ </span>
              <span className="hidden text-amber group-open:inline">- </span>
              variables and constraints
            </summary>
            <div className="animate-rise mt-3 grid gap-6 md:grid-cols-2">
              <div className="max-h-[300px] overflow-y-auto">
                <table className="tty">
                  <thead>
                    <tr>
                      <th>variable</th>
                      <th>type</th>
                      <th className="text-right">bounds</th>
                    </tr>
                  </thead>
                  <tbody>
                    {model.variables?.map((v) => (
                      <tr key={v.name}>
                        <td className="max-w-[200px] truncate">{v.name}</td>
                        <td className="text-dim">{v.type}</td>
                        <td className="text-right text-dim">
                          [{typeof v.lb === "number" ? fmt.compact(v.lb) : "-inf"}, {typeof v.ub === "number" ? fmt.compact(v.ub) : "inf"}]
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="max-h-[300px] overflow-y-auto">
                <table className="tty">
                  <thead>
                    <tr>
                      <th>constraint</th>
                      <th>rule</th>
                      <th className="text-right">terms</th>
                    </tr>
                  </thead>
                  <tbody>
                    {model.constraints?.map((c) => (
                      <tr key={c.name}>
                        <td className="max-w-[200px] truncate">{c.name}</td>
                        <td className="text-dim">{c.sense === "range" ? "range" : `${c.sense} ${fmt.compact(c.rhs)}`}</td>
                        <td className="text-right text-dim">{c.num_terms}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </details>
        ) : null}
      </Panel>

      <div className="min-w-0 space-y-6">
        <Panel title="composition">
          <CompositionBar model={model} total={total} />
          <Leader label={<><span className="text-faint">▒</span> continuous</>}>{fmt.int(model.num_continuous)}</Leader>
          <Leader label={<><span className="text-fg">█</span> integer</>}>{fmt.int(model.num_integer)}</Leader>
          <Leader label={<><span className="text-amber">█</span> binary</>}>{fmt.int(model.num_binary)}</Leader>
          <Leader label="nonzeros">{fmt.int(model.num_nonzeros)}</Leader>
          <Leader label="quadratic terms">{fmt.int(model.num_quadratic_terms)}</Leader>
        </Panel>
        {ruiz ? (
          <Panel title="scaling">
            <Comment className="mb-2">rows and columns are rebalanced so no coefficient dominates.</Comment>
            <Leader label="largest |a|" hint="before">
              {fmt.compact(ruiz.norm_before)}
            </Leader>
            <Leader label="largest |a|" hint="after">
              {fmt.compact(ruiz.norm_after)}
            </Leader>
            <Leader label="row factors">
              {fmt.compact(ruiz.d1_min)} .. {fmt.compact(ruiz.d1_max)}
            </Leader>
            <Leader label="column factors">
              {fmt.compact(ruiz.d2_min)} .. {fmt.compact(ruiz.d2_max)}
            </Leader>
            <Leader label="passes">{ruiz.iterations}</Leader>
          </Panel>
        ) : null}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------- 2: presolve */

const REDUCTIONS: Array<{ key: keyof PresolveData; label: string; explain: string }> = [
  { key: "fixed_vars_count", label: "fixed variables", explain: "bounds already pin the value" },
  { key: "singleton_rows_count", label: "singleton rows", explain: "one-variable rows turned into bounds" },
  { key: "empty_rows_count", label: "empty rows", explain: "rows left with no terms" },
  { key: "empty_cols_count", label: "empty columns", explain: "variables set at their cheapest bound" },
  { key: "redundant_rows_count", label: "redundant rows", explain: "can never be violated" },
  { key: "forcing_rows_count", label: "forcing rows", explain: "force every variable to a bound" },
  { key: "doubleton_eliminations", label: "doubleton aggregations", explain: "two-variable equations substituted out" },
  { key: "dominated_cols_count", label: "dominated columns", explain: "fixed where the objective prefers" },
  { key: "parallel_rows_count", label: "parallel rows", explain: "proportional rows merged" },
  { key: "duplicate_cols_count", label: "duplicate columns", explain: "proportional columns merged" },
  { key: "tightened_bounds_count", label: "integer bounds tightened", explain: "implied by row activity" },
  { key: "coefficient_tightenings", label: "coefficients tightened", explain: "stronger rows for binaries" },
];

export function PresolveView({ presolve, onRun, busy }: { presolve: PresolveData | null; onRun: () => void; busy: boolean }) {
  if (!presolve) {
    return (
      <Panel title="presolve">
        <Comment className="mb-3">presolve removes rows and variables the solver does not need to see, then maps the answer back.</Comment>
        {busy ? (
          <p className="text-dim">
            <Spinner /> simplifying the model
          </p>
        ) : (
          <Btn onClick={onRun}>run presolve</Btn>
        )}
      </Panel>
    );
  }
  const rows = [
    { label: "variables", before: presolve.original_vars, after: presolve.presolved_vars, pct: presolve.var_reduction_pct },
    { label: "constraints", before: presolve.original_cons, after: presolve.presolved_cons, pct: presolve.con_reduction_pct },
    { label: "nonzeros", before: presolve.original_nnz, after: presolve.presolved_nnz, pct: presolve.nnz_reduction_pct },
  ];
  const applied = REDUCTIONS.filter((r) => Number(presolve[r.key] ?? 0) > 0);
  return (
    <div className="grid gap-6 xl:grid-cols-2">
      <Panel
        title="before and after"
        right={
          <Btn onClick={onRun} disabled={busy}>
            run again
          </Btn>
        }
      >
        <Comment className="mb-3">bars show how much of the original problem remains.</Comment>
        <div className="overflow-x-auto">
          <table className="tty">
            <thead>
              <tr>
                <th></th>
                <th className="text-right">before</th>
                <th className="text-right">after</th>
                <th className="text-right">removed</th>
                <th>remaining</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.label}>
                  <td className="text-dim">{r.label}</td>
                  <td className="text-right">{fmt.int(r.before)}</td>
                  <td className="text-right">
                    <CountUp value={r.after} format={(x) => fmt.int(x)} />
                  </td>
                  <td className={`text-right ${r.pct > 0 ? "text-green" : "text-dim"}`}>{fmt.pct(r.pct)}</td>
                  <td>
                    <AsciiBar value={r.before > 0 ? r.after / r.before : 1} width={16} tone="dim" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {presolve.fixed_vars_list && presolve.fixed_vars_list.length > 0 ? (
          <div className="mt-5 max-h-[240px] overflow-y-auto border-t border-line pt-3">
            <p className="mb-1 text-[12.5px] text-dim">values fixed by presolve</p>
            {presolve.fixed_vars_list.map((fv) => (
              <Leader key={fv.name} label={fv.name}>
                {fmt.compact(fv.value)}
              </Leader>
            ))}
          </div>
        ) : null}
      </Panel>

      <Panel title="what presolve did" right={presolve.presolve_passes ? `${presolve.presolve_passes} passes` : undefined}>
        {applied.length === 0 ? (
          <Comment>no reductions applied. every row and column in this model is needed.</Comment>
        ) : (
          <div>
            {applied.map((r, i) => (
              <div key={r.key} className="animate-type" style={{ animationDelay: `${i * 70}ms` }}>
                <Leader label={r.label} hint={`(${r.explain})`}>
                  {fmt.int(Number(presolve[r.key]))}
                </Leader>
              </div>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}

/* ---------------------------------------------------------- 3: strategy */

export function StrategyView({ rec, onRun, busy }: { rec: MLRecommendation | null; onRun: () => void; busy: boolean }) {
  if (!rec) {
    return (
      <Panel title="strategy">
        <Comment className="mb-3">the strategy engine reads the model structure and suggests a method. it never decides the answer.</Comment>
        {busy ? (
          <p className="text-dim">
            <Spinner /> reading the model structure
          </p>
        ) : (
          <Btn onClick={onRun}>get a recommendation</Btn>
        )}
      </Panel>
    );
  }
  const probs = Object.entries(rec.algorithm_probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <div className="grid gap-6 xl:grid-cols-2">
      <Panel
        title="recommendation"
        right={
          <Btn onClick={onRun} disabled={busy}>
            ask again
          </Btn>
        }
      >
        <p className="text-[12.5px] text-dim">method</p>
        <p className="animate-type mt-0.5 text-[22px] leading-tight text-amber">{fmt.words(rec.recommended_algorithm)}</p>
        <p className="mt-1 text-[13.5px] text-dim">
          {fmt.pct(rec.confidence_score, 0)} confident · falls back to <span className="text-fg">{fmt.words(rec.deterministic_fallback)}</span>
        </p>

        <div className="mt-5 space-y-1">
          {probs.map(([algo, p]) => (
            <div key={algo} className="flex flex-wrap items-baseline gap-x-3 text-[13.5px]">
              <span className="w-[20ch] shrink-0 truncate text-dim">{fmt.words(algo)}</span>
              <AsciiBar value={p} width={20} tone={algo === rec.recommended_algorithm ? "warn" : "dim"} />
              <span className="w-[5ch] text-right">{fmt.pct(p * 100, 0)}</span>
            </div>
          ))}
        </div>

        <div className="mt-5 border-t border-line pt-3">
          {rec.branching_strategy && rec.branching_strategy !== "none" ? (
            <Leader label="branching">{fmt.words(rec.branching_strategy)}</Leader>
          ) : null}
          {rec.cut_strategy && rec.cut_strategy !== "none" ? <Leader label="cutting planes">{rec.cut_strategy}</Leader> : null}
          {rec.heuristic_intensity && rec.heuristic_intensity !== "none" ? (
            <Leader label="heuristics">{rec.heuristic_intensity}</Leader>
          ) : null}
          {rec.relaxation_solver ? <Leader label="relaxations">{fmt.words(rec.relaxation_solver)}</Leader> : null}
          <Leader label="hardware">{rec.recommended_hardware}</Leader>
        </div>
      </Panel>

      <Panel title="what it looked at">
        <Comment className="mb-2">signals that drove this choice</Comment>
        {Object.entries(rec.feature_attributions).map(([k, v]) => (
          <Leader key={k} label={fmt.words(k)}>
            {fmt.compact(v)}
          </Leader>
        ))}
        {rec.features ? (
          <details className="group mt-4 border-t border-line pt-3">
            <summary className="cursor-pointer list-none text-[13.5px] text-dim hover:text-fg">
              <span className="text-amber group-open:hidden">+ </span>
              <span className="hidden text-amber group-open:inline">- </span>
              all {Object.keys(rec.features).length} structural features
            </summary>
            <div className="animate-rise mt-2 max-h-[320px] overflow-y-auto">
              {Object.entries(rec.features).map(([k, v]) => (
                <Leader key={k} label={fmt.words(k)}>
                  {fmt.compact(v)}
                </Leader>
              ))}
            </div>
          </details>
        ) : null}
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------- 4: solve */

export function SolveView({ result, runId }: { result: SolveResult | null; runId: number }) {
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const trace = useMemo(() => result?.diagnostics?.iteration_trace ?? [], [result]);
  const tree = useMemo(() => result?.diagnostics?.tree_trace ?? [], [result]);

  const objective: ChartSeries[] = useMemo(
    () => [
      {
        name: "objective",
        points: trace.filter((t) => t.objective !== undefined).map((t) => ({ x: t.iteration, y: t.objective as number })),
        stroke: "stroke-amber",
        text: "text-amber",
      },
    ],
    [trace],
  );
  const residuals: ChartSeries[] = useMemo(() => {
    const out: ChartSeries[] = [];
    const add = (key: "norm_rp" | "norm_rd" | "rel_gap", name: string, stroke: string, text: string) => {
      const points = trace.filter((t) => typeof t[key] === "number").map((t) => ({ x: t.iteration, y: t[key] as number }));
      if (points.length > 1) out.push({ name, points, stroke, text });
    };
    add("norm_rp", "primal residual", "stroke-fg", "text-fg");
    add("norm_rd", "dual residual", "stroke-red", "text-red");
    add("rel_gap", "gap", "stroke-green", "text-green");
    return out;
  }, [trace]);

  if (!result) {
    return (
      <Panel title="solve">
        <Comment>nothing solved yet. pick a method in the session panel and press run (ctrl+enter).</Comment>
      </Panel>
    );
  }

  const d = result.diagnostics ?? {};
  const selected = tree.find((n) => n.node_id === selectedNode) ?? null;
  const uniqueNodes = new Set(tree.map((n) => n.node_id)).size;

  return (
    <div className="space-y-6">
      {result.notes?.length ? (
        <Panel title="notes">
          {result.notes.map((n) => (
            <p key={n} className="text-[13.5px] text-amber">
              ! {n}
            </p>
          ))}
        </Panel>
      ) : null}

      {tree.length > 0 ? (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
          <Panel title="search tree" right={`${fmt.int(result.nodes_explored)} nodes explored`}>
            <Comment className="mb-3">
              {uniqueNodes <= 1
                ? "solved at the root: cutting planes and heuristics closed the gap before any branching."
                : "each line is a subproblem. click one to inspect it."}
            </Comment>
            <TreeView key={`tree-${runId}`} nodes={tree} selected={selectedNode} onSelect={setSelectedNode} />
          </Panel>
          <div className="space-y-6">
            <Panel title={selected ? `node ${selected.node_id}` : "node"}>
              {selected ? (
                <div key={selected.node_id} className="animate-rise">
                  <Leader label="status">
                    <span className={treeWord(selected.status).cls}>{treeWord(selected.status).text}</span>
                  </Leader>
                  <Leader label="depth">{selected.depth}</Leader>
                  <Leader label="branch">{selected.parent_id === null ? "root" : selected.branch_condition}</Leader>
                  <Leader label="bound" hint={selected.bound_note ? "(parent)" : undefined}>
                    {fmt.compact(selected.lower_bound)}
                  </Leader>
                </div>
              ) : (
                <Comment>select a line in the tree.</Comment>
              )}
            </Panel>
            <Panel title="search summary">
              {d.root_lp_bound !== undefined && d.root_lp_bound !== null ? <Leader label="root relaxation">{fmt.compact(d.root_lp_bound)}</Leader> : null}
              {result.best_bound !== undefined && result.best_bound !== null ? <Leader label="best bound">{fmt.compact(result.best_bound)}</Leader> : null}
              {result.mip_gap !== null && result.mip_gap !== undefined ? <Leader label="gap">{fmt.pct(result.mip_gap * 100, 3)}</Leader> : null}
              {d.cuts_by_type
                ? Object.entries(d.cuts_by_type).map(([k, v]) => (
                    <Leader key={k} label={`${k} cuts`}>
                      {fmt.int(v)}
                    </Leader>
                  ))
                : null}
              {d.heuristic_solutions
                ? Object.entries(d.heuristic_solutions)
                    .filter(([, v]) => v > 0)
                    .map(([k, v]) => (
                      <Leader key={k} label={`found by ${k === "lp_integral" ? "relaxation" : k}`}>
                        {fmt.int(v)}
                      </Leader>
                    ))
                : null}
              {d.termination ? <Leader label="stopped">{d.termination}</Leader> : null}
            </Panel>
          </div>
        </div>
      ) : (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_400px]">
          <div className="min-w-0 space-y-6">
            <Panel title="objective by iteration" right={d.algorithm ?? fmt.words(result.algorithm_key ?? result.algorithm_used)}>
              {objective[0].points.length > 1 ? (
                <LineChart key={`obj-${runId}`} series={objective} ariaLabel="Objective value by iteration" />
              ) : (
                <Comment>converged in a single step, so there is no trajectory to draw.</Comment>
              )}
            </Panel>
            {residuals.length ? (
              <Panel title="residuals" right="log scale">
                <Comment className="mb-2">all three must fall below 1e-8 for the interior point method to stop.</Comment>
                <LineChart key={`res-${runId}`} series={residuals} log height={200} ariaLabel="Residuals by iteration" />
              </Panel>
            ) : null}
          </div>
          <Panel title="iteration log" right={`${trace.length} lines`}>
            <div className="max-h-[560px] overflow-y-auto">
              <table className="tty" key={`log-${runId}`}>
                <thead>
                  <tr>
                    <th>iter</th>
                    <th className="text-right">objective</th>
                    <th>step</th>
                  </tr>
                </thead>
                <tbody>
                  {trace.map((t, i) => (
                    <tr key={`${t.iteration}-${i}`} className="animate-fade" style={{ animationDelay: `${Math.min(i * 16, 600)}ms` }}>
                      <td className="text-dim">{t.iteration}</td>
                      <td className="text-right">{t.objective !== undefined ? fmt.compact(t.objective) : "-"}</td>
                      <td className="max-w-[180px] truncate text-dim">
                        {t.status === "OPTIMAL"
                          ? "optimal"
                          : t.entering_var
                            ? `+${t.entering_var}${t.leaving_var === "bound_flip" ? " flip" : t.leaving_var ? ` -${t.leaving_var}` : ""}`
                            : t.mu !== undefined
                              ? `mu ${fmt.sci(t.mu)}`
                              : t.p_norm !== undefined
                                ? `|p| ${fmt.sci(t.p_norm)}`
                                : ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
      )}
    </div>
  );
}

/* --------------------------------------------------------- 5: solution */

const ORIGIN: Record<string, string> = {
  OPTIMIZED_IN_CORE: "solved",
  FIXED_IN_PRESOLVE: "presolve",
  CANONICAL_RESTORED: "restored",
  DIRECT_SOLVE: "solved",
  ELIMINATED_DOUBLETON_EQUATION: "aggregated",
  ELIMINATED_DUPLICATE_COLUMN: "merged",
};

export function SolutionView({ result }: { result: SolveResult | null }) {
  const [query, setQuery] = useState("");
  const mapping = useMemo(() => result?.postsolve_mapping ?? [], [result]);
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? mapping.filter((m) => m.variable.toLowerCase().includes(q)) : mapping;
  }, [mapping, query]);
  const duals = Object.entries(result?.dual_solution ?? {});
  const reduced = result?.reduced_costs ?? {};
  const hasReduced = Object.keys(reduced).length > 0;

  if (!result) {
    return (
      <Panel title="solution">
        <Comment>solve the model to see the values of the original variables.</Comment>
      </Panel>
    );
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
      <Panel title="variables" right={`${rows.length} of ${mapping.length} shown`}>
        <label className="mb-3 flex items-center gap-2 text-[13.5px]">
          <span className="text-amber">filter&gt;</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            spellCheck={false}
            className="flex-1 bg-transparent text-fg caret-amber placeholder:text-faint focus:outline-none"
            placeholder="type part of a variable name"
          />
        </label>
        <div className="max-h-[540px] overflow-y-auto">
          <table className="tty">
            <thead>
              <tr>
                <th>variable</th>
                <th className="text-right">value</th>
                {hasReduced ? <th className="text-right">reduced cost</th> : null}
                <th className="text-right">from</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((m, i) => (
                <tr key={m.variable} className="animate-fade" style={{ animationDelay: `${Math.min(i * 14, 420)}ms` }}>
                  <td className="max-w-[240px] truncate">{m.variable}</td>
                  <td className={`text-right ${Math.abs(m.value) < 1e-9 ? "text-faint" : "text-fg"}`}>{fmt.compact(m.value)}</td>
                  {hasReduced ? <td className="text-right text-dim">{fmt.compact(reduced[m.variable])}</td> : null}
                  <td className="text-right text-dim">{ORIGIN[m.resolution_state] ?? m.resolution_state.toLowerCase()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="row duals">
        {duals.length ? (
          <>
            <Comment className="mb-2">change in objective per unit of right-hand side. 0 means the row is not binding.</Comment>
            <div className="max-h-[520px] overflow-y-auto">
              {duals.map(([name, v]) => (
                <Leader key={name} label={name}>
                  <span className={Math.abs(v) < 1e-9 ? "text-faint" : ""}>{fmt.compact(v)}</span>
                </Leader>
              ))}
            </div>
          </>
        ) : (
          <Comment>
            {result.problem_class === "MILP" || result.problem_class === "MIQP"
              ? "integer programs have no meaningful row duals. optimality is proven by the search bound instead."
              : "no duals were returned for this run."}
          </Comment>
        )}
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------- 6: certificate */

export function CertificateView({ result }: { result: SolveResult | null }) {
  const cert = result?.trust_certificate;
  const checks = useMemo(() => {
    if (!cert || cert.status === "NO_SOLUTION_TO_VERIFY") return [];
    const out: Array<{ label: string; ok: boolean; value: string }> = [
      { label: "constraints satisfied", ok: !!cert.checks.constraints_satisfied, value: `max violation ${fmt.sci(cert.max_primal_violation)}` },
      { label: "bounds respected", ok: !!cert.checks.bounds_satisfied, value: `max violation ${fmt.sci(cert.max_bound_violation)}` },
      { label: "integers integral", ok: !!cert.checks.integrality_satisfied, value: `max gap ${fmt.sci(cert.max_integrality_violation)}` },
      { label: "objective reproduced", ok: !!cert.checks.objective_matches, value: `difference ${fmt.sci(cert.objective_difference)}` },
    ];
    if (cert.max_dual_violation !== null && cert.max_dual_violation !== undefined) {
      out.push({ label: "duals feasible", ok: !!cert.checks.dual_feasible, value: `max violation ${fmt.sci(cert.max_dual_violation)}` });
    }
    if (cert.duality_gap !== null && cert.duality_gap !== undefined) {
      const mip = "mip_gap_closed" in cert.checks;
      out.push({
        label: mip ? "bound meets incumbent" : "strong duality holds",
        ok: !!(mip ? cert.checks.mip_gap_closed : cert.checks.duality_gap_closed),
        value: `gap ${fmt.sci(cert.duality_gap)}`,
      });
    }
    return out;
  }, [cert]);
  const shown = useReveal(checks.length, 140);

  if (!result || !cert) {
    return (
      <Panel title="certificate">
        <Comment>solve the model first. every answer is re-checked against the original model, independently of the solver.</Comment>
      </Panel>
    );
  }
  const v = verdict(cert);
  const timings = Object.entries(result.timings ?? {}).filter(([k]) => k !== "total");
  const slowest = Math.max(1e-9, ...timings.map(([, t]) => t));

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
      <Panel title="certificate">
        <p className="text-[12.5px] text-dim">verdict</p>
        <p className={`animate-type mt-0.5 text-[22px] leading-tight ${toneText[v.tone]}`}>{v.label}</p>
        {cert.optimality_method && cert.optimality_method !== "none" ? (
          <Comment className="mt-1">proven by {cert.optimality_method}</Comment>
        ) : null}

        <div className="mt-5 space-y-0.5">
          {checks.slice(0, shown).map((c) => (
            <div key={c.label} className="animate-type flex items-baseline gap-2 text-[13.5px]">
              <span className={c.ok ? "text-green" : "text-red"}>[{c.ok ? "✓" : "x"}]</span>
              <span className="shrink-0 text-fg">{c.label}</span>
              <span aria-hidden className="min-w-4 flex-1 -translate-y-[4px] border-b border-dotted border-faint/70" />
              <span className="shrink-0 text-dim">{c.value}</span>
            </div>
          ))}
        </div>

        {cert.violation_details.length ? (
          <div className="mt-4 space-y-0.5 border-t border-line pt-3">
            {cert.violation_details.map((line) => (
              <p key={line} className="text-[13px] text-red">
                ! {line}
              </p>
            ))}
          </div>
        ) : null}

        <Comment className="mt-5">
          recomputed from the raw model in float64. tolerances 1e-6 (feasibility) and 1e-6 (optimality), relative to max(1, |bound|).
        </Comment>
      </Panel>

      <Panel title="pipeline" right={result.timings?.total !== undefined ? fmt.dur(result.timings.total) : undefined}>
        <Comment className="mb-3">where the time went</Comment>
        {timings.map(([k, t]) => (
          <div key={k} className="flex flex-wrap items-baseline gap-x-3 text-[13.5px]">
            <span className="w-[12ch] text-dim">{k}</span>
            <AsciiBar value={t / slowest} width={18} tone="dim" />
            <span className="w-[9ch] text-right">{fmt.dur(t)}</span>
          </div>
        ))}
        {result.workflow_stages ? (
          <div className="mt-5 border-t border-line pt-3">
            {result.workflow_stages.map((st) => {
              const word = stageWord(st.status);
              return (
                <div key={st.stage} className="py-1 text-[13px]">
                  <span className={toneText[word.tone]}>[{word.text}]</span> <span className="text-fg">{st.name.toLowerCase()}</span>
                  <p className="pl-[7ch] text-dim">{st.summary}</p>
                </div>
              );
            })}
          </div>
        ) : null}
      </Panel>
    </div>
  );
}
