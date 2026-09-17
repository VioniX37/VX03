"use client";

import React, { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { LineChart, type ChartSeries } from "./charts";
import { Btn, Comment, Leader, Panel, Spinner, fmt, toneText, type Tone } from "./term";
import type { BenchRow, BenchSuite, Benchmarks, ScaleRow } from "./types";

const SUITES: Array<{ key: string; title: string; note: string }> = [
  { key: "netlib", title: "netlib lp", note: "the classic LP test set. reference = published optimum from the netlib readme." },
  { key: "netlib-large", title: "netlib lp (large) + kennington", note: "thousands of rows; includes degenerate and badly scaled models." },
  { key: "infeas", title: "netlib infeasible lp", note: "the engine must prove there is no solution, not just fail to find one." },
  { key: "qp", title: "maros-meszaros convex qp", note: "no published optima shipped with the set, so HiGHS is the reference." },
  { key: "qp-large", title: "maros-meszaros qp (large)", note: "" },
  { key: "miplib", title: "miplib 2017 milp", note: "reference = miplib .solu optimum. at the time limit we report the incumbent and gap." },
];

function tone(r: BenchRow): Tone {
  if (r.ok) return "ok";
  if (r.verdict?.startsWith("limit")) return "warn";
  return "bad";
}

function obj(v: number | null | undefined, status?: string | null) {
  if (status === "infeasible") return "infeasible";
  if (v === null || v === undefined) return "-";
  const a = Math.abs(v);
  return a !== 0 && (a >= 1e7 || a < 1e-3) ? v.toExponential(6) : v.toPrecision(10).replace(/\.?0+$/, "");
}

function Summary({ suite }: { suite: BenchSuite }) {
  const rows = suite.rows;
  const ok = rows.filter((r) => r.ok).length;
  const both = rows.filter((r) => r.ok && r.ours.time && r.highs.time);
  const geo = both.length ? Math.exp(both.reduce((s, r) => s + Math.log((r.ours.time as number) / (r.highs.time as number)), 0) / both.length) : null;
  const certified = rows.filter((r) => r.ours.certificate === "OPTIMAL_CERTIFIED").length;
  return (
    <div className="grid gap-x-8 sm:grid-cols-3">
      <Leader label="solved correctly">
        <span className={ok === rows.length ? "text-green" : "text-amber"}>
          {ok} / {rows.length}
        </span>
      </Leader>
      <Leader label="independently certified">{certified}</Leader>
      <Leader label="time vs HiGHS" hint="geo. mean">
        {geo === null ? "-" : `${geo.toFixed(1)}x`}
      </Leader>
    </div>
  );
}

function SuiteTable({ suite }: { suite: BenchSuite }) {
  const isMip = suite.rows.some((r) => (r.int_vars ?? 0) > 0);
  return (
    <div className="mt-3 overflow-x-auto">
      <table className="w-full min-w-[860px] text-[13px]">
        <thead>
          <tr className="border-b border-line text-left text-dim">
            <th className="py-1 pr-3 font-normal">instance</th>
            <th className="py-1 pr-3 text-right font-normal">rows x cols</th>
            <th className="py-1 pr-3 text-right font-normal">reference</th>
            <th className="py-1 pr-3 text-right font-normal">ours</th>
            <th className="py-1 pr-3 text-right font-normal" title="smaller of: error vs published optimum, error vs HiGHS">rel. err</th>
            <th className="py-1 pr-3 text-right font-normal">ours</th>
            <th className="py-1 pr-3 text-right font-normal">HiGHS</th>
            <th className="py-1 pr-3 text-right font-normal">{isMip ? "nodes" : "iters"}</th>
            <th className="py-1 font-normal">verdict</th>
          </tr>
        </thead>
        <tbody>
          {suite.rows.map((r) => {
            const errs = [r.rel_error, r.rel_error_vs_highs].filter((e): e is number => e !== null && e !== undefined);
            const err = errs.length ? Math.min(...errs) : null;
            return (
              <tr key={r.name} className="border-b border-line-soft">
                <td className="py-1 pr-3 text-fg">
                  {r.name}
                  {r.tags?.length ? <span className="text-faint"> {r.tags.join(" ")}</span> : null}
                  {r.note ? (
                    <span title={r.note} className="cursor-help text-amber">
                      {" "}*
                    </span>
                  ) : null}
                </td>
                <td className="py-1 pr-3 text-right text-dim">
                  {fmt.int(r.rows)} x {fmt.int(r.cols)}
                </td>
                <td className="py-1 pr-3 text-right text-dim">{obj(r.reference)}</td>
                <td className="py-1 pr-3 text-right">{obj(r.ours.objective, r.ours.status)}</td>
                <td className="py-1 pr-3 text-right text-dim">{err === null || err === undefined ? "-" : fmt.sci(err)}</td>
                <td className="py-1 pr-3 text-right">{fmt.dur(r.ours.time)}</td>
                <td className="py-1 pr-3 text-right text-dim">{fmt.dur(r.highs.time)}</td>
                <td className="py-1 pr-3 text-right text-dim">{fmt.int(isMip ? r.ours.nodes : r.ours.iterations)}</td>
                <td className={`py-1 ${toneText[tone(r)]}`}>{r.verdict}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RobustnessTable({ suite }: { suite: BenchSuite }) {
  return (
    <div className="mt-3 overflow-x-auto">
      <table className="w-full min-w-[900px] text-[13px]">
        <thead>
          <tr className="border-b border-line text-left text-dim">
            <th className="py-1 pr-3 font-normal">instance</th>
            <th className="py-1 pr-3 font-normal">why it is hard</th>
            <th className="py-1 pr-3 text-right font-normal">coef. range raw</th>
            <th className="py-1 pr-3 text-right font-normal">after scaling</th>
            <th className="py-1 pr-3 text-right font-normal">degenerate pivots</th>
            <th className="py-1 pr-3 text-right font-normal">perturb. / bland</th>
            <th className="py-1 pr-3 text-right font-normal">time</th>
            <th className="py-1 font-normal">verdict</th>
          </tr>
        </thead>
        <tbody>
          {suite.rows.map((r) => (
            <tr key={r.name} className="border-b border-line-soft">
              <td className="py-1 pr-3 text-fg">{r.name}</td>
              <td className="py-1 pr-3 text-dim">{r.tags?.join(", ")}</td>
              <td className="py-1 pr-3 text-right text-dim">{fmt.sci(r.range_before)}</td>
              <td className="py-1 pr-3 text-right">{fmt.sci(r.range_after)}</td>
              <td className="py-1 pr-3 text-right text-dim">{fmt.int(r.counters?.degenerate_pivots)}</td>
              <td className="py-1 pr-3 text-right text-dim">
                {fmt.int(r.counters?.bound_perturbations)} / {fmt.int(r.counters?.bland_pivots)}
              </td>
              <td className="py-1 pr-3 text-right">{fmt.dur(r.ours.time)}</td>
              <td className={`py-1 ${toneText[tone(r)]}`}>{r.verdict}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const METHOD_STYLE: Record<string, { stroke: string; text: string; label: string }> = {
  highs: { stroke: "stroke-dim", text: "text-dim", label: "HiGHS (1 thread)" },
  dual_simplex: { stroke: "stroke-fg", text: "text-fg", label: "dual simplex" },
  interior_point: { stroke: "stroke-red", text: "text-red", label: "interior point" },
  pdlp_cpu: { stroke: "stroke-amber", text: "text-amber", label: "pdlp cpu" },
  pdlp_cuda: { stroke: "stroke-green", text: "text-green", label: "pdlp gpu" },
};

function ScalePanel({ kind, suite }: { kind: string; suite: BenchSuite }) {
  const rows = suite.rows as unknown as ScaleRow[];
  const methods = useMemo(() => Array.from(new Set(rows.flatMap((r) => Object.keys(r.results)))), [rows]);
  const series: ChartSeries[] = methods
    .map((m) => ({
      name: METHOD_STYLE[m]?.label ?? m,
      stroke: METHOD_STYLE[m]?.stroke ?? "stroke-dim",
      text: METHOD_STYLE[m]?.text ?? "text-dim",
      points: rows
        .filter((r) => ["optimal"].includes(String(r.results[m]?.status)) && r.results[m]?.time)
        .map((r) => ({ x: Math.round(Math.log10(r.cols) * 10) / 10, y: r.results[m].time as number })),
    }))
    .filter((s) => s.points.length > 1);
  const gpu = suite.meta?.machine?.gpu;
  return (
    <Panel title={`scaling · ${kind}`} right={gpu && gpu !== "None" ? `measured on ${gpu}` : `${suite.meta?.machine?.cpu_count ?? "?"} cpu threads`}>
      <Comment>
        production-distribution lp from benchmarks/industrial/supply_chain.py. x axis: log10(variables), y axis: seconds (log). pdlp stops at
        relative tolerance {suite.meta?.pdlp_tol ?? "1e-4"}; its objective error vs HiGHS is in the table.
      </Comment>
      {series.length ? <div className="mt-3"><LineChart series={series} log height={220} ariaLabel="solve time by model size" /></div> : null}
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[760px] text-[13px]">
          <thead>
            <tr className="border-b border-line text-left text-dim">
              <th className="py-1 pr-3 font-normal">variables</th>
              <th className="py-1 pr-3 text-right font-normal">rows</th>
              <th className="py-1 pr-3 text-right font-normal">nonzeros</th>
              {methods.map((m) => (
                <th key={m} className="py-1 pr-3 text-right font-normal">
                  {METHOD_STYLE[m]?.label ?? m}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.cols} className="border-b border-line-soft">
                <td className="py-1 pr-3 text-fg">{fmt.int(r.cols)}</td>
                <td className="py-1 pr-3 text-right text-dim">{fmt.int(r.rows)}</td>
                <td className="py-1 pr-3 text-right text-dim">{fmt.int(r.nnz)}</td>
                {methods.map((m) => {
                  const x = r.results[m];
                  const okay = x?.status === "optimal";
                  return (
                    <td key={m} className={`py-1 pr-3 text-right ${okay ? "text-fg" : "text-faint"}`}>
                      {okay ? fmt.dur(x.time) : String(x?.status ?? "-").replace(" (size)", "")}
                      {okay && x.rel_error_vs_highs !== undefined && m !== "highs" ? (
                        <span className="text-faint"> {fmt.sci(x.rel_error_vs_highs)}</span>
                      ) : null}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

export function BenchmarksView() {
  const [data, setData] = useState<Benchmarks | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchResults = () =>
    api
      .benchmarks()
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  const load = () => {
    setLoading(true);
    void fetchResults();
  };
  useEffect(() => {
    void fetchResults();
  }, []);

  if (!data) {
    return (
      <Panel title="benchmarks">
        {error ? <p className="text-red">error: {error}</p> : <p className="text-dim"><Spinner /> loading saved results</p>}
      </Panel>
    );
  }
  const scaleKinds = Object.keys(data).filter((k) => k.startsWith("scale")).sort();
  const known = new Set(SUITES.map((s) => s.key));
  const tagged = Object.keys(data)
    .filter((k) => !known.has(k) && !k.startsWith("scale") && k !== "robustness" && data[k]?.rows?.length)
    .sort()
    .map((k) => {
      const gpu = data[k].meta?.machine?.gpu;
      return {
        key: k,
        title: k.replace(/_/g, " "),
        note: `tagged run${gpu && gpu !== "None" ? `, measured on ${gpu}` : ""}${(data[k].meta as { algorithm?: string })?.algorithm ? ` with method ${(data[k].meta as { algorithm?: string }).algorithm}` : ""}.`,
      };
    });
  const present = [...SUITES.filter((s) => data[s.key]?.rows?.length), ...tagged];
  return (
    <div className="space-y-6">
      <Panel title="public benchmarks vs HiGHS" right={<Btn onClick={load} busy={loading}>reload</Btn>}>
        <Comment>
          every instance is read from the official collection files by our own parser, solved by the sovereign engine, checked against the published
          optimum and against HiGHS (open-source, single thread), and certified independently. regenerate with python -m benchmarks.compare --suite
          &lt;name&gt;.
        </Comment>
        {present.length === 0 ? <Comment className="mt-3">no saved results yet. run python -m benchmarks.compare --suite netlib.</Comment> : null}
      </Panel>
      {present.map((s) => (
        <Panel key={s.key} title={s.title} right={data[s.key].file}>
          {s.note ? <Comment>{s.note}</Comment> : null}
          <div className="mt-3">
            <Summary suite={data[s.key]} />
          </div>
          <SuiteTable suite={data[s.key]} />
          {data[s.key].rows
            .filter((r) => r.note)
            .map((r) => (
              <Comment key={r.name} className="mt-2">
                <span className="text-amber">*</span> {r.name}: {r.note}
              </Comment>
            ))}
        </Panel>
      ))}
      {data.robustness?.rows?.length ? (
        <Panel title="numerical robustness" right={data.robustness.file}>
          <Comment>
            hard models grouped by why they are hard. coefficient range = largest / smallest |a| before and after the engine&apos;s scaling. degenerate
            pivots make no progress; the engine counters them with bound perturbation and bland&apos;s rule.
          </Comment>
          <RobustnessTable suite={data.robustness} />
        </Panel>
      ) : null}
      {scaleKinds.map((k) => (
        <ScalePanel key={k} kind={k} suite={data[k]} />
      ))}
    </div>
  );
}
