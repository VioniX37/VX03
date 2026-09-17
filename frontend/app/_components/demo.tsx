"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { Btn, Comment, Leader, Panel, Select, Spinner, Toggle, fmt } from "./term";
import type { DemoInstance, DemoJob, DemoRecordedEntry, DemoSolverResult, DemoSolverSpec } from "./types";

const POLL_MS = 500;
const STORE_KEY = "demo.jobs";

const SOLVER_TONE: Record<string, string> = {
  ours: "text-amber",
  highs: "text-dim",
  gurobi: "text-cyan",
};
const SOLVER_BAR: Record<string, string> = {
  ours: "bg-amber",
  highs: "bg-faint",
  gurobi: "bg-cyan",
};

const recorded = (inst: DemoInstance | undefined, key: string): DemoRecordedEntry | null => {
  const r = inst?.recorded?.[key];
  return r && typeof r === "object" ? (r as DemoRecordedEntry) : null;
};

/** 0.094 means we are 10.6x slower; say so in words rather than printing a fraction. */
function ratioPhrase(ours: number | null | undefined, other: number | null | undefined) {
  if (!ours || !other) return null;
  const r = other / ours;
  return r >= 1
    ? { n: r, word: "faster", tone: "text-amber" }
    : { n: 1 / r, word: "slower", tone: "text-dim" };
}

// ------------------------------------------------------------------ job polling
function readStore(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY) ?? "{}");
  } catch {
    return {};
  }
}

function writeStore(next: Record<string, string>) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(next));
  } catch {
    /* private window: the demo still works, it just cannot reattach after a reload */
  }
}

/**
 * Polls one race job. Long instances run for minutes, so this never gives up on a transient
 * fetch failure -- it backs off and keeps trying, which is what a browser reload mid-demo needs.
 */
function useRaceJobs() {
  const [jobs, setJobs] = useState<Record<string, DemoJob>>({});
  const ids = useRef<Record<string, string>>({});
  const fails = useRef(0);

  useEffect(() => {
    ids.current = readStore();
    const boot = Object.entries(ids.current);
    if (!boot.length) return;
    Promise.allSettled(boot.map(([, id]) => api.demoJob(id))).then((rs) => {
      const next: Record<string, DemoJob> = {};
      rs.forEach((r) => {
        if (r.status === "fulfilled") next[r.value.instance_id] = r.value;
      });
      setJobs((prev) => ({ ...next, ...prev }));
    });
  }, []);

  useEffect(() => {
    const live = Object.values(jobs).filter((j) => j.state === "running");
    if (!live.length) return;
    const delay = fails.current > 2 ? 2000 : document.hidden ? 2000 : POLL_MS;
    const t = setTimeout(async () => {
      const rs = await Promise.allSettled(live.map((j) => api.demoJob(j.job_id)));
      const ok = rs.filter((r) => r.status === "fulfilled") as PromiseFulfilledResult<DemoJob>[];
      fails.current = ok.length === rs.length ? 0 : fails.current + 1;
      if (ok.length) setJobs((prev) => ({ ...prev, ...Object.fromEntries(ok.map((r) => [r.value.instance_id, r.value])) }));
    }, delay);
    return () => clearTimeout(t);
  }, [jobs]);

  const track = useCallback((job: DemoJob) => {
    ids.current = { ...ids.current, [job.instance_id]: job.job_id };
    writeStore(ids.current);
    setJobs((prev) => ({ ...prev, [job.instance_id]: job }));
  }, []);

  return { jobs, track };
}

/** Smooth client-side clock so a bar keeps growing between 500 ms polls. */
function useTicker(active: boolean) {
  const [, force] = useState(0);
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => force((n) => n + 1), 100);
    return () => clearInterval(t);
  }, [active]);
}

// ------------------------------------------------------------------ pieces
function ProblemCard({ inst }: { inst: DemoInstance }) {
  const s = inst.size;
  return (
    <Panel title={`${inst.letter} · ${inst.problem_class.toLowerCase()}`} right={inst.industry}>
      <h2 className="text-[26px] leading-tight text-fg sm:text-[30px]">{inst.title}</h2>
      <Comment className="mt-2 max-w-[75ch] !text-[15px]">{inst.story}</Comment>
      <div className="mt-4 grid gap-x-10 sm:grid-cols-2 lg:grid-cols-4">
        <Leader label="constraints">{fmt.int(s.rows)}</Leader>
        <Leader label="variables">{fmt.int(s.cols)}</Leader>
        <Leader label="nonzeros">{fmt.int(s.nnz)}</Leader>
        <Leader label={s.int_vars ? "binaries" : "type"}>{s.int_vars ? fmt.int(s.int_vars) : "continuous"}</Leader>
      </div>
    </Panel>
  );
}

function SolverRow({
  spec,
  res,
  rec,
  scale,
}: {
  spec: DemoSolverSpec;
  res: DemoSolverResult | undefined;
  rec: DemoRecordedEntry | null;
  scale: number;
}) {
  const state = res?.state ?? "queued";
  const running = state === "running";
  useTicker(running);

  const live = running && res?.elapsed ? res.elapsed : null;
  const done = state === "done" ? res?.solve_time ?? null : null;
  const shown = done ?? live;
  // while running the clock includes process start-up; clamp so the bar never overflows
  const width = shown ? Math.min(100, (shown / scale) * 100) : 0;
  const recPct = rec?.time ? Math.min(100, (rec.time / scale) * 100) : null;

  const tone = SOLVER_TONE[spec.key] ?? "text-fg";
  const skipped = state === "skipped";

  return (
    <div className="py-3">
      <div className="flex items-baseline justify-between gap-4">
        <span className={`text-[15px] ${tone}`}>{spec.label}</span>
        <span className="shrink-0 tabular-nums">
          {state === "error" ? (
            <span className="text-[15px] text-red">failed</span>
          ) : skipped ? (
            <span className="text-[15px] text-faint">not run</span>
          ) : done ? (
            <span className={`text-[32px] leading-none sm:text-[44px] ${tone}`}>{fmt.dur(done)}</span>
          ) : running ? (
            <span className="text-[20px] leading-none text-dim sm:text-[28px]">
              <Spinner className="mr-2 inline" />
              {fmt.dur(live ?? 0)}
            </span>
          ) : (
            <span className="text-[15px] text-faint">queued</span>
          )}
        </span>
      </div>

      <div className="relative mt-2 h-4 w-full border border-line-soft bg-panel">
        {recPct !== null ? (
          <span
            aria-hidden
            className="absolute top-0 h-full border-l border-dashed border-faint/80"
            style={{ left: `${recPct}%` }}
            title={`recorded: ${fmt.dur(rec?.time)}`}
          />
        ) : null}
        <span
          className={`block h-full ${SOLVER_BAR[spec.key] ?? "bg-fg"} transition-[width] duration-150 ease-linear`}
          style={{ width: `${width}%`, opacity: running ? 0.55 : 1 }}
        />
      </div>

      <p className="mt-1 text-[12.5px] text-faint">
        {skipped ? (
          spec.skip_reason ?? res?.note ?? "not run"
        ) : state === "error" ? (
          <span className="text-red">{res?.error?.slice(0, 180)}</span>
        ) : running ? (
          "elapsed includes process start-up; the measured solve time replaces it on finish"
        ) : done ? (
          [
            res?.algorithm ? `algorithm ${res.algorithm}` : null,
            res?.build_time ? `build ${fmt.dur(res.build_time)}` : null,
            res?.iterations ? `${fmt.int(res.iterations)} iterations` : null,
            res?.nodes ? `${fmt.int(res.nodes)} nodes` : null,
            res?.certificate ? `certificate ${res.certificate}` : null,
            res?.version ? `v${res.version}` : null,
          ]
            .filter(Boolean)
            .join(" · ")
        ) : (
          ""
        )}
      </p>
    </div>
  );
}

function RacePanel({ inst, job }: { inst: DemoInstance; job: DemoJob | undefined }) {
  const times: number[] = [];
  inst.solvers.forEach((s) => {
    const r = job?.solvers?.[s.key];
    if (r?.state === "done" && r.solve_time) times.push(r.solve_time);
    if (r?.state === "running" && r.elapsed) times.push(r.elapsed);
    const rec = recorded(inst, s.key);
    if (rec?.time) times.push(rec.time);
  });
  const scale = Math.max(...times, 0.5) * 1.05;

  return (
    <Panel
      title="wall clock"
      right={job ? `source: ${job.source}${job.source_file ? ` · ${job.source_file}` : ""}` : "not run"}
    >
      <Comment>
        one solver at a time, each with the whole machine. HiGHS and Gurobi are pinned to a single thread.
        bars are on a linear scale — full width = {fmt.dur(scale)}, dashed tick = previously recorded time.
      </Comment>
      <div className="mt-2 divide-y divide-line-soft">
        {inst.solvers.map((s) => (
          <SolverRow key={s.key} spec={s} res={job?.solvers?.[s.key]} rec={recorded(inst, s.key)} scale={scale} />
        ))}
      </div>
    </Panel>
  );
}

function ObjectiveMatch({ inst, job }: { inst: DemoInstance; job: DemoJob | undefined }) {
  const m = job?.objective_match;
  const keys = Object.keys(m?.values ?? {});
  const isMip = (inst.size.int_vars ?? 0) > 0;
  // PDLP is a first-order method run to a 1e-4 relative KKT tolerance, so an LP it solves
  // lands near there by design; simplex/interior-point and MILP answers are exact to rounding.
  const tol = isMip ? 1e-4 : inst.problem_class === "LP" ? 5e-4 : 1e-6;
  const worst = keys.length ? Math.max(...keys.map((k) => m?.rel_error?.[k] ?? 0)) : null;

  return (
    <Panel title="objective agreement" right={m?.reference ? `reference: ${m.reference}` : "—"}>
      {keys.length < 2 ? (
        <Comment>run at least two solvers to compare the optimal values.</Comment>
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[620px] text-[15px] tabular-nums">
              <tbody>
                {keys.map((k) => {
                  const err = m?.rel_error?.[k];
                  const ok = err !== undefined && err <= tol;
                  return (
                    <tr key={k} className="border-b border-line-soft last:border-0">
                      <td className={`py-2 pr-6 ${SOLVER_TONE[k] ?? ""}`}>
                        {inst.solvers.find((s) => s.key === k)?.label ?? k}
                      </td>
                      <td className="py-2 pr-6 text-right font-mono text-[17px] text-fg">
                        {m!.values[k].toPrecision(16)}
                      </td>
                      <td className={`py-2 text-right text-[13px] ${ok ? "text-green" : "text-amber"}`}>
                        {err === 0 ? "exact" : `Δ ${fmt.sci(err, 1)}`}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <Comment className="mt-3">
            {worst !== null && worst <= 1e-12
              ? "identical to double precision — every solver found the same optimum."
              : `same optimum: the solvers agree to ${fmt.sci(worst, 1)} relative error` +
                (inst.problem_class === "LP" && worst !== null && worst > 1e-6
                  ? ", the first-order tolerance our PDLP solver is asked for."
                  : ".")}
          </Comment>
        </>
      )}
    </Panel>
  );
}

function SpeedupBadge({ inst, job }: { inst: DemoInstance; job: DemoJob | undefined }) {
  const ours = job?.solvers?.ours;
  const highs = job?.solvers?.highs;
  const gurobi = job?.solvers?.gurobi;
  const vsH = ratioPhrase(ours?.solve_time, highs?.state === "done" ? highs.solve_time : null);
  const vsG = ratioPhrase(ours?.solve_time, gurobi?.state === "done" ? gurobi.solve_time : null);
  if (!vsH && !vsG) return null;
  const lead = vsH ?? vsG!;

  return (
    <Panel title="headline">
      <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2">
        <span className={`text-[48px] leading-none sm:text-[64px] ${lead.tone}`}>
          {lead.n >= 100 ? Math.round(lead.n) : lead.n.toFixed(1)}x
        </span>
        <span className="text-[17px] text-fg">
          {lead.word} than {vsH ? "HiGHS" : "Gurobi"} on {fmt.int(inst.size.cols)} variables
          {vsH && vsG ? ` · ${vsG.n.toFixed(1)}x ${vsG.word} than Gurobi` : ""}
        </span>
      </div>
      {inst.notes?.honesty ? <Comment className="mt-3 max-w-[80ch] !text-[14px]">{inst.notes.honesty}</Comment> : null}
    </Panel>
  );
}

function GpuPanel({ inst }: { inst: DemoInstance }) {
  const g = inst.gpu_panel;
  if (!g) return null;
  const scale = Math.max(g.ours, g.highs, g.gurobi ?? 0) * 1.05;
  const rows: Array<[string, number, string]> = [
    ["sovereign PDLP on GPU", g.ours, "bg-green"],
    ...(g.gurobi ? ([["Gurobi (this laptop)", g.gurobi, "bg-cyan"]] as Array<[string, number, string]>) : []),
    ["HiGHS (1 thread)", g.highs, "bg-faint"],
  ];
  return (
    <Panel title={`the same model on a ${g.device}`} right={g.source}>
      <p className="text-[13.5px] text-amber">{g.note}</p>
      <div className="mt-3 space-y-3">
        {rows.map(([label, t, bar]) => (
          <div key={label}>
            <div className="flex items-baseline justify-between gap-4">
              <span className="text-[15px] text-dim">{label}</span>
              <span className="text-[28px] leading-none tabular-nums text-fg">{fmt.dur(t)}</span>
            </div>
            <div className="mt-1.5 h-4 w-full border border-line-soft bg-panel">
              <span className={`block h-full ${bar}`} style={{ width: `${(t / scale) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 flex flex-wrap items-baseline gap-x-5">
        <span className="text-[40px] leading-none text-green">{g.speedup.toFixed(1)}x</span>
        <span className="text-[15px] text-fg">
          faster than HiGHS on {fmt.int(g.cols)} variables — the CPU code path and the GPU code path are the
          same solver, only the backend changes.
        </span>
      </div>
    </Panel>
  );
}

// ------------------------------------------------------------------ view
export function DemoView() {
  const [instances, setInstances] = useState<DemoInstance[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [useRecorded, setUseRecorded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const { jobs, track } = useRaceJobs();

  useEffect(() => {
    api
      .demoInstances()
      .then((list) => {
        const visible = list.filter((i) => !i.hidden);
        setInstances(list);
        setSelected((prev) => prev ?? visible[0]?.id ?? null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const visible = useMemo(() => instances.filter((i) => !i.hidden), [instances]);
  const inst = useMemo(() => instances.find((i) => i.id === selected), [instances, selected]);
  const job = inst ? jobs[inst.id] : undefined;
  const running = job?.state === "running";

  const start = useCallback(
    async (id: string, force: boolean) => {
      setBusy(id);
      setError(null);
      try {
        track(await api.demoStart(id, useRecorded, force));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(null);
      }
    },
    [track, useRecorded],
  );

  const cancel = useCallback(async () => {
    if (!job) return;
    try {
      track(await api.demoCancel(job.job_id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [job, track]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName)) return;
      const n = Number(e.key);
      if (n >= 1 && n <= visible.length) {
        setSelected(visible[n - 1].id);
      } else if (e.key === "r" && selected && !running) {
        void start(selected, true);
      } else if (e.key === "c" && running) {
        void cancel();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible, selected, running, start, cancel]);

  const background = Object.values(jobs).filter((j) => j.state === "running" && j.instance_id !== selected);

  if (error && !instances.length) {
    return (
      <Panel title="demo">
        <p className="text-[14px] text-red">{error}</p>
      </Panel>
    );
  }
  if (!inst) return <Spinner />;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-3 border-b border-line pb-4">
        <Select
          label="instance"
          value={inst.id}
          onChange={setSelected}
          options={visible.map((i, n) => ({ value: i.id, label: `${n + 1}. ${i.letter} — ${i.title}` }))}
        />
        <Btn variant="primary" onClick={() => start(inst.id, true)} busy={busy === inst.id} disabled={running}>
          {running ? "racing" : "run race"}
        </Btn>
        {running ? <Btn onClick={cancel}>cancel</Btn> : null}
        {inst.long_running && !running && !jobs[inst.id] ? (
          <Btn onClick={() => start(inst.id, true)}>start in background</Btn>
        ) : null}
        <Toggle checked={useRecorded} onChange={setUseRecorded} label="use recorded results" />
        <span className="ml-auto text-[13px] text-faint">
          {background.length
            ? background
                .map((j) => {
                  const other = instances.find((i) => i.id === j.instance_id);
                  const active = Object.entries(j.solvers).find(([, v]) => v.state === "running");
                  return `${other?.letter ?? j.instance_id}: ${active?.[0] ?? "starting"} ${fmt.dur(active?.[1]?.elapsed)}`;
                })
                .join("  ·  ")
            : "keys: 1-4 instance · r run · c cancel"}
        </span>
      </div>

      {error ? <p className="text-[13.5px] text-red">{error}</p> : null}

      <ProblemCard inst={inst} />
      <RacePanel inst={inst} job={job} />
      <SpeedupBadge inst={inst} job={job} />
      <ObjectiveMatch inst={inst} job={job} />
      <GpuPanel inst={inst} />
    </div>
  );
}
