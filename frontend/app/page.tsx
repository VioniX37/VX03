"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { API_BASE, api } from "./_components/api";
import type { MLRecommendation, ModelMetadata, Preset, PresolveData, SolveResult, SystemInfo, ViewId } from "./_components/types";
import { Btn, Comment, Kbd, Panel, Select, Spinner, Toggle, ViewTabs, fmt, toneText } from "./_components/term";
import {
  CertificateView,
  ModelView,
  PresolveView,
  ResultStrip,
  RunLog,
  SolutionView,
  SolveView,
  StrategyView,
  statusLabel,
  statusTone,
} from "./_components/views";
import { BenchmarksView } from "./_components/bench";
import { DemoView } from "./_components/demo";

const METHODS = [
  { value: "auto", label: "auto (engine picks)" },
  { value: "simplex", label: "simplex          lp" },
  { value: "dual_simplex", label: "dual_simplex     lp" },
  { value: "interior_point", label: "interior_point   lp" },
  { value: "pdlp", label: "pdlp (gpu-ready)  lp" },
  { value: "hybrid_pdlp", label: "hybrid_pdlp      lp" },
  { value: "concurrent", label: "concurrent (multi-core race)" },
  { value: "branch_and_bound", label: "branch_and_bound milp, miqp" },
  { value: "qp_interior_point", label: "qp_interior_point qp" },
  { value: "active_set", label: "active_set       qp" },
];

const VIEWS: Array<{ id: ViewId; label: string }> = [
  { id: 1, label: "model" },
  { id: 2, label: "presolve" },
  { id: 3, label: "strategy" },
  { id: 4, label: "solve" },
  { id: 5, label: "solution" },
  { id: 6, label: "certificate" },
];

const CLASS_ORDER = ["LP", "QP", "MILP", "MIQP"];

// demo models first (grouped by class), then the public benchmark library (grouped by collection)
const isLibrary = (p: Preset) => p.id.includes("/");
const groupLabel = (p: Preset) => (isLibrary(p) ? p.category.toLowerCase() : p.problem_class.toLowerCase());
const groupRank = (p: Preset) => (isLibrary(p) ? 10 : 0) + Math.max(0, CLASS_ORDER.indexOf(p.problem_class));

type Busy = null | "load" | "presolve" | "recommend" | "solve";

const BUSY_TEXT: Record<Exclude<Busy, null>, string> = {
  load: "loading model",
  presolve: "running presolve",
  recommend: "asking the strategy engine",
  solve: "presolve > strategy > solve > postsolve > certify",
};

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function Workbench() {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [cursor, setCursor] = useState(0);
  const [uploadedName, setUploadedName] = useState<string | null>(null);

  const [model, setModel] = useState<ModelMetadata | null>(null);
  const [presolve, setPresolve] = useState<PresolveData | null>(null);
  const [rec, setRec] = useState<MLRecommendation | null>(null);
  const [result, setResult] = useState<SolveResult | null>(null);
  const [runId, setRunId] = useState(0);
  const [lastCommand, setLastCommand] = useState("");

  const [view, setView] = useState<ViewId>(1);
  const [page, setPage] = useState<"workbench" | "benchmarks" | "demo">("workbench");
  const [algorithm, setAlgorithm] = useState("auto");
  const [timeLimit, setTimeLimit] = useState("60");
  const [enablePresolve, setEnablePresolve] = useState(true);
  const [busy, setBusy] = useState<Busy>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const fetchedFor = useRef(new Set<string>());

  const ordered = useMemo(
    () => [...presets].sort((a, b) => groupRank(a) - groupRank(b)),
    [presets],
  );

  const clearDerived = () => {
    setPresolve(null);
    setRec(null);
    setResult(null);
    fetchedFor.current.clear();
  };

  const loadPreset = useCallback(async (id: string) => {
    setBusy("load");
    setError(null);
    clearDerived();
    setSelected(id);
    setUploadedName(null);
    try {
      setModel(await api.loadPreset(id));
      setView(1);
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(null);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    api
      .systemInfo()
      .then((info) => {
        if (cancelled) return;
        setSystemInfo(info);
        setOnline(true);
      })
      .catch(() => {
        if (!cancelled) setOnline(false);
      });
    api
      .presets()
      .then((list) => {
        if (cancelled) return;
        setPresets(list);
        const sorted = [...list].sort((a, b) => groupRank(a) - groupRank(b));
        const index = Math.max(0, sorted.findIndex((p) => p.id === "refinery_blending_lp"));
        setCursor(index);
        if (sorted[index]) void loadPreset(sorted[index].id);
      })
      .catch((e) => {
        if (cancelled) return;
        setOnline(false);
        setError(message(e));
      });
    return () => {
      cancelled = true;
    };
  }, [loadPreset]);

  const upload = async (file: File) => {
    setBusy("load");
    setError(null);
    clearDerived();
    try {
      setModel(await api.upload(file));
      setSelected(null);
      setUploadedName(file.name);
      setView(1);
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(null);
    }
  };

  const runPresolve = useCallback(async () => {
    setBusy("presolve");
    setError(null);
    try {
      setPresolve(await api.presolve());
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(null);
    }
  }, []);

  const runRecommend = useCallback(async () => {
    setBusy("recommend");
    setError(null);
    try {
      setRec(await api.recommend());
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(null);
    }
  }, []);

  const command = useMemo(() => {
    const parts = ["python -m sovereign_opt.cli.app"];
    if (uploadedName) parts.push("solve", uploadedName);
    else parts.push("demo", "--preset", selected ?? "<preset>");
    if (algorithm !== "auto") parts.push("--algorithm", algorithm);
    if (!enablePresolve) parts.push("--no-presolve");
    return parts.join(" ");
  }, [uploadedName, selected, algorithm, enablePresolve]);

  const solve = useCallback(async () => {
    if (!model || busy) return;
    setBusy("solve");
    setError(null);
    setLastCommand(command);
    try {
      const r = await api.solve(algorithm, enablePresolve, Number(timeLimit));
      setResult(r);
      setRunId((n) => n + 1);
      setView(4);
      // fill the presolve and strategy views for this run in the background
      void Promise.allSettled([enablePresolve ? api.presolve() : Promise.resolve(null), api.recommend()]).then(([p, m]) => {
        if (p.status === "fulfilled" && p.value) setPresolve(p.value);
        if (m.status === "fulfilled") setRec(m.value);
      });
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(null);
    }
  }, [model, busy, algorithm, enablePresolve, command, timeLimit]);

  const openView = useCallback(
    (next: ViewId) => {
      setView(next);
      if (!model || busy) return;
      const key = `${next}:${model.name}`;
      if (fetchedFor.current.has(key)) return;
      if (next === 2 && !presolve) {
        fetchedFor.current.add(key);
        void runPresolve();
      } else if (next === 3 && !rec) {
        fetchedFor.current.add(key);
        void runRecommend();
      }
    },
    [model, busy, presolve, rec, runPresolve, runRecommend],
  );

  // keyboard: 1-6 views, j/k move through problems, enter opens, ctrl+enter runs
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      const typing = !!el && ["INPUT", "SELECT", "TEXTAREA"].includes(el.tagName);
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        if (page !== "workbench") return;
        e.preventDefault();
        void solve();
        return;
      }
      if (typing || e.ctrlKey || e.metaKey || e.altKey || page !== "workbench") return;
      if (e.key >= "1" && e.key <= "6") {
        openView(Number(e.key) as ViewId);
      } else if (e.key === "j") {
        setCursor((c) => Math.min(c + 1, Math.max(ordered.length - 1, 0)));
      } else if (e.key === "k") {
        setCursor((c) => Math.max(c - 1, 0));
      } else if (e.key === "Enter" && el?.tagName !== "BUTTON") {
        const p = ordered[cursor];
        if (p && p.id !== selected) void loadPreset(p.id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [solve, openView, ordered, cursor, selected, loadPreset, page]);

  const copyCommand = () => {
    void navigator.clipboard?.writeText(command).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  };

  const current = presets.find((p) => p.id === selected) ?? null;
  const views = VIEWS.map((v) => ({
    ...v,
    done:
      v.id === 1 ? !!model : v.id === 2 ? !!presolve : v.id === 3 ? !!rec : v.id === 6 ? !!result?.trust_certificate.is_valid : !!result,
  }));

  return (
    <div className="flex min-h-screen flex-col">
      {/* status bar */}
      <header className="sticky top-0 z-40 border-b border-line bg-bg/95 backdrop-blur-sm">
        <div className="mx-auto flex h-10 max-w-[1360px] items-center gap-3 px-4 text-[13px] sm:px-6">
          <span className="text-amber">sovereign</span>
          <span className="text-faint">{systemInfo?.engine_version ? `v${systemInfo.engine_version}` : ""}</span>
          <span className="text-faint">│</span>
          <span className="flex gap-1" role="tablist" aria-label="page">
            {(["demo", "workbench", "benchmarks"] as const).map((p) => (
              <button
                key={p}
                role="tab"
                aria-selected={page === p}
                onClick={() => setPage(p)}
                className={`px-1.5 ${page === p ? "bg-fg text-bg" : "text-dim hover:text-fg"}`}
              >
                {p}
              </button>
            ))}
          </span>
          <span className="text-faint">│</span>
          <span className={online === null ? "text-dim" : online ? "text-green" : "text-red"}>
            ● {online === null ? "connecting" : online ? "api online" : "api offline"}
          </span>
          <span className="hidden truncate text-faint sm:inline">{API_BASE}</span>
          <span className="ml-auto hidden text-dim md:inline">
            {systemInfo ? `${systemInfo.cpu_cores} cores · ${systemInfo.has_cuda ? systemInfo.cuda_device_name ?? "gpu" : "cpu"}` : ""}
          </span>
        </div>
      </header>

      {page === "demo" ? (
        <main className="demo-root mx-auto w-full max-w-[1360px] flex-1 px-4 pb-10 pt-7 sm:px-6">
          <DemoView />
        </main>
      ) : page === "benchmarks" ? (
        <main className="mx-auto w-full max-w-[1360px] flex-1 px-4 pb-10 pt-7 sm:px-6">
          <BenchmarksView />
        </main>
      ) : (
      <main className="mx-auto grid w-full max-w-[1360px] flex-1 gap-6 px-4 pb-10 pt-7 sm:px-6 lg:grid-cols-[250px_minmax(0,1fr)]">
        {/* problem list */}
        <aside className="lg:sticky lg:top-16 lg:self-start">
          <Panel title="problems">
            <ul aria-label="problems" className="-mx-1 max-h-[62vh] overflow-y-auto text-[13.5px]">
              {presets.length === 0 && online !== false ? (
                <li className="px-1 text-dim">
                  <Spinner /> loading
                </li>
              ) : null}
              {ordered.map((p, i) => {
                const active = p.id === selected;
                const newGroup = i === 0 || groupLabel(ordered[i - 1]) !== groupLabel(p);
                const firstLibrary = isLibrary(p) && (i === 0 || !isLibrary(ordered[i - 1]));
                return (
                  <li key={p.id}>
                    {firstLibrary ? <p className="mt-4 border-t border-line px-1 pt-3 text-[12px] text-amber">public benchmark library</p> : null}
                    {newGroup ? <p className={`px-1 text-[12px] text-faint ${i === 0 ? "" : "mt-2.5"}`}># {groupLabel(p)}</p> : null}
                    <button
                      title={p.name}
                      onClick={() => {
                        setCursor(i);
                        void loadPreset(p.id);
                      }}
                      disabled={busy === "load"}
                      className={`flex w-full items-baseline gap-1 whitespace-pre px-1 py-[1px] text-left transition-colors duration-100 ${
                        active ? "bg-fg text-bg" : cursor === i ? "bg-line-soft text-fg" : "text-dim hover:text-fg"
                      }`}
                    >
                      <span className={active ? "text-bg" : "text-amber"}>{active ? ">" : cursor === i ? "·" : " "}</span>
                      <span className="truncate">{isLibrary(p) ? p.id.split("/")[1] : p.id}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
            <label className="mt-4 block cursor-pointer border-t border-line pt-3 text-[13.5px] text-dim hover:text-fg">
              <span className="text-faint">[</span>open .mps / .lp<span className="text-faint">]</span>
              <input
                type="file"
                accept=".mps,.qps,.lp"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void upload(f);
                  e.target.value = "";
                }}
              />
            </label>
          </Panel>
          <p className="mt-2 hidden px-1 text-[12px] text-faint lg:block">
            <Kbd>j</Kbd> <Kbd>k</Kbd> move · <Kbd>enter</Kbd> open
          </p>
        </aside>

        <div className="min-w-0 space-y-6">
          {/* session */}
          <Panel title="session" right={model ? model.problem_class.toLowerCase() : undefined}>
            <div className={`transition-opacity duration-200 ${busy === "load" ? "opacity-50" : ""}`}>
              <p className="text-[19px] leading-snug text-fg">
                {current?.name ?? uploadedName ?? model?.name ?? "loading model"}
              </p>
              {current?.description ? <Comment className="mt-1">{current.description}</Comment> : null}
              {model ? (
                <p className="mt-2 text-[13px] text-dim">
                  {fmt.int(model.num_variables)} variables · {fmt.int(model.num_constraints)} constraints · {fmt.int(model.num_nonzeros)} nonzeros
                  {model.num_quadratic_terms ? ` · ${fmt.int(model.num_quadratic_terms)} quadratic terms` : ""}
                </p>
              ) : null}
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-3 border-t border-line pt-3">
              <span className="flex items-center gap-2 text-[13.5px] text-dim">
                method
                <Select label="method" value={algorithm} onChange={setAlgorithm} options={METHODS} />
              </span>
              <span className="flex items-center gap-2 text-[13.5px] text-dim">
                limit
                <Select
                  label="time limit"
                  value={timeLimit}
                  onChange={setTimeLimit}
                  options={[
                    { value: "60", label: "60 s" },
                    { value: "300", label: "5 min" },
                    { value: "900", label: "15 min" },
                  ]}
                />
              </span>
              <Toggle checked={enablePresolve} onChange={setEnablePresolve} label="presolve" />
              <span className="flex-1" />
              <Btn variant="primary" busy={busy === "solve"} disabled={!model || (busy !== null && busy !== "solve")} onClick={() => void solve()}>
                {busy === "solve" ? "running" : "run"}
                {busy === "solve" ? null : <span className="text-[12px] font-normal text-bg/60">ctrl+enter</span>}
              </Btn>
            </div>
          </Panel>

          {/* output */}
          <Panel
            title="output"
            right={
              <Btn onClick={copyCommand} aria-label="copy the equivalent CLI command">
                {copied ? "copied" : "copy command"}
              </Btn>
            }
          >
            {error ? (
              <div className="animate-type mb-3 flex items-baseline justify-between gap-4 text-[13.5px]">
                <p className="text-red">error: {error}</p>
                <Btn onClick={() => setError(null)}>dismiss</Btn>
              </div>
            ) : null}
            {busy === "solve" ? (
              <div className="space-y-0.5 text-[13.5px]">
                <p className="whitespace-pre-wrap break-all">
                  <span className="text-amber">$ </span>
                  {lastCommand}
                </p>
                <p className="text-dim">
                  <Spinner /> {BUSY_TEXT.solve}
                </p>
              </div>
            ) : result ? (
              <>
                <RunLog key={`log-${runId}`} result={result} command={lastCommand || command} />
                <div className="mt-4 border-t border-line pt-4">
                  <ResultStrip key={`strip-${runId}`} result={result} />
                </div>
              </>
            ) : (
              <div className="space-y-1 text-[13.5px]">
                <p className="whitespace-pre-wrap break-all">
                  <span className="text-amber">$ </span>
                  {command}
                  <span className="cursor ml-1" aria-hidden />
                </p>
                <Comment>press run or ctrl+enter to solve this model. the same command works in your terminal.</Comment>
              </div>
            )}
          </Panel>

          {/* views */}
          <div>
            <ViewTabs items={views} active={view} onChange={openView} />
            <div key={`${view}-${model?.name ?? ""}-${runId}`} className="animate-rise pt-6">
              {!model ? (
                <Comment>{online === false ? `the solver api is not reachable at ${API_BASE}.` : "loading model"}</Comment>
              ) : view === 1 ? (
                <ModelView model={model} />
              ) : view === 2 ? (
                <PresolveView presolve={presolve} onRun={() => void runPresolve()} busy={busy === "presolve"} />
              ) : view === 3 ? (
                <StrategyView rec={rec} onRun={() => void runRecommend()} busy={busy === "recommend"} />
              ) : view === 4 ? (
                <SolveView result={result} runId={runId} />
              ) : view === 5 ? (
                <SolutionView result={result} />
              ) : (
                <CertificateView result={result} />
              )}
            </div>
          </div>
        </div>
      </main>
      )}

      {/* mode line */}
      <footer
        className={`sticky bottom-0 z-40 border-t border-line bg-bg/95 backdrop-blur-sm ${page === "demo" ? "hidden" : ""}`}
      >
        <div className="mx-auto flex h-8 max-w-[1360px] items-center gap-3 px-4 text-[12.5px] sm:px-6">
          <span className="bg-amber px-1.5 text-bg">{VIEWS[view - 1].label}</span>
          {busy ? (
            <span className="text-dim">
              <Spinner /> {BUSY_TEXT[busy]}
            </span>
          ) : result ? (
            <span className="text-dim">
              last run: <span className={toneText[statusTone(result.status)]}>{statusLabel(result.status)}</span> in{" "}
              {fmt.dur(result.timings?.total ?? result.runtime_seconds)}
            </span>
          ) : (
            <span className="text-dim">ready</span>
          )}
          <span className="ml-auto hidden text-faint md:inline">
            <Kbd>1</Kbd>-<Kbd>6</Kbd> views · <Kbd>ctrl+enter</Kbd> run · <Kbd>j</Kbd>/<Kbd>k</Kbd> problems
          </span>
        </div>
      </footer>
    </div>
  );
}
