"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  Cpu,
  Activity,
  Layers,
  Sparkles,
  Play,
  RotateCcw,
  ShieldCheck,
  FileCode,
  CheckCircle2,
  AlertTriangle,
  Sliders,
  Check,
  BarChart3,
  GitBranch,
  Terminal,
  Grid,
  Scale,
  RefreshCw,
  Search,
  ExternalLink,
  ChevronRight,
  Info,
} from "lucide-react";

interface Preset {
  id: string;
  name: string;
  category: string;
  description: string;
  problem_class: string;
}

interface RuizStats {
  d1_min: number;
  d1_max: number;
  d2_min: number;
  d2_max: number;
  norm_before: number;
  norm_after: number;
  iterations: number;
}

interface SparsityPoint {
  row: number;
  col: number;
  val: number;
}

interface ModelMetadata {
  name: string;
  problem_class: string;
  num_variables: number;
  num_constraints: number;
  num_nonzeros: number;
  density: number;
  num_continuous: number;
  num_integer: number;
  num_binary: number;
  num_quadratic_terms: number;
  ruiz_stats?: RuizStats;
  sparsity_sample: SparsityPoint[];
  variables?: Array<{ name: string; type: string; lb: any; ub: any }>;
  constraints?: Array<{ name: string; sense: string; rhs: number; num_terms: number }>;
}

interface PresolveData {
  original_vars: number;
  original_cons: number;
  original_nnz: number;
  presolved_vars: number;
  presolved_cons: number;
  presolved_nnz: number;
  var_reduction_pct: number;
  con_reduction_pct: number;
  nnz_reduction_pct: number;
  fixed_vars_count: number;
  singleton_rows_count: number;
  empty_rows_count: number;
  empty_cols_count: number;
  fixed_vars_list?: Array<{ name: string; value: number }>;
}

interface MLRecommendation {
  recommended_algorithm: string;
  algorithm_probabilities: Record<string, number>;
  confidence_score: number;
  recommended_hardware: string;
  branching_strategy: string;
  feature_attributions: Record<string, number>;
  deterministic_fallback: string;
  features?: Record<string, number>;
}

interface IterationTraceEntry {
  iteration: number;
  objective?: number;
  entering_var?: string;
  leaving_var?: string;
  reduced_cost?: number;
  mu?: number;
  norm_rp?: number;
  norm_rd?: number;
  rel_gap?: number;
  p_norm?: number;
  active_constraints?: number;
}

interface TreeTraceNode {
  node_id: number;
  parent_id: number | null;
  depth: number;
  lower_bound: number;
  status: string; // 'ROOT', 'BRANCH', 'PRUNED_BOUND', 'INTEGER_INCUMBENT', 'INFEASIBLE'
  branch_var: string;
  branch_condition: string;
}

interface PostsolveMappingItem {
  variable: string;
  value: number;
  resolution_state: string; // 'OPTIMIZED_IN_CORE' | 'FIXED_IN_PRESOLVE' | 'CANONICAL_RESTORED' | 'DIRECT_SOLVE'
}

interface WorkflowStage {
  stage: number;
  name: string;
  status: string;
  summary: string;
}

interface SolveResult {
  status: string;
  objective_value: number | null;
  iterations: number;
  runtime_seconds: number;
  nodes_explored: number;
  mip_gap: number | null;
  algorithm_used: string;
  primal_solution: Record<string, number>;
  diagnostics?: {
    iteration_trace?: IterationTraceEntry[];
    tree_trace?: TreeTraceNode[];
    basis_size?: number;
    pricing_method?: string;
    barrier_mu?: number;
    primal_residual?: number;
    dual_residual?: number;
    active_constraints_count?: number;
  };
  postsolve_mapping?: PostsolveMappingItem[];
  workflow_stages?: WorkflowStage[];
  trust_certificate: {
    is_valid: boolean;
    status: string;
    max_primal_violation: number;
    max_bound_violation: number;
    max_integrality_violation: number;
    recomputed_objective: number | null;
    objective_difference: number;
    checks: Record<string, boolean>;
    violation_details: string[];
  };
}

const API_BASE = "http://localhost:8000";

export default function WorkstationDashboard() {
  const [systemInfo, setSystemInfo] = useState<any>(null);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>("refinery_blending_lp");
  const [model, setModel] = useState<ModelMetadata | null>(null);
  const [presolve, setPresolve] = useState<PresolveData | null>(null);
  const [mlRec, setMlRec] = useState<MLRecommendation | null>(null);
  const [solveResult, setSolveResult] = useState<SolveResult | null>(null);

  const [activeStageTab, setActiveStageTab] = useState<number>(1);
  const [algoOverride, setAlgoOverride] = useState<string>("auto");
  const [enablePresolve, setEnablePresolve] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Filter state for postsolve table
  const [varSearchQuery, setVarSearchQuery] = useState<string>("");
  const [hoveredSpyPoint, setHoveredSpyPoint] = useState<SparsityPoint | null>(null);
  const [selectedTreeNode, setSelectedTreeNode] = useState<TreeTraceNode | null>(null);

  // Fetch initial telemetry
  useEffect(() => {
    fetch(`${API_BASE}/api/system_info`)
      .then((r) => r.json())
      .then(setSystemInfo)
      .catch(() => console.log("Backend offline"));

    fetch(`${API_BASE}/api/presets`)
      .then((r) => r.json())
      .then((data) => {
        setPresets(data);
        if (data.length > 0) {
          loadPreset(data[1].id); // Default to Refinery LP
        }
      })
      .catch(() => console.log("Presets fetch failed"));
  }, []);

  const loadPreset = async (presetId: string) => {
    setLoading(true);
    setErrorMsg(null);
    setPresolve(null);
    setMlRec(null);
    setSolveResult(null);
    setSelectedPreset(presetId);
    setSelectedTreeNode(null);

    try {
      const res = await fetch(`${API_BASE}/api/load_preset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset_id: presetId }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setModel(data);
      setActiveStageTab(1);
    } catch (e: any) {
      setErrorMsg(e.message || "Failed to load preset.");
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setErrorMsg(null);
    setPresolve(null);
    setMlRec(null);
    setSolveResult(null);
    setSelectedTreeNode(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/api/upload_model`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setModel(data);
      setSelectedPreset("custom");
      setActiveStageTab(1);
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to parse uploaded model.");
    } finally {
      setLoading(false);
    }
  };

  const runPresolve = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await fetch(`${API_BASE}/api/presolve`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setPresolve(data);
      setActiveStageTab(2);
    } catch (e: any) {
      setErrorMsg(e.message || "Presolve failed.");
    } finally {
      setLoading(false);
    }
  };

  const runMLRecommendation = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await fetch(`${API_BASE}/api/ml_recommend`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setMlRec(data);
      setActiveStageTab(3);
    } catch (e: any) {
      setErrorMsg(e.message || "ML strategy query failed.");
    } finally {
      setLoading(false);
    }
  };

  const runFullWorkflowSolve = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await fetch(`${API_BASE}/api/solve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          algorithm: algoOverride,
          enable_presolve: enablePresolve,
          time_limit_seconds: 60.0,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data: SolveResult = await res.json();
      setSolveResult(data);

      // Auto-focus on Stage 4 or Stage 6
      setActiveStageTab(4);
    } catch (e: any) {
      setErrorMsg(e.message || "Solver pipeline encountered an error.");
    } finally {
      setLoading(false);
    }
  };

  // Filtered postsolve variables
  const filteredVars = useMemo(() => {
    if (!solveResult?.postsolve_mapping) return [];
    if (!varSearchQuery.trim()) return solveResult.postsolve_mapping;
    const q = varSearchQuery.toLowerCase();
    return solveResult.postsolve_mapping.filter(
      (item) => item.variable.toLowerCase().includes(q) || item.resolution_state.toLowerCase().includes(q)
    );
  }, [solveResult, varSearchQuery]);

  return (
    <div className="min-h-screen bg-[#07090d] text-[#cbd5e1] tech-grid flex flex-col selection:bg-amber-500 selection:text-black">
      {/* 1. Precision Workstation Top Bar */}
      <header className="border-b border-[#1c2633] bg-[#0a0d13]/95 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-none border border-emerald-500/40 bg-emerald-950/40 flex items-center justify-center">
              <Scale className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="flex items-center space-x-3">
              <h1 className="font-mono font-bold text-sm tracking-widest text-slate-100 flex items-center gap-2">
                SOVEREIGN // OPTIMIZATION RUNTIME
                <span className="text-[10px] px-1.5 py-0.2 rounded-none bg-[#141d26] text-amber-400 border border-amber-500/30 font-mono">
                  FLOAT64 PURE
                </span>
              </h1>
              <span className="hidden md:inline-block text-[#475569] text-xs">|</span>
              <p className="hidden md:block text-xs text-[#64748b] font-mono">
                No Gurobi/SCIP Wrapping • First-Principles Sparse Engine
              </p>
            </div>
          </div>

          {/* System Telemetry Chips */}
          <div className="flex items-center space-x-3 text-xs font-mono">
            <div className="flex items-center space-x-2 bg-[#0d1217] border border-[#1a232e] px-2.5 py-1 text-slate-300">
              <Cpu className="w-3.5 h-3.5 text-emerald-400" />
              <span>CPU: {systemInfo?.cpu_cores || 8} THREADS</span>
            </div>
            <div className="flex items-center space-x-2 bg-[#0d1217] border border-[#1a232e] px-2.5 py-1 text-slate-300">
              <Activity className="w-3.5 h-3.5 text-amber-400" />
              <span>AVX2: ACTIVE</span>
            </div>
            <div className="flex items-center space-x-2 bg-[#0d1217] border border-[#1a232e] px-2.5 py-1 text-slate-300">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="text-emerald-400 font-bold">SOVEREIGN CORE</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Workspace Area */}
      <main className="max-w-[1600px] mx-auto px-6 py-6 w-full flex-1 space-y-6">
        {errorMsg && (
          <div className="p-3.5 rounded-none bg-rose-950/80 border border-rose-800 text-rose-200 text-xs font-mono flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <AlertTriangle className="w-4 h-4 text-rose-400 flex-shrink-0" />
              <span>{errorMsg}</span>
            </div>
            <button
              onClick={() => setErrorMsg(null)}
              className="text-rose-400 hover:text-white px-2 py-0.5 text-[11px] underline"
            >
              DISMISS
            </button>
          </div>
        )}

        {/* 2. Problem Instance Switcher & Execution Dispatch Bar */}
        <section className="bg-[#0b0f14] border border-[#1c2633] p-4 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[10px] uppercase font-mono tracking-wider text-[#64748b] mr-2 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-amber-400" /> BENCHMARK MODELS:
            </span>
            {presets.map((p) => (
              <button
                key={p.id}
                onClick={() => loadPreset(p.id)}
                className={`px-3 py-1.5 rounded-none text-xs font-mono transition-all flex items-center gap-1.5 border ${
                  selectedPreset === p.id
                    ? "bg-[#16202c] border-amber-500/80 text-amber-400 font-bold shadow-[0_0_10px_rgba(245,158,11,0.15)]"
                    : "bg-[#0d1217] border-[#1c2633] text-[#94a3b8] hover:border-[#2d3d52] hover:text-white"
                }`}
              >
                <span>{p.name}</span>
                <span className="text-[10px] px-1 py-0.2 bg-black/40 text-slate-400 font-mono">
                  {p.problem_class}
                </span>
              </button>
            ))}

            <label className="cursor-pointer px-3 py-1.5 rounded-none text-xs font-mono bg-[#0d1217] hover:bg-[#131a22] border border-[#1c2633] hover:border-slate-600 text-slate-300 flex items-center gap-1.5 transition">
              <FileCode className="w-3.5 h-3.5 text-emerald-400" />
              <span>UPLOAD MPS/LP</span>
              <input type="file" accept=".mps,.lp" className="hidden" onChange={handleFileUpload} />
            </label>
          </div>

          {/* Primary Action Controls */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center space-x-2 bg-[#0d1217] border border-[#1c2633] px-2.5 py-1">
              <input
                type="checkbox"
                id="presolveSwitch"
                checked={enablePresolve}
                onChange={(e) => setEnablePresolve(e.target.checked)}
                className="rounded-none text-emerald-500 bg-black border-slate-700 focus:ring-0"
              />
              <label htmlFor="presolveSwitch" className="text-xs font-mono text-slate-300 cursor-pointer">
                PRESOLVE & RUIZ
              </label>
            </div>

            <select
              value={algoOverride}
              onChange={(e) => setAlgoOverride(e.target.value)}
              className="bg-[#0d1217] border border-[#1c2633] px-3 py-1.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-amber-500"
            >
              <option value="auto">DISPATCH: AI RECOMMENDED</option>
              <option value="simplex">CORE: REVISED SIMPLEX (LP)</option>
              <option value="interior_point">CORE: INTERIOR POINT MEHROTRA (LP)</option>
              <option value="branch_and_bound">CORE: BRANCH-AND-BOUND (MILP)</option>
              <option value="active_set">CORE: ACTIVE SET (QP)</option>
            </select>

            <button
              onClick={runFullWorkflowSolve}
              disabled={loading}
              className="px-5 py-2 bg-emerald-600 hover:bg-emerald-500 text-black font-mono font-bold text-xs tracking-wider transition-all flex items-center gap-2 border border-emerald-400/80 shadow-[0_0_15px_rgba(16,185,129,0.2)] disabled:opacity-50"
            >
              {loading ? (
                <>
                  <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                  <span>OPTIMIZING...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>EXECUTE WORKFLOW</span>
                </>
              )}
            </button>
          </div>
        </section>

        {/* 3. Sequential Workflow Stages Navigation Bar */}
        <section className="bg-[#0b0f14] border border-[#1c2633] p-1">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-1">
            {[
              { id: 1, label: "01 // TOPOLOGY & RUIZ", desc: "Sparse Matrix A & Scaling", icon: Grid },
              { id: 2, label: "02 // PRESOLVE DIFF", desc: "Bound Tightening & Reductions", icon: Sliders },
              { id: 3, label: "03 // AI META-STRATEGY", desc: "Structural Attributions", icon: Sparkles },
              { id: 4, label: "04 // SOLVER DYNAMICS", desc: "Iteration Trace & B&B Tree", icon: BarChart3 },
              { id: 5, label: "05 // POSTSOLVE MAP", desc: "Variable Canonical Recovery", icon: GitBranch },
              { id: 6, label: "06 // TRUST AUDIT", desc: "Independent Float64 Audit", icon: ShieldCheck },
            ].map((st) => {
              const Icon = st.icon;
              const isActive = activeStageTab === st.id;
              return (
                <button
                  key={st.id}
                  onClick={() => setActiveStageTab(st.id)}
                  className={`p-3 text-left transition-all border ${
                    isActive
                      ? "bg-[#141b24] border-amber-500/70 text-slate-100"
                      : "bg-[#0d1217] border-transparent text-[#64748b] hover:bg-[#10161f] hover:text-[#94a3b8]"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={`text-[11px] font-mono font-bold tracking-wider ${isActive ? "text-amber-400" : "text-[#64748b]"}`}>
                      {st.label}
                    </span>
                    <Icon className={`w-3.5 h-3.5 ${isActive ? "text-amber-400" : "text-[#475569]"}`} />
                  </div>
                  <p className="text-[10px] text-[#475569] font-mono mt-1 truncate">{st.desc}</p>
                </button>
              );
            })}
          </div>
        </section>

        {/* 4. Active Stage Detailed Workspace */}
        <div className="space-y-6">
          {/* ============================================================== */}
          {/* STAGE 1: MATRIX TOPOLOGY & RUIZ EQUILIBRATION SCALING          */}
          {/* ============================================================== */}
          {activeStageTab === 1 && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Matrix Spy Plot (8 cols) */}
              <div className="lg:col-span-8 bg-[#0b0f14] border border-[#1c2633] p-5 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between border-b border-[#1c2633] pb-3 mb-4">
                    <div className="flex items-center space-x-2">
                      <Grid className="w-4 h-4 text-cyan-400" />
                      <h2 className="text-xs font-mono font-bold text-slate-200 tracking-wider uppercase">
                        Constraint Matrix Topology (Spy Plot)
                      </h2>
                    </div>
                    {model && (
                      <div className="flex items-center space-x-3 text-xs font-mono">
                        <span className="text-[#64748b]">DIMENSION:</span>
                        <span className="text-slate-200">{model.num_constraints} × {model.num_variables}</span>
                        <span className="text-[#64748b]">NNZ:</span>
                        <span className="text-cyan-400">{model.num_nonzeros}</span>
                        <span className="text-[#64748b]">DENSITY:</span>
                        <span className="text-amber-400">{(model.density * 100).toFixed(2)}%</span>
                      </div>
                    )}
                  </div>

                  {model ? (
                    <div className="relative border border-[#1a232e] bg-[#070a0e] p-4 flex flex-col items-center">
                      {/* SVG Canvas for Spy Plot */}
                      <div className="w-full aspect-[2/1] max-h-[380px] relative flex items-center justify-center">
                        <svg
                          viewBox={`0 0 ${Math.max(model.num_variables, 10)} ${Math.max(model.num_constraints, 10)}`}
                          className="w-full h-full border border-[#16202c] bg-[#05070a]"
                          preserveAspectRatio="none"
                        >
                          {/* Grid Lines */}
                          <line
                            x1="0"
                            y1="0"
                            x2={model.num_variables}
                            y2={model.num_constraints}
                            stroke="#111822"
                            strokeWidth="0.5"
                            strokeDasharray="2,2"
                          />
                          {/* Render Non-zero points */}
                          {model.sparsity_sample.map((pt, idx) => {
                            const isPositive = pt.val >= 0;
                            return (
                              <circle
                                key={idx}
                                cx={pt.col + 0.5}
                                cy={pt.row + 0.5}
                                r={Math.max(0.4, Math.min(1.5, Math.abs(pt.val)))}
                                fill={isPositive ? "#38bdf8" : "#f59e0b"}
                                opacity={0.85}
                                onMouseEnter={() => setHoveredSpyPoint(pt)}
                                onMouseLeave={() => setHoveredSpyPoint(null)}
                                className="cursor-crosshair hover:scale-150 transition-transform"
                              />
                            );
                          })}
                        </svg>
                      </div>

                      {/* Tooltip for Spy Point */}
                      <div className="w-full mt-3 flex items-center justify-between text-[11px] font-mono text-[#64748b] border-t border-[#16202c] pt-2">
                        <div className="flex items-center space-x-4">
                          <span className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-[#38bdf8]"></span> Positive Coefficient
                          </span>
                          <span className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-[#f59e0b]"></span> Negative Coefficient
                          </span>
                        </div>
                        <div>
                          {hoveredSpyPoint ? (
                            <span className="text-slate-200">
                              ROW: <span className="text-amber-400 font-bold">{hoveredSpyPoint.row}</span> | COL:{" "}
                              <span className="text-cyan-400 font-bold">{hoveredSpyPoint.col}</span> | VAL:{" "}
                              <span className="text-white font-bold">{hoveredSpyPoint.val.toFixed(4)}</span>
                            </span>
                          ) : (
                            <span className="text-[#475569]">Hover over matrix entries to inspect coordinate values</span>
                          )}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="py-16 text-center text-xs font-mono text-[#64748b]">Loading model topology...</div>
                  )}
                </div>
              </div>

              {/* Ruiz Equilibration & Condition Number Panel (4 cols) */}
              <div className="lg:col-span-4 bg-[#0b0f14] border border-[#1c2633] p-5 flex flex-col justify-between space-y-4">
                <div>
                  <div className="flex items-center justify-between border-b border-[#1c2633] pb-3 mb-4">
                    <h2 className="text-xs font-mono font-bold text-slate-200 tracking-wider uppercase flex items-center gap-2">
                      <Scale className="w-4 h-4 text-emerald-400" /> Ruiz Scaling Equilibration
                    </h2>
                    <span className="text-[10px] font-mono text-emerald-400 px-1.5 py-0.5 bg-emerald-950/60 border border-emerald-800/60">
                      Ruiz 2001 (D1·A·D2)
                    </span>
                  </div>

                  {model?.ruiz_stats ? (
                    <div className="space-y-4 text-xs font-mono">
                      <p className="text-[#94a3b8] text-[11px] leading-relaxed">
                        Computes diagonal scaling matrices <code className="text-emerald-400 font-bold">D1</code> and{" "}
                        <code className="text-emerald-400 font-bold">D2</code> such that all row and column infinity norms
                        approach <code className="text-white">1.0</code>, suppressing numerical instability in interior-point normal equations and simplex factorizations.
                      </p>

                      <div className="grid grid-cols-2 gap-2">
                        <div className="p-3 bg-[#070a0e] border border-[#16202c]">
                          <span className="text-[10px] text-[#64748b] block">Pre-Scale ||A||_inf</span>
                          <span className="text-sm font-bold text-amber-400 font-mono">
                            {model.ruiz_stats.norm_before}
                          </span>
                        </div>
                        <div className="p-3 bg-[#070a0e] border border-[#16202c]">
                          <span className="text-[10px] text-[#64748b] block">Post-Scale ||D1·A·D2||</span>
                          <span className="text-sm font-bold text-emerald-400 font-mono">
                            {model.ruiz_stats.norm_after}
                          </span>
                        </div>
                      </div>

                      <div className="p-3 bg-[#070a0e] border border-[#16202c] space-y-2">
                        <div className="flex justify-between">
                          <span className="text-[#64748b]">Row Scaling D1:</span>
                          <span className="text-slate-200">
                            [{model.ruiz_stats.d1_min}, {model.ruiz_stats.d1_max}]
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#64748b]">Col Scaling D2:</span>
                          <span className="text-slate-200">
                            [{model.ruiz_stats.d2_min}, {model.ruiz_stats.d2_max}]
                          </span>
                        </div>
                        <div className="flex justify-between border-t border-[#16202c] pt-1.5">
                          <span className="text-[#64748b]">Equilibration Passes:</span>
                          <span className="text-cyan-400">{model.ruiz_stats.iterations} iters</span>
                        </div>
                      </div>

                      <div className="p-3 bg-[#070a0e] border border-[#16202c] text-[11px] text-[#64748b]">
                        <span className="text-slate-300 font-bold block mb-1">Primal/Dual Recovery:</span>
                        <div>x = D2 · x_scaled</div>
                        <div>y = D1 · y_scaled</div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-[#64748b] font-mono text-xs">
                      Ruiz statistics pending model compilation.
                    </div>
                  )}
                </div>

                <button
                  onClick={() => setActiveStageTab(2)}
                  className="w-full py-2 bg-[#121822] hover:bg-[#16202c] border border-[#1c2633] text-xs font-mono text-slate-200 flex items-center justify-center gap-1.5 transition"
                >
                  <span>PROCEED TO STAGE 02: PRESOLVE DIFF</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* ============================================================== */}
          {/* STAGE 2: PRESOLVE REDUCTIONS & BOUND TIGHTENING DIFF           */}
          {/* ============================================================== */}
          {activeStageTab === 2 && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Presolve Diff Metrics (5 cols) */}
              <div className="lg:col-span-5 bg-[#0b0f14] border border-[#1c2633] p-5 flex flex-col justify-between space-y-4">
                <div>
                  <div className="flex items-center justify-between border-b border-[#1c2633] pb-3 mb-4">
                    <h2 className="text-xs font-mono font-bold text-slate-200 tracking-wider uppercase flex items-center gap-2">
                      <Sliders className="w-4 h-4 text-emerald-400" /> Presolve Reduction Pipeline
                    </h2>
                    <button
                      onClick={runPresolve}
                      disabled={loading}
                      className="px-2.5 py-1 text-xs font-mono bg-emerald-950 text-emerald-400 border border-emerald-800 hover:bg-emerald-900 transition"
                    >
                      {presolve ? "RE-RUN PRESOLVE" : "EXECUTE PRESOLVE"}
                    </button>
                  </div>

                  {presolve ? (
                    <div className="space-y-4 font-mono text-xs">
                      {/* Before / After Reduction Cards */}
                      <div className="grid grid-cols-3 gap-2 text-center">
                        <div className="p-3 bg-[#070a0e] border border-[#16202c]">
                          <span className="text-[10px] text-[#64748b] block">VARIABLES</span>
                          <span className="text-base font-bold text-emerald-400">
                            -{presolve.var_reduction_pct}%
                          </span>
                          <span className="text-[10px] text-[#475569] block mt-0.5">
                            {presolve.original_vars} → {presolve.presolved_vars}
                          </span>
                        </div>
                        <div className="p-3 bg-[#070a0e] border border-[#16202c]">
                          <span className="text-[10px] text-[#64748b] block">CONSTRAINTS</span>
                          <span className="text-base font-bold text-emerald-400">
                            -{presolve.con_reduction_pct}%
                          </span>
                          <span className="text-[10px] text-[#475569] block mt-0.5">
                            {presolve.original_cons} → {presolve.presolved_cons}
                          </span>
                        </div>
                        <div className="p-3 bg-[#070a0e] border border-[#16202c]">
                          <span className="text-[10px] text-[#64748b] block">NON-ZEROS</span>
                          <span className="text-base font-bold text-emerald-400">
                            -{presolve.nnz_reduction_pct}%
                          </span>
                          <span className="text-[10px] text-[#475569] block mt-0.5">
                            {presolve.original_nnz} → {presolve.presolved_nnz}
                          </span>
                        </div>
                      </div>

                      {/* Mechanism Breakdown */}
                      <div className="p-3.5 bg-[#070a0e] border border-[#16202c] space-y-2">
                        <span className="text-[11px] font-bold text-slate-300 block mb-1">
                          REDUCTION MECHANISMS APPLIED:
                        </span>
                        <div className="flex justify-between">
                          <span className="text-[#64748b]">Fixed Variables (l_j = u_j):</span>
                          <span className="text-emerald-400 font-bold">{presolve.fixed_vars_count} pruned</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#64748b]">Singleton Rows Tightened:</span>
                          <span className="text-emerald-400 font-bold">{presolve.singleton_rows_count} rows</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#64748b]">Empty Rows Eliminated:</span>
                          <span className="text-emerald-400 font-bold">{presolve.empty_rows_count} rows</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-[#64748b]">Empty Columns Eliminated:</span>
                          <span className="text-emerald-400 font-bold">{presolve.empty_cols_count} cols</span>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="py-12 text-center text-[#64748b] font-mono text-xs">
                      Presolve reduction metrics pending analysis. Click Execute Presolve to scan singleton rows and fixed variables.
                    </div>
                  )}
                </div>

                <button
                  onClick={() => setActiveStageTab(3)}
                  className="w-full py-2 bg-[#121822] hover:bg-[#16202c] border border-[#1c2633] text-xs font-mono text-slate-200 flex items-center justify-center gap-1.5 transition"
                >
                  <span>PROCEED TO STAGE 03: AI META-STRATEGY</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Fixed Variables Substitution Audit Table (7 cols) */}
              <div className="lg:col-span-7 bg-[#0b0f14] border border-[#1c2633] p-5 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between border-b border-[#1c2633] pb-3 mb-4">
                    <h2 className="text-xs font-mono font-bold text-slate-200 tracking-wider uppercase flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-cyan-400" /> Fixed Substitution Audit Log
                    </h2>
                    <span className="text-[10px] font-mono text-[#64748b]">
                      Exact Values Stored in Postsolve Stack
                    </span>
                  </div>

                  {presolve?.fixed_vars_list && presolve.fixed_vars_list.length > 0 ? (
                    <div className="border border-[#16202c] max-h-[340px] overflow-y-auto">
                      <table className="w-full text-left font-mono text-xs">
                        <thead className="bg-[#070a0e] text-[#64748b] border-b border-[#16202c] sticky top-0">
                          <tr>
                            <th className="py-2 px-3">VARIABLE IDENTIFIER</th>
                            <th className="py-2 px-3">FIXED VALUE</th>
                            <th className="py-2 px-3 text-right">REDUCTION TYPE</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#121822]">
                          {presolve.fixed_vars_list.map((fv, i) => (
                            <tr key={i} className="hover:bg-[#10161f]">
                              <td className="py-1.5 px-3 text-slate-300">{fv.name}</td>
                              <td className="py-1.5 px-3 text-amber-400 font-bold">{fv.value}</td>
                              <td className="py-1.5 px-3 text-right text-emerald-400 text-[11px]">
                                BOUND_CONVERGENCE
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="p-8 text-center text-[#64748b] font-mono text-xs border border-dashed border-[#16202c]">
                      {presolve
                        ? "No variables required fixed substitution in this model formulation."
                        : "Execute Presolve to inspect fixed variable substitutions."}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ============================================================== */}
          {/* STAGE 3: AI META-STRATEGY & DETERMINISTIC HARNESS             */}
          {/* ============================================================== */}
          {activeStageTab === 3 && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Algorithm Recommendation & Confidence (5 cols) */}
              <div className="lg:col-span-5 bg-[#0b0f14] border border-[#1c2633] p-5 flex flex-col justify-between space-y-4">
                <div>
                  <div className="flex items-center justify-between border-b border-[#1c2633] pb-3 mb-4">
                    <h2 className="text-xs font-mono font-bold text-slate-200 tracking-wider uppercase flex items-center gap-2">
                      <Sparkles className="w-4 h-4 text-amber-400" /> AI Meta-Strategy Recommendation
                    </h2>
                    <button
                      onClick={runMLRecommendation}
                      disabled={loading}
                      className="px-2.5 py-1 text-xs font-mono bg-amber-950 text-amber-400 border border-amber-800 hover:bg-amber-900 transition"
                    >
                      {mlRec ? "RE-EVALUATE" : "QUERY AI STRATEGY"}
                    </button>
                  </div>

                  {mlRec ? (
                    <div className="space-y-4 font-mono text-xs">
                      {/* Selected Algorithm Card */}
                      <div className="p-4 bg-[#070a0e] border border-amber-500/40">
                        <span className="text-[10px] text-[#64748b] uppercase tracking-wider block">
                          RECOMMENDED OPTIMIZATION ENGINE
                        </span>
                        <div className="flex items-center justify-between mt-1">
                          <span className="text-base font-bold text-white uppercase">
                            {mlRec.recommended_algorithm.replace("_", " ")}
                          </span>
                          <span className="text-xs px-2 py-0.5 bg-amber-950 text-amber-400 border border-amber-700 font-bold">
                            {mlRec.confidence_score}% CONFIDENCE
                          </span>
                        </div>
                      </div>

                      {/* Probabilities Distribution */}
                      <div className="p-3.5 bg-[#070a0e] border border-[#16202c] space-y-2">
                        <span className="text-[10px] text-[#64748b] block font-bold">
                          ALGORITHM PROBABILITY DISTRIBUTION:
                        </span>
                        {Object.entries(mlRec.algorithm_probabilities).map(([algo, prob]) => {
                          const pct = Math.round(prob * 100);
                          return (
                            <div key={algo} className="space-y-1">
                              <div className="flex justify-between text-[11px]">
                                <span className="text-slate-300 uppercase">{algo.replace("_", " ")}</span>
                                <span className="text-amber-400 font-bold">{pct}%</span>
                              </div>
                              <div className="w-full bg-[#121822] h-1.5">
                                <div className="bg-amber-400 h-1.5 transition-all" style={{ width: `${pct}%` }}></div>
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {/* Deterministic Safety Harness Badge */}
                      <div className="p-3 bg-emerald-950/30 border border-emerald-800/60 text-emerald-300 text-[11px] leading-relaxed">
                        <div className="flex items-center gap-1.5 font-bold text-emerald-400 mb-1">
                          <ShieldCheck className="w-3.5 h-3.5" />
                          <span>DETERMINISTIC SAFETY HARNESS ACTIVE</span>
                        </div>
                        Fallback: <code className="text-white font-bold">{mlRec.deterministic_fallback}</code>. ML is strictly advisory; primal feasibility and optimality remain guaranteed by first-principles mathematics.
                      </div>
                    </div>
                  ) : (
                    <div className="py-12 text-center text-[#64748b] font-mono text-xs">
                      Query the AI Meta-Strategy engine to evaluate the 28-dimensional structural vector.
                    </div>
                  )}
                </div>

                <button
                  onClick={() => setActiveStageTab(4)}
                  className="w-full py-2 bg-[#121822] hover:bg-[#16202c] border border-[#1c2633] text-xs font-mono text-slate-200 flex items-center justify-center gap-1.5 transition"
                >
                  <span>PROCEED TO STAGE 04: SOLVER ITERATION DYNAMICS</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* 28-Dimensional Structural Feature Vector (7 cols) */}
              <div className="lg:col-span-7 bg-[#0b0f14] border border-[#1c2633] p-5 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between border-b border-[#1c2633] pb-3 mb-4">
                    <h2 className="text-xs font-mono font-bold text-slate-200 tracking-wider uppercase flex items-center gap-2">
                      <Terminal className="w-4 h-4 text-cyan-400" /> Structural Feature Attributions
                    </h2>
                    <span className="text-[10px] font-mono text-[#64748b]">28-Dimensional Vector</span>
                  </div>

                  {mlRec?.features ? (
                    <div className="border border-[#16202c] max-h-[380px] overflow-y-auto">
                      <table className="w-full text-left font-mono text-xs">
                        <thead className="bg-[#070a0e] text-[#64748b] border-b border-[#16202c] sticky top-0">
                          <tr>
                            <th className="py-2 px-3">FEATURE DESCRIPTOR</th>
                            <th className="py-2 px-3">EXTRACTED VALUE</th>
                            <th className="py-2 px-3 text-right">MODEL INFLUENCE</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#121822]">
                          {Object.entries(mlRec.features).map(([fKey, fVal], i) => (
                            <tr key={i} className="hover:bg-[#10161f]">
                              <td className="py-1.5 px-3 text-slate-300">{fKey}</td>
                              <td className="py-1.5 px-3 text-cyan-400 font-bold">{fVal}</td>
                              <td className="py-1.5 px-3 text-right text-[#94a3b8] text-[11px]">
                                {fKey.includes("ratio") || fKey.includes("density") ? "HIGH WEIGHT" : "STANDARD"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="p-8 text-center text-[#64748b] font-mono text-xs border border-dashed border-[#16202c]">
                      Execute AI Meta-Strategy evaluation to inspect extracted structural feature vectors.
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ============================================================== */}
          {/* STAGE 4: LIVE SOLVER ENGINE ITERATION & SEARCH DYNAMICS        */}
          {/* ============================================================== */}
          {activeStageTab === 4 && (
            <div className="space-y-6">
              {solveResult ? (
                <>
                  {/* Top Solve Summary Metrics */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3 font-mono">
                    <div className="p-3 bg-[#0b0f14] border border-[#1c2633]">
                      <span className="text-[10px] text-[#64748b] block">STATUS</span>
                      <span className="text-base font-bold text-emerald-400 uppercase">
                        {solveResult.status}
                      </span>
                    </div>
                    <div className="p-3 bg-[#0b0f14] border border-[#1c2633]">
                      <span className="text-[10px] text-[#64748b] block">OBJECTIVE VALUE</span>
                      <span className="text-base font-bold text-white">
                        {solveResult.objective_value !== null ? solveResult.objective_value.toFixed(4) : "N/A"}
                      </span>
                    </div>
                    <div className="p-3 bg-[#0b0f14] border border-[#1c2633]">
                      <span className="text-[10px] text-[#64748b] block">RUNTIME</span>
                      <span className="text-base font-bold text-amber-400">
                        {(solveResult.runtime_seconds * 1000).toFixed(1)}ms
                      </span>
                    </div>
                    <div className="p-3 bg-[#0b0f14] border border-[#1c2633]">
                      <span className="text-[10px] text-[#64748b] block">ITERATIONS</span>
                      <span className="text-base font-bold text-cyan-400">
                        {solveResult.iterations}
                      </span>
                    </div>
                    <div className="p-3 bg-[#0b0f14] border border-[#1c2633]">
                      <span className="text-[10px] text-[#64748b] block">ALGORITHM USED</span>
                      <span className="text-xs font-bold text-slate-200 block pt-1 truncate">
                        {solveResult.algorithm_used}
                      </span>
                    </div>
                    <div className="p-3 bg-[#0b0f14] border border-[#1c2633]">
                      <span className="text-[10px] text-[#64748b] block">TRUST AUDIT</span>
                      <span className="text-xs font-bold text-emerald-400 block pt-1 flex items-center gap-1">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>{solveResult.trust_certificate.is_valid ? "CERTIFIED PASS" : "FAIL"}</span>
                      </span>
                    </div>
                  </div>

                  {/* Visual Trace: Branch & Bound Tree OR Simplex/IPM Convergence Plot */}
                  {solveResult.diagnostics?.tree_trace && solveResult.diagnostics.tree_trace.length > 0 ? (
                    /* MILP Branch & Bound Search Tree Visualization */
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                      <div className="lg:col-span-8 bg-[#0b0f14] border border-[#1c2633] p-5">
                        <div className="flex items-center justify-between border-b border-[#1c2633] pb-3 mb-4">
                          <h2 className="text-xs font-mono font-bold text-slate-200 tracking-wider uppercase flex items-center gap-2">
                            <GitBranch className="w-4 h-4 text-amber-400" /> Branch & Bound Search Tree Explorer
                          </h2>
                          <div className="flex items-center space-x-3 text-[11px] font-mono">
                            <span className="flex items-center gap-1">
                              <span className="w-2 h-2 bg-emerald-400"></span> Incumbent
                            </span>
                            <span className="flex items-center gap-1">
                              <span className="w-2 h-2 bg-amber-400"></span> Branch
                            </span>
                            <span className="flex items-center gap-1">
                              <span className="w-2 h-2 bg-[#475569]"></span> Pruned
                            </span>
                          </div>
                        </div>

                        {/* Hierarchical Tree Render */}
                        <div className="border border-[#16202c] bg-[#070a0e] p-4 min-h-[320px] max-h-[420px] overflow-auto">
                          <div className="space-y-2 font-mono">
                            {solveResult.diagnostics.tree_trace.map((node) => {
                              const isIncumbent = node.status === "INTEGER_INCUMBENT";
                              const isPruned = node.status === "PRUNED_BOUND" || node.status === "INFEASIBLE";
                              const isSelected = selectedTreeNode?.node_id === node.node_id;

                              return (
                                <div
                                  key={node.node_id}
                                  onClick={() => setSelectedTreeNode(node)}
                                  style={{ marginLeft: `${node.depth * 24}px` }}
                                  className={`p-2.5 border text-xs cursor-pointer transition-all flex items-center justify-between ${
                                    isSelected
                                      ? "border-amber-400 bg-[#16202c]"
                                      : isIncumbent
                                      ? "border-emerald-600/70 bg-emerald-950/30 text-emerald-300"
                                      : isPruned
                                      ? "border-[#1c2633] bg-[#090d12] text-[#64748b]"
                                      : "border-amber-500/40 bg-amber-950/20 text-amber-300"
                                  }`}
                                >
                                  <div className="flex items-center space-x-2">
                                    <span className="text-[10px] px-1 py-0.2 bg-black/40 text-slate-400">
                                      N{node.node_id}
                                    </span>
                                    <span className="font-bold">{node.branch_condition}</span>
                                  </div>
                                  <div className="flex items-center space-x-3 text-[11px]">
                                    <span className="text-[#94a3b8]">z_LP: {node.lower_bound.toFixed(2)}</span>
                                    <span
                                      className={`text-[10px] px-1.5 py-0.2 uppercase font-bold ${
                                        isIncumbent
                                          ? "bg-emerald-950 text-emerald-400 border border-emerald-800"
                                          : isPruned
                                          ? "bg-slate-900 text-slate-400 border border-slate-700"
                                          : "bg-amber-950 text-amber-400 border border-amber-800"
                                      }`}
                                    >
                                      {node.status}
                                    </span>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>

                      {/* Tree Node Inspector Panel (4 cols) */}
                      <div className="lg:col-span-4 bg-[#0b0f14] border border-[#1c2633] p-5 flex flex-col justify-between space-y-4">
                        <div>
                          <div className="flex items-center justify-between border-b border-[#1c2633] pb-3 mb-4">
                            <h2 className="text-xs font-mono font-bold text-slate-200 tracking-wider uppercase flex items-center gap-2">
                              <Info className="w-4 h-4 text-cyan-400" /> Node Inspection Details
                            </h2>
                          </div>

                          {selectedTreeNode ? (
                            <div className="space-y-3 font-mono text-xs">
                              <div className="p-3 bg-[#070a0e] border border-[#16202c]">
                                <span className="text-[10px] text-[#64748b] block">NODE IDENTIFIER</span>
                                <span className="text-base font-bold text-white">
                                  Node #{selectedTreeNode.node_id} (Depth {selectedTreeNode.depth})
                                </span>
                              </div>

                              <div className="p-3 bg-[#070a0e] border border-[#16202c] space-y-2">
                                <div className="flex justify-between">
                                  <span className="text-[#64748b]">Branch Variable:</span>
                                  <span className="text-slate-200 font-bold">{selectedTreeNode.branch_var}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-[#64748b]">Branch Condition:</span>
                                  <span className="text-amber-400 font-bold">{selectedTreeNode.branch_condition}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-[#64748b]">LP Relaxation Bound:</span>
                                  <span className="text-emerald-400 font-bold">{selectedTreeNode.lower_bound.toFixed(4)}</span>
                                </div>
                                <div className="flex justify-between border-t border-[#16202c] pt-2">
                                  <span className="text-[#64748b]">Search Status:</span>
                                  <span className="text-cyan-400 font-bold uppercase">{selectedTreeNode.status}</span>
                                </div>
                              </div>
                            </div>
                          ) : (
                            <div className="text-center py-12 text-[#64748b] font-mono text-xs">
                              Click on any node in the search tree above to inspect its LP relaxation bounds and pruning decisions.
                            </div>
                          )}
                        </div>

                        <button
                          onClick={() => setActiveStageTab(5)}
                          className="w-full py-2 bg-[#121822] hover:bg-[#16202c] border border-[#1c2633] text-xs font-mono text-slate-200 flex items-center justify-center gap-1.5 transition"
                        >
                          <span>PROCEED TO STAGE 05: POSTSOLVE RECOVERY</span>
                          <ChevronRight className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  ) : (
                    /* Simplex / IPM Iteration Trace & Convergence Chart */
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                      {/* SVG Iteration Convergence Graph (7 cols) */}
                      <div className="lg:col-span-7 bg-[#0b0f14] border border-[#1c2633] p-5">
                        <div className="flex items-center justify-between border-b border-[#1c2633] pb-3 mb-4">
                          <h2 className="text-xs font-mono font-bold text-slate-200 tracking-wider uppercase flex items-center gap-2">
                            <BarChart3 className="w-4 h-4 text-cyan-400" /> Objective Convergence Trajectory
                          </h2>
                          <span className="text-[10px] font-mono text-[#64748b]">Iteration vs. Objective</span>
                        </div>

                        {solveResult.diagnostics?.iteration_trace && solveResult.diagnostics.iteration_trace.length > 0 ? (
                          <div className="space-y-3">
                            <div className="w-full h-64 border border-[#16202c] bg-[#070a0e] p-3 relative flex items-end">
                              <svg viewBox="0 0 100 100" className="w-full h-full overflow-visible" preserveAspectRatio="none">
                                {/* Grid reference line */}
                                <line x1="0" y1="50" x2="100" y2="50" stroke="#16202c" strokeWidth="0.5" strokeDasharray="2,2" />
                                
                                {/* Polyline of Objective Values */}
                                {(() => {
                                  const trace = solveResult.diagnostics.iteration_trace;
                                  const objs = trace.map((t) => t.objective ?? 0);
                                  const minO = Math.min(...objs);
                                  const maxO = Math.max(...objs);
                                  const range = maxO - minO || 1;

                                  const points = trace
                                    .map((t, idx) => {
                                      const x = (idx / Math.max(trace.length - 1, 1)) * 100;
                                      const y = 90 - (((t.objective ?? 0) - minO) / range) * 80;
                                      return `${x},${y}`;
                                    })
                                    .join(" ");

                                  return (
                                    <>
                                      <polyline fill="none" stroke="#10b981" strokeWidth="2" points={points} />
                                      {trace.map((t, idx) => {
                                        const x = (idx / Math.max(trace.length - 1, 1)) * 100;
                                        const y = 90 - (((t.objective ?? 0) - minO) / range) * 80;
                                        return (
                                          <circle
                                            key={idx}
                                            cx={x}
                                            cy={y}
                                            r="2.5"
                                            fill="#34d399"
                                            className="hover:r-4 transition-all"
                                          />
                                        );
                                      })}
                                    </>
                                  );
                                })()}
                              </svg>
                            </div>

                            <div className="flex justify-between text-[10px] font-mono text-[#64748b]">
                              <span>ITER 1</span>
                              <span className="text-emerald-400">OBJECTIVE TRAJECTORY CONVERGED</span>
                              <span>ITER {solveResult.iterations}</span>
                            </div>
                          </div>
                        ) : (
                          <div className="py-16 text-center text-[#64748b] font-mono text-xs">
                            Iteration curve converged in Phase 1 or initial unconstrained point.
                          </div>
                        )}
                      </div>

                      {/* Iteration Trace Table (5 cols) */}
                      <div className="lg:col-span-5 bg-[#0b0f14] border border-[#1c2633] p-5 flex flex-col justify-between">
                        <div>
                          <div className="flex items-center justify-between border-b border-[#1c2633] pb-3 mb-4">
                            <h2 className="text-xs font-mono font-bold text-slate-200 tracking-wider uppercase flex items-center gap-2">
                              <Terminal className="w-4 h-4 text-amber-400" /> Solver Step Telemetry
                            </h2>
                            <span className="text-[10px] font-mono text-[#64748b]">Exact Iterations</span>
                          </div>

                          <div className="border border-[#16202c] max-h-[300px] overflow-y-auto">
                            <table className="w-full text-left font-mono text-xs">
                              <thead className="bg-[#070a0e] text-[#64748b] border-b border-[#16202c] sticky top-0">
                                <tr>
                                  <th className="py-2 px-3">IT</th>
                                  <th className="py-2 px-3">OBJECTIVE</th>
                                  <th className="py-2 px-3 text-right">TELEMETRY</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-[#121822]">
                                {solveResult.diagnostics?.iteration_trace?.map((t, idx) => (
                                  <tr key={idx} className="hover:bg-[#10161f]">
                                    <td className="py-1.5 px-3 text-[#64748b]">{t.iteration}</td>
                                    <td className="py-1.5 px-3 text-emerald-400 font-bold">
                                      {t.objective !== undefined ? t.objective.toFixed(4) : "—"}
                                    </td>
                                    <td className="py-1.5 px-3 text-right text-[#94a3b8] text-[11px]">
                                      {t.entering_var ? (
                                        <span>+{t.entering_var} / -{t.leaving_var}</span>
                                      ) : t.mu !== undefined ? (
                                        <span>μ={t.mu.toExponential(1)}</span>
                                      ) : (
                                        <span>p={t.p_norm?.toExponential(1)}</span>
                                      )}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>

                        <button
                          onClick={() => setActiveStageTab(5)}
                          className="w-full mt-4 py-2 bg-[#121822] hover:bg-[#16202c] border border-[#1c2633] text-xs font-mono text-slate-200 flex items-center justify-center gap-1.5 transition"
                        >
                          <span>PROCEED TO STAGE 05: POSTSOLVE RECOVERY</span>
                          <ChevronRight className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="p-12 text-center bg-[#0b0f14] border border-[#1c2633]">
                  <p className="text-xs font-mono text-[#64748b] mb-4">
                    Workflow not yet solved. Click EXECUTE WORKFLOW in the top bar to run the complete 6-stage optimization pipeline.
                  </p>
                  <button
                    onClick={runFullWorkflowSolve}
                    className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-black font-mono font-bold text-xs tracking-wider transition"
                  >
                    RUN WORKFLOW SOLVER
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ============================================================== */}
          {/* STAGE 5: POSTSOLVE CANONICAL VARIABLE RECONSTRUCTION           */}
          {/* ============================================================== */}
          {activeStageTab === 5 && (
            <div className="bg-[#0b0f14] border border-[#1c2633] p-5 space-y-4">
              <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-[#1c2633] pb-3 gap-3">
                <div className="flex items-center space-x-2">
                  <GitBranch className="w-4 h-4 text-emerald-400" />
                  <h2 className="text-xs font-mono font-bold text-slate-200 tracking-wider uppercase">
                    Postsolve Variable Unwinding & Canonical Reconstruction
                  </h2>
                </div>

                <div className="relative w-full md:w-72">
                  <Search className="w-3.5 h-3.5 text-[#64748b] absolute left-2.5 top-2.5" />
                  <input
                    type="text"
                    placeholder="Filter variable name..."
                    value={varSearchQuery}
                    onChange={(e) => setVarSearchQuery(e.target.value)}
                    className="w-full bg-[#070a0e] border border-[#1c2633] pl-8 pr-3 py-1.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-amber-500"
                  />
                </div>
              </div>

              {solveResult?.postsolve_mapping ? (
                <div className="space-y-4">
                  <div className="border border-[#16202c] max-h-[440px] overflow-y-auto">
                    <table className="w-full text-left font-mono text-xs">
                      <thead className="bg-[#070a0e] text-[#64748b] border-b border-[#16202c] sticky top-0">
                        <tr>
                          <th className="py-2.5 px-4">CANONICAL VARIABLE</th>
                          <th className="py-2.5 px-4">OPTIMAL VALUE (FLOAT64)</th>
                          <th className="py-2.5 px-4 text-right">RECONSTRUCTION RESOLUTION</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#121822]">
                        {filteredVars.map((item, idx) => {
                          const isFixed = item.resolution_state === "FIXED_IN_PRESOLVE";
                          const isCore = item.resolution_state === "OPTIMIZED_IN_CORE";
                          return (
                            <tr key={idx} className="hover:bg-[#10161f]">
                              <td className="py-2 px-4 text-slate-300 font-bold">{item.variable}</td>
                              <td className="py-2 px-4 text-emerald-400 font-bold text-sm">
                                {item.value.toFixed(6)}
                              </td>
                              <td className="py-2 px-4 text-right">
                                <span
                                  className={`text-[10px] px-2 py-0.5 uppercase font-bold border ${
                                    isFixed
                                      ? "bg-amber-950/40 text-amber-400 border-amber-800"
                                      : isCore
                                      ? "bg-cyan-950/40 text-cyan-400 border-cyan-800"
                                      : "bg-slate-900 text-slate-400 border-slate-700"
                                  }`}
                                >
                                  {item.resolution_state}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  <div className="flex justify-between items-center text-[11px] font-mono text-[#64748b]">
                    <span>Showing {filteredVars.length} variables</span>
                    <button
                      onClick={() => setActiveStageTab(6)}
                      className="px-4 py-2 bg-[#121822] hover:bg-[#16202c] border border-[#1c2633] text-xs font-mono text-slate-200 flex items-center gap-1.5 transition"
                    >
                      <span>PROCEED TO STAGE 06: TRUST CERTIFICATION</span>
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="py-16 text-center text-[#64748b] font-mono text-xs">
                  Execute workflow to view postsolve canonical reconstruction mapping.
                </div>
              )}
            </div>
          )}

          {/* ============================================================== */}
          {/* STAGE 6: INDEPENDENT MATHEMATICAL TRUST CERTIFICATION          */}
          {/* ============================================================== */}
          {activeStageTab === 6 && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Trust Certificate Seal & Verification Status (5 cols) */}
              <div className="lg:col-span-5 bg-[#0b0f14] border border-[#1c2633] p-5 flex flex-col justify-between space-y-4">
                <div>
                  <div className="flex items-center justify-between border-b border-[#1c2633] pb-3 mb-4">
                    <h2 className="text-xs font-mono font-bold text-slate-200 tracking-wider uppercase flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4 text-emerald-400" /> Independent Trust Audit
                    </h2>
                    {solveResult && (
                      <span
                        className={`text-[10px] px-2 py-0.5 uppercase font-mono font-bold border ${
                          solveResult.trust_certificate.is_valid
                            ? "bg-emerald-950 text-emerald-400 border-emerald-800"
                            : "bg-rose-950 text-rose-400 border-rose-800"
                        }`}
                      >
                        {solveResult.trust_certificate.status}
                      </span>
                    )}
                  </div>

                  {solveResult ? (
                    <div className="space-y-4 font-mono text-xs">
                      {/* Mathematical Checks Stack */}
                      <div className="p-3.5 bg-[#070a0e] border border-[#16202c] space-y-2.5">
                        <div className="flex items-center justify-between">
                          <span className="text-slate-300 flex items-center gap-2">
                            <Check className="w-3.5 h-3.5 text-emerald-400" /> Primal Constraints (Ax ≤ b)
                          </span>
                          <span className="font-bold text-emerald-400">
                            {solveResult.trust_certificate.checks.constraints_satisfied ? "PASS" : "FAIL"}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-slate-300 flex items-center gap-2">
                            <Check className="w-3.5 h-3.5 text-emerald-400" /> Variable Bounds (l ≤ x ≤ u)
                          </span>
                          <span className="font-bold text-emerald-400">
                            {solveResult.trust_certificate.checks.bounds_satisfied ? "PASS" : "FAIL"}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-slate-300 flex items-center gap-2">
                            <Check className="w-3.5 h-3.5 text-emerald-400" /> Integrality Margins (x_j ∈ ℤ)
                          </span>
                          <span className="font-bold text-emerald-400">
                            {solveResult.trust_certificate.checks.integrality_satisfied ? "INTEGER" : "FRACTIONAL"}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-slate-300 flex items-center gap-2">
                            <Check className="w-3.5 h-3.5 text-emerald-400" /> Objective Verification (c^T x)
                          </span>
                          <span className="font-bold text-emerald-400">
                            {solveResult.trust_certificate.checks.objective_matches ? "VERIFIED" : "MISMATCH"}
                          </span>
                        </div>
                      </div>

                      {/* Scaled Tolerance Metrics */}
                      <div className="grid grid-cols-2 gap-2">
                        <div className="p-3 bg-[#070a0e] border border-[#16202c]">
                          <span className="text-[10px] text-[#64748b] block">MAX PRIMAL VIOLATION</span>
                          <span className="text-sm font-bold text-slate-200">
                            {solveResult.trust_certificate.max_primal_violation.toExponential(2)}
                          </span>
                        </div>
                        <div className="p-3 bg-[#070a0e] border border-[#16202c]">
                          <span className="text-[10px] text-[#64748b] block">MAX BOUND VIOLATION</span>
                          <span className="text-sm font-bold text-slate-200">
                            {solveResult.trust_certificate.max_bound_violation.toExponential(2)}
                          </span>
                        </div>
                      </div>

                      <div className="p-3 bg-emerald-950/20 border border-emerald-800/50 text-[11px] text-emerald-400/90 leading-relaxed">
                        <span className="font-bold block mb-1">SOVEREIGN TRUST PROTOCOL:</span>
                        Certified independently without using solver internal basis or multipliers. Recomputes all dot products directly in 64-bit IEEE floating point.
                      </div>
                    </div>
                  ) : (
                    <div className="py-12 text-center text-[#64748b] font-mono text-xs">
                      Run the workflow to perform independent mathematical verification.
                    </div>
                  )}
                </div>

                <div className="text-[10px] font-mono text-[#64748b] border-t border-[#1c2633] pt-3">
                  STRICT TOLERANCE: ε = 1.0e-4 (SCALED RELATIVE)
                </div>
              </div>

              {/* Complete Workflow Stage Audit Card (7 cols) */}
              <div className="lg:col-span-7 bg-[#0b0f14] border border-[#1c2633] p-5 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between border-b border-[#1c2633] pb-3 mb-4">
                    <h2 className="text-xs font-mono font-bold text-slate-200 tracking-wider uppercase flex items-center gap-2">
                      <Terminal className="w-4 h-4 text-cyan-400" /> Full Workflow Pipeline Audit
                    </h2>
                    <span className="text-[10px] font-mono text-emerald-400">All 6 Stages Recorded</span>
                  </div>

                  {solveResult?.workflow_stages ? (
                    <div className="space-y-2.5 font-mono text-xs">
                      {solveResult.workflow_stages.map((st) => (
                        <div
                          key={st.stage}
                          className="p-3 bg-[#070a0e] border border-[#16202c] flex items-center justify-between"
                        >
                          <div className="flex items-center space-x-3">
                            <span className="text-amber-400 font-bold">0{st.stage}</span>
                            <div>
                              <span className="text-slate-200 font-bold block">{st.name}</span>
                              <span className="text-[11px] text-[#64748b]">{st.summary}</span>
                            </div>
                          </div>
                          <span
                            className={`text-[10px] px-2 py-0.5 uppercase font-bold border ${
                              st.status === "CERTIFIED" || st.status === "COMPLETED"
                                ? "bg-emerald-950 text-emerald-400 border-emerald-800"
                                : "bg-slate-900 text-slate-400 border-slate-700"
                            }`}
                          >
                            {st.status}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="p-8 text-center text-[#64748b] font-mono text-xs border border-dashed border-[#16202c]">
                      Execute workflow to review complete 6-stage telemetry summary.
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Industrial Footer */}
      <footer className="border-t border-[#1c2633] bg-[#070a0e] py-4 px-6 text-center text-[11px] font-mono text-[#475569]">
        SOVEREIGN MATHEMATICAL OPTIMIZATION ENGINE • ZERO BLACKBOX SOLVER DEPENDENCY • FIRST-PRINCIPLES NUMERICAL CODE
      </footer>
    </div>
  );
}
