export interface SystemInfo {
  cpu_cores: number;
  has_cuda: boolean;
  cuda_device_name: string | null;
  preferred_device: string;
  engine_version?: string;
  algorithms?: string[];
}

export interface Preset {
  id: string;
  name: string;
  category: string;
  description: string;
  problem_class: string;
}

export interface RuizStats {
  d1_min: number;
  d1_max: number;
  d2_min: number;
  d2_max: number;
  norm_before: number;
  norm_after: number;
  iterations: number;
}

export interface SparsityPoint {
  row: number;
  col: number;
  val: number;
}

export interface ModelMetadata {
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
  variables?: Array<{ name: string; type: string; lb: number | string; ub: number | string }>;
  constraints?: Array<{ name: string; sense: string; rhs: number; num_terms: number }>;
}

export interface PresolveData {
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
  redundant_rows_count?: number;
  forcing_rows_count?: number;
  doubleton_eliminations?: number;
  dominated_cols_count?: number;
  parallel_rows_count?: number;
  duplicate_cols_count?: number;
  tightened_bounds_count?: number;
  coefficient_tightenings?: number;
  presolve_passes?: number;
  fixed_vars_list?: Array<{ name: string; value: number }>;
}

export interface MLRecommendation {
  recommended_algorithm: string;
  algorithm_probabilities: Record<string, number>;
  confidence_score: number;
  recommended_hardware: string;
  branching_strategy: string;
  feature_attributions: Record<string, number>;
  deterministic_fallback: string;
  cut_strategy?: string;
  heuristic_intensity?: string;
  relaxation_solver?: string | null;
  features?: Record<string, number>;
}

export interface IterationTraceEntry {
  iteration: number;
  objective?: number;
  phase?: string;
  status?: string;
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

export interface TreeTraceNode {
  node_id: number;
  parent_id: number | null;
  depth: number;
  lower_bound: number;
  status: string; // ROOT | BRANCH | PRUNED_BOUND | INTEGER_INCUMBENT | INFEASIBLE
  branch_var: string;
  branch_condition: string;
  bound_note?: string;
}

export interface PostsolveMappingItem {
  variable: string;
  value: number;
  resolution_state: string;
}

export interface WorkflowStage {
  stage: number;
  name: string;
  status: string;
  summary: string;
}

export interface TrustCertificate {
  is_valid: boolean;
  status: string;
  max_primal_violation: number;
  max_bound_violation: number;
  max_integrality_violation: number;
  recomputed_objective: number | null;
  objective_difference: number;
  checks: Record<string, boolean>;
  violation_details: string[];
  optimality_certified?: boolean;
  optimality_method?: string;
  max_dual_violation?: number | null;
  duality_gap?: number | null;
}

export interface SolveResult {
  status: string;
  objective_value: number | null;
  iterations: number;
  runtime_seconds: number;
  nodes_explored: number;
  mip_gap: number | null;
  best_bound?: number | null;
  algorithm_used: string;
  algorithm_key?: string;
  problem_class?: string;
  notes?: string[];
  timings?: Record<string, number>;
  primal_solution: Record<string, number>;
  dual_solution?: Record<string, number>;
  reduced_costs?: Record<string, number>;
  diagnostics?: {
    iteration_trace?: IterationTraceEntry[];
    tree_trace?: TreeTraceNode[];
    basis_size?: number;
    pricing_method?: string;
    barrier_mu?: number;
    primal_residual?: number;
    dual_residual?: number;
    active_constraints_count?: number;
    algorithm?: string;
    termination?: string;
    cuts_applied?: number;
    cuts_by_type?: Record<string, number>;
    heuristic_solutions?: Record<string, number>;
    root_lp_bound?: number | null;
    dual_recovery?: string;
    crossover_pivots?: number;
  };
  postsolve_mapping?: PostsolveMappingItem[];
  workflow_stages?: WorkflowStage[];
  trust_certificate: TrustCertificate;
}

export type ViewId = 1 | 2 | 3 | 4 | 5 | 6;
