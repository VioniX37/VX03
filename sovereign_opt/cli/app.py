"""
Interactive command-line interface for the Sovereign Mathematical Optimization Engine (v2).
Powered by rich terminal formatting and the unified solve pipeline.
"""
import sys

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from sovereign_opt import __version__
from sovereign_opt.parsers.mps_parser import MPSParser
from sovereign_opt.parsers.lp_parser import LPParser
from sovereign_opt.solvers.dispatch import solve_model, ALL_ALGORITHMS
from sovereign_opt.runtime.device import DeviceDetector
from benchmarks.netlib.afiro import build_netlib_afiro
from benchmarks.industrial.refinery_blending import build_refinery_blending_model
from benchmarks.industrial.power_dispatch import build_unit_commitment_model
from benchmarks.industrial.portfolio_selection import build_portfolio_selection_model

console = Console(legacy_windows=False)

PRESETS = {
    "netlib_afiro": build_netlib_afiro,
    "refinery_blending_lp": lambda: build_refinery_blending_model(as_qp=False),
    "refinery_blending_qp": lambda: build_refinery_blending_model(as_qp=True),
    "power_unit_commitment": lambda: build_unit_commitment_model(time_periods=4),
    "power_unit_commitment_24h": lambda: build_unit_commitment_model(time_periods=24),
    "portfolio_miqp": build_portfolio_selection_model,
}


def display_model_card(model):
    meta = model.get_metadata()
    table = Table(title="[MODEL STRUCTURE]", show_header=True, header_style="bold cyan")
    table.add_column("Property", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Model Name", meta.name)
    table.add_row("Problem Class", f"[yellow]{meta.problem_class}[/yellow]")
    table.add_row("Variables", f"{meta.num_variables:,} (Continuous: {meta.num_continuous}, Integer: {meta.num_integer}, Binary: {meta.num_binary})")
    table.add_row("Constraints", f"{meta.num_constraints:,}")
    table.add_row("Nonzeros (nnz)", f"{meta.num_nonzeros:,}")
    table.add_row("Quadratic Terms", f"{meta.num_quadratic_terms:,}")
    table.add_row("Matrix Density", f"{meta.density * 100:.2f}%")
    table.add_row("Dynamic Range", f"{meta.min_coefficient:.2e} to {meta.max_coefficient:.2e}")
    console.print(table)


def _fmt(value, spec=",.6f"):
    return format(value, spec) if value is not None else "N/A"


def run_pipeline(model, algo_choice="auto", use_presolve=True, time_limit=60.0):
    console.print(Panel.fit(
        f"[bold blue]Sovereign Mathematical Optimization Engine v{__version__}[/bold blue]\n"
        "[dim]AI-Guided | Sparse-First | Certified Optimality[/dim]"
    ))
    display_model_card(model)

    with console.status("[bold green]Presolve -> strategy -> core solve -> postsolve -> certificate..."):
        outcome = solve_model(model, algorithm=algo_choice, enable_presolve=use_presolve, time_limit_seconds=time_limit)

    stats = outcome.presolve_stats
    if stats is not None:
        ptable = Table(title="[PRESOLVE REDUCTIONS]", show_header=True, header_style="bold magenta")
        ptable.add_column("Metric", style="dim")
        ptable.add_column("Original")
        ptable.add_column("Presolved")
        ptable.add_column("Reduction", style="bold green")
        ptable.add_row("Variables", str(stats.original_vars), str(stats.presolved_vars), f"-{stats.var_reduction_pct:.1f}%")
        ptable.add_row("Constraints", str(stats.original_cons), str(stats.presolved_cons), f"-{stats.con_reduction_pct:.1f}%")
        ptable.add_row("Nonzeros", str(stats.original_nnz), str(stats.presolved_nnz), f"-{stats.nnz_reduction_pct:.1f}%")
        for label, count in (
            ("Fixed variables", stats.fixed_vars_count), ("Singleton rows", stats.singleton_rows_count),
            ("Redundant rows", stats.redundant_rows_count), ("Forcing rows", stats.forcing_rows_count),
            ("Doubleton aggregations", stats.doubleton_eliminations), ("Dominated columns", stats.dominated_cols_count),
            ("Parallel rows", stats.parallel_rows_count), ("Duplicate columns", stats.duplicate_cols_count),
            ("Integer bound tightenings", stats.tightened_bounds_count), ("Coefficient tightenings", stats.coefficient_tightenings),
        ):
            if count:
                ptable.add_row(label, "", "", str(count))
        console.print(ptable)

    rec = outcome.recommendation
    ml_card = (
        f"[bold]Recommended Algorithm:[/bold] [green]{rec.recommended_algorithm.upper()}[/green]"
        f"  ->  dispatched: [green]{outcome.algorithm.upper()}[/green] ({outcome.solver_name})\n"
        f"[bold]Confidence Score:[/bold] {rec.confidence_score * 100:.1f}%\n"
        f"[bold]Target Hardware:[/bold] {rec.recommended_hardware.upper()}\n"
        f"[bold]Branching / Cuts / Heuristics:[/bold] {rec.branching_strategy} / {rec.cut_strategy} / {rec.heuristic_intensity}\n"
        f"[bold]Safety Fallback:[/bold] {rec.deterministic_fallback}"
    )
    for note in outcome.notes:
        ml_card += f"\n[yellow]note:[/yellow] {note}"
    console.print(Panel(ml_card, title="[ML STRATEGY RECOMMENDATION]", border_style="cyan"))

    result, cert = outcome.result, outcome.certificate
    rtable = Table(title="[SOLVER RESULTS & TELEMETRY]", show_header=True, header_style="bold green")
    rtable.add_column("Field")
    rtable.add_column("Value", style="bold")
    status_text = result.status.value.upper()
    rtable.add_row("Status", f"[bold green]{status_text}[/bold green]" if result.is_optimal else f"[yellow]{status_text}[/yellow]")
    rtable.add_row("Objective Value", _fmt(result.objective_value))
    rtable.add_row("Runtime", f"{outcome.timings.get('total', 0.0):.4f} seconds")
    rtable.add_row("Iterations / Pivots", str(result.iterations))
    if result.nodes_explored > 0:
        rtable.add_row("Nodes Explored", str(result.nodes_explored))
        d = result.diagnostics
        rtable.add_row("Cuts Applied", f"{d.get('cuts_applied', 0)} {d.get('cuts_by_type', {})}")
        rtable.add_row("Heuristic Incumbents", str(d.get("heuristic_solutions", {})))
    if result.best_bound is not None:
        rtable.add_row("Best Bound", _fmt(result.best_bound))
    if result.mip_gap is not None:
        rtable.add_row("Final MIP Gap", f"{result.mip_gap * 100:.4f}%")
    if result.dual_solution:
        rtable.add_row("Row Duals / Reduced Costs", f"{len(result.dual_solution)} / {len(result.reduced_costs)}")
    console.print(rtable)

    ok = cert.is_valid
    style = "bold green" if ok else "bold red"
    lines = [f"Validation Status: [{style}]{cert.status}[/{style}]"]
    if cert.status != "NO_SOLUTION_TO_VERIFY":
        lines += [
            f"- Constraints Feasibility: {'OK' if cert.checks.get('constraints_satisfied') else 'VIOLATED'} (Max viol: {cert.max_primal_violation:.2e})",
            f"- Bounds Feasibility: {'OK' if cert.checks.get('bounds_satisfied') else 'VIOLATED'} (Max viol: {cert.max_bound_violation:.2e})",
            f"- Integrality: {'OK' if cert.checks.get('integrality_satisfied') else 'FRACTIONAL'} (Max viol: {cert.max_integrality_violation:.2e})",
            f"- Objective Verification: Recomputed {_fmt(cert.recomputed_objective, ',.4f')} vs Reported {_fmt(cert.reported_objective, ',.4f')} (diff: {cert.objective_difference:.2e})",
            f"- Optimality: {'CERTIFIED' if cert.optimality_certified else 'NOT PROVEN'} via {cert.optimality_method}"
            + (f" (dual viol {cert.max_dual_violation:.2e})" if cert.max_dual_violation is not None else "")
            + (f" (gap {cert.duality_gap:.2e})" if cert.duality_gap is not None else ""),
        ]
    else:
        lines += cert.violation_details
    console.print(Panel("\n".join(lines), title="[INDEPENDENT MATHEMATICAL TRUST CERTIFICATE]", border_style="green" if ok else "red"))
    return outcome


def main():
    parser = argparse.ArgumentParser(description=f"Sovereign Mathematical Optimization Engine v{__version__}")
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="Run a built-in benchmark demonstration")
    demo_parser.add_argument("--preset", choices=list(PRESETS), default="refinery_blending_lp")
    demo_parser.add_argument("--algorithm", choices=list(ALL_ALGORITHMS), default="auto")
    demo_parser.add_argument("--no-presolve", action="store_true")
    demo_parser.add_argument("--time-limit", type=float, default=60.0)

    solve_parser = subparsers.add_parser("solve", help="Solve an MPS / QPS or LP file")
    solve_parser.add_argument("file", help="Path to .mps, .qps or .lp file")
    solve_parser.add_argument("--algorithm", choices=list(ALL_ALGORITHMS), default="auto")
    solve_parser.add_argument("--no-presolve", action="store_true")
    solve_parser.add_argument("--time-limit", type=float, default=60.0)

    subparsers.add_parser("info", help="Display hardware and runtime device info")

    args = parser.parse_args()

    if args.command == "info":
        info = DeviceDetector.get_info()
        console.print(Panel(
            f"Engine version: {__version__}\nCPU Cores: {info.cpu_cores}\nCUDA Hardware: {info.has_cuda} ({info.cuda_device_name})\n"
            f"Preferred Execution Device: {info.preferred_device}\nAlgorithms: {', '.join(ALL_ALGORITHMS)}",
            title="System Info",
        ))
    elif args.command == "solve":
        path = args.file.lower()
        model = MPSParser.parse_file(args.file) if path.endswith((".mps", ".qps")) else LPParser.parse_file(args.file)
        run_pipeline(model, algo_choice=args.algorithm, use_presolve=not args.no_presolve, time_limit=args.time_limit)
    else:
        preset = getattr(args, "preset", "refinery_blending_lp")
        run_pipeline(PRESETS[preset](), algo_choice=getattr(args, "algorithm", "auto"),
                     use_presolve=not getattr(args, "no_presolve", False), time_limit=getattr(args, "time_limit", 60.0))


if __name__ == "__main__":
    main()
