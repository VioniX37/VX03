"""
Interactive command-line interface for the Sovereign Mathematical Optimization Engine.
Powered by rich terminal formatting.
"""
import sys
import os

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from sovereign_opt.parsers.mps_parser import MPSParser
from sovereign_opt.parsers.lp_parser import LPParser
from sovereign_opt.presolve.presolver import Presolver
from sovereign_opt.ml.strategy import MLStrategyEngine
from sovereign_opt.solvers.lp.simplex import RevisedSimplexSolver
from sovereign_opt.solvers.lp.interior_point import InteriorPointSolver
from sovereign_opt.solvers.milp.branch_bound import BranchAndBoundSolver
from sovereign_opt.solvers.qp.active_set import ActiveSetQPSolver
from sovereign_opt.validation.validator import IndependentValidator
from sovereign_opt.runtime.device import DeviceDetector
from benchmarks.netlib.afiro import build_netlib_afiro
from benchmarks.industrial.refinery_blending import build_refinery_blending_model
from benchmarks.industrial.power_dispatch import build_unit_commitment_model

console = Console(legacy_windows=False)


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
    table.add_row("Matrix Density", f"{meta.density * 100:.2f}%")
    table.add_row("Dynamic Range", f"{meta.min_coefficient:.2e} to {meta.max_coefficient:.2e}")
    console.print(table)


def run_pipeline(model, algo_choice="auto", use_presolve=True):
    console.print(Panel.fit("[bold blue]Sovereign Mathematical Optimization Engine[/bold blue]\n[dim]AI-Guided | Sparse-First | Sovereign Mathematical Foundations[/dim]"))
    display_model_card(model)

    mapper = None
    presolve_stats = None
    model_to_solve = model

    # 1. Presolve
    if use_presolve:
        with console.status("[bold green]Executing Stage 4 Presolve Reductions..."):
            presolver = Presolver()
            p_model, mapper, presolve_stats = presolver.presolve(model)
            model_to_solve = p_model

        ptable = Table(title="[PRESOLVE REDUCTIONS]", show_header=True, header_style="bold magenta")
        ptable.add_column("Metric", style="dim")
        ptable.add_column("Original")
        ptable.add_column("Presolved")
        ptable.add_column("Reduction", style="bold green")

        ptable.add_row("Variables", str(presolve_stats.original_vars), str(presolve_stats.presolved_vars), f"-{presolve_stats.var_reduction_pct:.1f}%")
        ptable.add_row("Constraints", str(presolve_stats.original_cons), str(presolve_stats.presolved_cons), f"-{presolve_stats.con_reduction_pct:.1f}%")
        ptable.add_row("Nonzeros", str(presolve_stats.original_nnz), str(presolve_stats.presolved_nnz), f"-{presolve_stats.nnz_reduction_pct:.1f}%")
        console.print(ptable)

    # 2. ML Strategy
    with console.status("[bold blue]Running Stage 5 ML Strategy Engine..."):
        ml_engine = MLStrategyEngine()
        rec = ml_engine.recommend(model, presolve_stats)

    ml_card = (
        f"[bold]Recommended Algorithm:[/bold] [green]{rec.recommended_algorithm.upper()}[/green]\n"
        f"[bold]Confidence Score:[/bold] {rec.confidence_score * 100:.1f}%\n"
        f"[bold]Target Hardware:[/bold] {rec.recommended_hardware.upper()}\n"
        f"[bold]Branching Strategy:[/bold] {rec.branching_strategy}\n"
        f"[bold]Safety Fallback:[/bold] {rec.deterministic_fallback}"
    )
    console.print(Panel(ml_card, title="[ML STRATEGY RECOMMENDATION]", border_style="cyan"))

    # Algorithm selection
    selected_algo = rec.recommended_algorithm if algo_choice == "auto" else algo_choice
    p_class = model_to_solve.classify()

    if p_class == "MILP" or selected_algo == "branch_and_bound":
        solver = BranchAndBoundSolver()
    elif p_class == "QP" or selected_algo == "active_set":
        solver = ActiveSetQPSolver()
    elif selected_algo == "interior_point":
        solver = InteriorPointSolver()
    else:
        solver = RevisedSimplexSolver()

    # 3. Solve
    console.print(f"\n[bold]Dispatching to Sovereign Solver:[/bold] [yellow]{solver.name}[/yellow]...")
    start = time.time()
    result = solver.solve(model_to_solve)
    solve_time = time.time() - start

    # 4. Postsolve
    final_primal = dict(result.primal_solution)
    if mapper is not None and result.is_feasible:
        x_full = mapper.restore_primal(result.primal_solution)
        final_primal = {name: float(x_full[i]) for i, name in enumerate(mapper.original_var_names)}

    result.primal_solution = final_primal
    result.runtime_seconds = solve_time

    # 5. Validation
    validator = IndependentValidator()
    cert = validator.verify(model, result)

    # Display Results
    rtable = Table(title="[SOLVER RESULTS & TELEMETRY]", show_header=True, header_style="bold green")
    rtable.add_column("Field")
    rtable.add_column("Value", style="bold")
    rtable.add_row("Status", f"[bold green]{result.status.value.upper()}[/bold green]" if result.is_optimal else f"[yellow]{result.status.value.upper()}[/yellow]")
    rtable.add_row("Objective Value", f"{result.objective_value:,.4f}" if result.objective_value is not None else "N/A")
    rtable.add_row("Runtime", f"{result.runtime_seconds:.4f} seconds")
    rtable.add_row("Iterations / Pivots", str(result.iterations))
    if result.nodes_explored > 0:
        rtable.add_row("Nodes Explored", str(result.nodes_explored))
    if result.mip_gap is not None:
        rtable.add_row("Final MIP Gap", f"{result.mip_gap * 100:.4f}%")
    console.print(rtable)

    # Display Certificate
    cert_style = "bold green" if cert.is_valid else "bold red"
    cert_status = "PASSED" if cert.is_valid else "FAILED"
    con_check = "OK" if cert.checks.get("constraints_satisfied") else "VIOLATED"
    bnd_check = "OK" if cert.checks.get("bounds_satisfied") else "VIOLATED"
    int_check = "OK" if cert.checks.get("integrality_satisfied") else "FRACTIONAL"

    cert_text = (
        f"Validation Status: [{cert_style}]{cert_status}[/{cert_style}]\n"
        f"- Constraints Feasibility: [{cert_style}]{con_check}[/{cert_style}] (Max viol: {cert.max_primal_violation:.2e})\n"
        f"- Bounds Feasibility: [{cert_style}]{bnd_check}[/{cert_style}]\n"
        f"- Integrality: [{cert_style}]{int_check}[/{cert_style}] (Max viol: {cert.max_integrality_violation:.2e})\n"
        f"- Objective Verification: Recomputed {cert.recomputed_objective:,.4f} vs Reported {cert.reported_objective:,.4f} (diff: {cert.objective_difference:.2e})"
    )
    console.print(Panel(cert_text, title="[INDEPENDENT MATHEMATICAL TRUST CERTIFICATE]", border_style="green" if cert.is_valid else "red"))


def main():
    parser = argparse.ArgumentParser(description="Sovereign Mathematical Optimization Engine")
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="Run a built-in benchmark demonstration")
    demo_parser.add_argument("--preset", choices=["netlib_afiro", "refinery_blending_lp", "refinery_blending_qp", "power_unit_commitment"], default="refinery_blending_lp")
    demo_parser.add_argument("--algorithm", choices=["auto", "simplex", "interior_point", "branch_and_bound", "active_set"], default="auto")

    solve_parser = subparsers.add_parser("solve", help="Solve an MPS or LP file")
    solve_parser.add_argument("file", help="Path to .mps or .lp file")
    solve_parser.add_argument("--algorithm", choices=["auto", "simplex", "interior_point", "branch_and_bound", "active_set"], default="auto")

    subparsers.add_parser("info", help="Display hardware and runtime device info")

    args = parser.parse_args()

    if args.command == "info":
        info = DeviceDetector.get_info()
        console.print(Panel(f"CPU Cores: {info.cpu_cores}\nCUDA Hardware: {info.has_cuda} ({info.cuda_device_name})\nPreferred Execution Device: {info.preferred_device}", title="System Info"))
    elif args.command == "demo" or args.command is None:
        preset = getattr(args, "preset", "refinery_blending_lp")
        algo = getattr(args, "algorithm", "auto")
        if preset == "netlib_afiro":
            m = build_netlib_afiro()
        elif preset == "refinery_blending_lp":
            m = build_refinery_blending_model(as_qp=False)
        elif preset == "refinery_blending_qp":
            m = build_refinery_blending_model(as_qp=True)
        else:
            m = build_unit_commitment_model(time_periods=2)
        run_pipeline(m, algo_choice=algo)
    elif args.command == "solve":
        if args.file.endswith(".mps"):
            m = MPSParser.parse_file(args.file)
        else:
            m = LPParser.parse_file(args.file)
        run_pipeline(m, algo_choice=args.algorithm)


if __name__ == "__main__":
    main()
