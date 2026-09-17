"""
Public benchmark instance registry.

Files live in benchmarks/data/ (downloaded by benchmarks/fetch.py, not committed):
- netlib/   Netlib LP collection, compressed EMPS text (+ Kennington *.gz)
- infeas/   Netlib infeasible LP collection
- miplib/   MIPLIB 2017 instances (*.mps.gz) + miplib2017-v31.solu optimal values
- qp/       Maros-Meszaros convex QP collection (*.QPS)

Published optimal objectives come from the collections themselves (Netlib readme summary
table, MIPLIB .solu). Where no published value exists, the comparison solver is the reference.
"""
import gzip
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sovereign_opt.parsers.emps import expand_emps, is_emps

DATA = os.path.join(os.path.dirname(__file__), "data")
CACHE = os.path.join(DATA, "cache")


@dataclass
class Instance:
    name: str
    collection: str  # netlib | kennington | infeas | miplib | qp
    path: str
    tags: List[str] = field(default_factory=list)
    published_optimum: Optional[float] = None
    expected_status: str = "optimal"


# Categories used by the robustness report. Tags follow the literature on these instances.
TAGS: Dict[str, List[str]] = {
    "degen2": ["degenerate"], "degen3": ["degenerate"], "25fv47": ["degenerate"],
    "scsd8": ["degenerate"], "ship12l": ["degenerate"], "80bau3b": ["degenerate", "scale"],
    "perold": ["ill-conditioned"], "pilot4": ["ill-conditioned"], "greenbea": ["ill-conditioned", "scale"],
    "d2q06c": ["ill-conditioned", "scale"], "fit2p": ["scale", "dense-columns"],
    "ken-07": ["scale", "network"], "osa-07": ["scale"], "pds-02": ["scale", "network"], "cre-a": ["scale"],
    "pk1": ["weak-relaxation"], "markshare_4_0": ["weak-relaxation"], "enlight_hard": ["weak-relaxation"],
    "mas76": ["weak-relaxation"], "gt2": ["big-M"], "misc07": ["weak-relaxation"],
    "stcqp1": ["scale"], "aug2dc": ["scale"], "cvxqp1_m": ["scale"],
}

# Known quirks of published reference values, shown next to the result.
NOTES: Dict[str, str] = {
    "e226": "Netlib's published optimum omits the objective constant (-7.113) stored in the RHS of the cost row; "
            "with the constant, HiGHS and the sovereign engine both give -11.6389.",
    "greenbea": "The Netlib readme summary lists -7.2462405908E+07 (MINOS, 1989), while the same readme reports that "
                "CPLEX finds -7.2555248130E+07, which is lower. HiGHS and the sovereign engine both reach -7.2555248130E+07 "
                "and our point is independently certified feasible and optimal.",
    "GOULDQP2": "Our certified-feasible point has a lower objective than HiGHS's; the published Maros-Meszaros optimum "
                "is 1.8427534e-04, matching ours.",
}

NETLIB_SMALL = ["afiro", "sc50a", "sc50b", "adlittle", "blend", "share2b", "sc105", "stocfor1", "kb2",
                "recipe", "scagr7", "israel", "sc205", "brandy", "e226", "agg", "bandm", "scfxm1",
                "ship04s", "25fv47", "degen2", "perold", "pilot4"]
NETLIB_LARGE = ["degen3", "sctap3", "scsd8", "ship12l", "stocfor2", "greenbea", "d2q06c", "80bau3b", "fit2p"]
KENNINGTON = ["ken-07", "pds-02", "cre-a", "osa-07"]
INFEASIBLE = ["itest2", "itest6", "klein1", "bgprtr", "woodinfe", "galenet"]
MIPLIB = ["flugpl", "gen-ip002", "p0201", "gt2", "neos5", "pk1", "markshare_4_0", "enlight_hard",
          "mas76", "misc07", "dcmulti"]
QP_SMALL = ["HS21", "HS35", "HS51", "HS76", "HS118", "GENHS28", "HS268", "LOTSCHD", "QPCBLEND",
            "CVXQP1_S", "CVXQP2_S", "CVXQP3_S", "DUALC1", "PRIMALC1", "DUAL1", "GOULDQP2", "QPCBOEI2"]
QP_LARGE = ["MOSARQP1", "AUG3DCQP", "CVXQP1_M", "STCQP1", "AUG2DC"]

SUITES = {
    "quick": ["netlib:afiro", "netlib:sc50a", "netlib:blend", "miplib:flugpl", "qp:HS21"],
    "ci": ["netlib:afiro", "netlib:sc50a", "netlib:blend", "netlib:israel", "netlib:brandy", "infeas:itest2",
           "infeas:klein1", "miplib:flugpl", "miplib:gt2"],
    "netlib": [f"netlib:{n}" for n in NETLIB_SMALL],
    "netlib-large": [f"netlib:{n}" for n in NETLIB_LARGE] + [f"kennington:{n}" for n in KENNINGTON],
    "infeas": [f"infeas:{n}" for n in INFEASIBLE],
    "miplib": [f"miplib:{n}" for n in MIPLIB],
    "qp": [f"qp:{n}" for n in QP_SMALL],
    "qp-large": [f"qp:{n}" for n in QP_LARGE],
}
SUITES["robust"] = ["stress:blend:3", "stress:blend:6", "stress:israel:4", "stress:share2b:5",

    "netlib:degen2", "netlib:degen3", "netlib:25fv47", "netlib:scsd8",
    "netlib:perold", "netlib:pilot4", "netlib:greenbea", "netlib:d2q06c",
    "miplib:pk1", "miplib:markshare_4_0", "miplib:enlight_hard", "miplib:gt2",
] + SUITES["infeas"]
SUITES["all"] = (SUITES["netlib"] + SUITES["netlib-large"] + SUITES["infeas"] + SUITES["miplib"]
                 + SUITES["qp"] + SUITES["qp-large"])


def _netlib_optima() -> Dict[str, float]:
    path = os.path.join(DATA, "netlib", "readme")
    out: Dict[str, float] = {}
    if not os.path.exists(path):
        return out
    row = re.compile(r"^([A-Z0-9\-]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(?:[BR]+\s+)?([-+0-9.E]+)\s*$")
    for line in open(path, encoding="latin-1"):
        m = row.match(line.rstrip())
        if m and m.group(1).lower() not in out:
            out[m.group(1).lower()] = float(m.group(6))
    return out


# Kennington optima from lp/data/kennington/readme (published to 8 significant digits)
KENNINGTON_OPTIMA = {"ken-07": -6.7952044e+08, "pds-02": 2.8857862e+10, "cre-a": 2.3595407e+07,
                     "osa-07": 5.3572252e+05}


def _miplib_optima() -> Dict[str, float]:
    path = os.path.join(DATA, "miplib", "miplib2017-v31.solu")
    out: Dict[str, float] = {}
    if os.path.exists(path):
        for line in open(path):
            parts = line.split()
            if len(parts) == 3 and parts[0] == "=opt=":
                out[parts[1]] = float(parts[2])
    return out


def get_instance(key: str) -> Instance:
    collection, name = key.split(":", 1)
    tags = list(TAGS.get(name.lower(), []))
    if collection == "netlib":
        return Instance(name, collection, os.path.join(DATA, "netlib", name), tags, _netlib_optima().get(name))
    if collection == "kennington":
        return Instance(name, collection, os.path.join(DATA, "netlib", name + ".gz"), tags,
                        KENNINGTON_OPTIMA.get(name))
    if collection == "infeas":
        return Instance(name, collection, os.path.join(DATA, "infeas", name), tags + ["infeasible"],
                        None, expected_status="infeasible")
    if collection == "miplib":
        return Instance(name, collection, os.path.join(DATA, "miplib", name + ".mps.gz"), tags,
                        _miplib_optima().get(name))
    if collection == "qp":
        return Instance(name, collection, os.path.join(DATA, "qp", name + ".QPS"), tags, None)
    if collection == "stress":
        # stress:<netlib name>:<decades>  rows and columns rescaled by random powers of ten
        base, decades = name.split(":")
        b = get_instance("netlib:" + base)
        return Instance(name.replace(":", "_x1e"), collection, b.path, ["ill-conditioned", "synthetic-stress"],
                        b.published_optimum)
    raise KeyError(key)


def suite(name: str) -> List[Instance]:
    return [get_instance(k) for k in SUITES[name]]


def stress_model(base: str, decades: int, seed: int = 11):
    """
    Badly scaled but mathematically equivalent copy of a Netlib LP: row i multiplied by 10^r_i and
    column j substituted x_j = 10^s_j x'_j, with r, s uniform integers in [-decades, decades].
    Row scaling leaves the feasible set unchanged; column substitution keeps the optimal objective.
    """
    import numpy as np
    from sovereign_opt.model.model import OptimizationModel
    from sovereign_opt.model.constraint import ConstraintSense
    from sovereign_opt.parsers.mps_parser import MPSParser
    m = MPSParser.parse_file(mps_path(get_instance("netlib:" + base)))
    rng = np.random.default_rng(seed)
    rs = {c: 10.0 ** rng.integers(-decades, decades + 1) for c in m.constraint_names}
    cs = {v: 10.0 ** rng.integers(-decades, decades + 1) for v in m.variable_names}
    out = OptimizationModel(name=f"{base}_stress_1e{decades}")
    for v in m.variable_names:
        var = m.variables[v]
        out.add_variable(v, var.lower_bound / cs[v], var.upper_bound / cs[v], var.var_type)
    for cn in m.constraint_names:
        con = m.constraints[cn]
        r = rs[cn]
        coeffs = {v: a * r * cs[v] for v, a in con.coefficients.items()}
        if con.sense == ConstraintSense.RANGE:
            out.add_constraint(cn, coeffs, ConstraintSense.RANGE, lower_bound=con.lower_bound * r, upper_bound=con.upper_bound * r)
        else:
            out.add_constraint(cn, coeffs, con.sense, rhs=con.rhs * r)
    out.set_objective({v: c * cs[v] for v, c in m.objective.linear_coefficients.items()}, m.objective.sense,
                      offset=m.objective.offset)
    return out


def mps_path(inst: Instance) -> str:
    """Plain free-format MPS copy of the instance (readable by both our parser and HiGHS)."""
    os.makedirs(CACHE, exist_ok=True)
    if inst.collection == "stress":
        out = os.path.join(CACHE, f"stress_{inst.name}.mps")
        if not os.path.exists(out):
            from sovereign_opt.parsers.mps_writer import write_mps
            base, dec = inst.name.split("_x1e")
            with open(out, "w") as f:
                f.write(write_mps(stress_model(base, int(dec))))
        return out
    out = os.path.join(CACHE, f"{inst.collection}_{inst.name}.mps")
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(inst.path):
        return out
    if inst.path.endswith(".gz"):
        with gzip.open(inst.path, "rt", encoding="latin-1") as f:
            text = f.read()
    else:
        text = open(inst.path, encoding="latin-1").read()
    if is_emps(text):
        text = expand_emps(text)
    with open(out, "w", encoding="latin-1") as f:
        f.write(text)
    return out
