"""
Netlib LP Benchmark Instance: AFIRO.
Known optimal objective value: -464.75314286.
32 constraints, 27 variables, 83 nonzeros.
"""
from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense


def build_netlib_afiro() -> OptimizationModel:
    model = OptimizationModel(name="AFIRO")

    # Variables
    vars_info = [
        ("X01", 0.0, float("inf")), ("X02", 0.0, float("inf")), ("X03", 0.0, float("inf")),
        ("X04", 0.0, float("inf")), ("X06", 0.0, float("inf")), ("X07", 0.0, float("inf")),
        ("X08", 0.0, float("inf")), ("X09", 0.0, float("inf")), ("X10", 0.0, float("inf")),
        ("X11", 0.0, float("inf")), ("X12", 0.0, float("inf")), ("X13", 0.0, float("inf")),
        ("X14", 0.0, float("inf")), ("X15", 0.0, float("inf")), ("X16", 0.0, float("inf")),
        ("X17", 0.0, float("inf")), ("X18", 0.0, float("inf")), ("X19", 0.0, float("inf")),
        ("X20", 0.0, float("inf")), ("X21", 0.0, float("inf")), ("X22", 0.0, float("inf")),
        ("X23", 0.0, float("inf")), ("X24", 0.0, float("inf")), ("X27", 0.0, float("inf")),
        ("X28", 0.0, float("inf")), ("X29", 0.0, float("inf")), ("X40", 0.0, float("inf")),
    ]
    for v_name, lb, ub in vars_info:
        model.add_variable(name=v_name, lower_bound=lb, upper_bound=ub)

    # Constraints
    cons_data = [
        ("R09", {"X01": -1.0, "X02": 1.0}, ConstraintSense.LE, 0.0),
        ("R10", {"X03": -1.0, "X04": 1.0}, ConstraintSense.LE, 0.0),
        ("X05", {"X01": 1.0, "X03": 1.0}, ConstraintSense.EQ, 80.0),
        ("R12", {"X06": -1.0, "X07": 1.0}, ConstraintSense.LE, 0.0),
        ("R13", {"X08": -1.0, "X09": 1.0}, ConstraintSense.LE, 0.0),
        ("X21_con", {"X06": 1.0, "X08": 1.0}, ConstraintSense.EQ, 80.0),
        ("R14", {"X02": 1.0, "X04": 1.0, "X10": -1.0}, ConstraintSense.EQ, 0.0),
        ("R15", {"X07": 1.0, "X09": 1.0, "X11": -1.0}, ConstraintSense.EQ, 0.0),
        ("R16", {"X10": -0.4, "X11": -0.32, "X12": 1.0}, ConstraintSense.EQ, 0.0),
        ("R17", {"X10": 1.0, "X13": -1.0}, ConstraintSense.EQ, 0.0),
        ("R18", {"X11": 1.0, "X14": -1.0}, ConstraintSense.EQ, 0.0),
        ("R19", {"X12": -1.0, "X15": 1.0}, ConstraintSense.LE, 0.0),
        ("R20", {"X12": -1.0, "X16": 1.0}, ConstraintSense.LE, 0.0),
        ("X22_con", {"X13": 1.0, "X14": 1.0}, ConstraintSense.LE, 500.0),
        ("R21", {"X15": 1.0, "X17": -1.0}, ConstraintSense.EQ, 0.0),
        ("R22", {"X16": 1.0, "X18": -1.0}, ConstraintSense.EQ, 0.0),
        ("R23", {"X17": 1.0, "X19": -1.0}, ConstraintSense.EQ, 0.0),
        ("R24", {"X18": 1.0, "X20": -1.0}, ConstraintSense.EQ, 0.0),
        ("R25", {"X19": -1.0, "X21": 1.0}, ConstraintSense.LE, 0.0),
        ("R26", {"X20": -1.0, "X22": 1.0}, ConstraintSense.LE, 0.0),
        ("X23_con", {"X21": 1.0, "X22": 1.0}, ConstraintSense.LE, 500.0),
        ("R27", {"X23": 1.0, "X24": -1.0}, ConstraintSense.EQ, 0.0),
        ("R28", {"X24": 1.0, "X27": -1.0}, ConstraintSense.EQ, 0.0),
        ("R29", {"X27": -1.0, "X28": 1.0}, ConstraintSense.LE, 0.0),
        ("X25_con", {"X28": 1.0}, ConstraintSense.LE, 300.0),
        ("R30", {"X29": 1.0, "X40": -1.0}, ConstraintSense.EQ, 0.0),
        ("X26_con", {"X40": 1.0}, ConstraintSense.LE, 44.0),
    ]

    for name, coeffs, sense, rhs in cons_data:
        model.add_constraint(name=name, coefficients=coeffs, sense=sense, rhs=rhs)

    # Objective: Minimize
    obj_coeffs = {
        "X01": 0.4, "X02": 0.0, "X03": 0.4, "X04": 0.0,
        "X06": 0.32, "X07": 0.0, "X08": 0.32, "X09": 0.0,
        "X10": -0.6, "X11": -0.48, "X12": 10.0, "X13": 0.0,
        "X14": 0.0, "X15": -10.0, "X16": -10.0, "X17": 0.0,
        "X18": 0.0, "X19": -10.0, "X20": -10.0, "X21": 0.0,
        "X22": 0.0, "X23": 0.0, "X24": 0.0, "X27": -10.0,
        "X28": 0.0, "X29": -10.0, "X40": 0.0,
    }
    model.set_objective(linear_coefficients=obj_coeffs, sense=ObjectiveSense.MINIMIZE)
    return model
