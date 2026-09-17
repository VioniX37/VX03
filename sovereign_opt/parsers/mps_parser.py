"""
Mathematical Programming System (MPS) format parser.
Supports standard fixed-format and free-format MPS files, integer markers, and bounds.
"""
from typing import Dict, List, Optional, Tuple
import re
from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.variable import VariableType
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense


class MPSParser:
    """
    Parses fixed and free format MPS files into an OptimizationModel.
    """

    @classmethod
    def parse_file(cls, filepath: str) -> OptimizationModel:
        """Reads .mps / .qps / .mps.gz files and Netlib compressed (EMPS) files."""
        if str(filepath).endswith(".gz"):
            import gzip
            with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as f:
                content = f.read()
        else:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        return cls.parse_string(content)

    @classmethod
    def parse_string(cls, content: str) -> OptimizationModel:
        from sovereign_opt.parsers.emps import is_emps, expand_emps
        if is_emps(content):
            content = expand_emps(content)
        lines = content.splitlines()
        model_name = "MPS_Model"
        section = None

        row_types: Dict[str, str] = {}  # row_name -> 'N', 'L', 'G', 'E'
        objective_name: Optional[str] = None
        con_coeffs: Dict[str, Dict[str, float]] = {}
        obj_coeffs: Dict[str, float] = {}
        rhs_values: Dict[str, float] = {}
        ranges: Dict[str, float] = {}
        var_bounds: Dict[str, Tuple[float, float, VariableType]] = {}
        var_order: List[str] = []
        known_vars: set = set()
        quad_terms: Dict[Tuple[str, str], float] = {}
        obj_offset = 0.0
        obj_sense = ObjectiveSense.MINIMIZE
        explicit_lower: set = set()

        is_integer_block = False

        for raw_line in lines:
            # Strip comments (asterisk in column 0 or after content)
            line = raw_line.rstrip()
            if not line or line.startswith("*"):
                continue

            # Check for top-level section headers (no leading whitespace)
            if not line.startswith(" ") and not line.startswith("\t"):
                parts = line.split()
                header = parts[0].upper()
                if header == "NAME":
                    model_name = parts[1] if len(parts) > 1 else "MPS_Model"
                    section = "NAME"
                elif header == "OBJSENSE":
                    section = "OBJSENSE"
                    if len(parts) > 1 and parts[1].upper().startswith("MAX"):
                        obj_sense = ObjectiveSense.MAXIMIZE
                elif header in ("ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "QUADOBJ", "QMATRIX", "ENDATA"):
                    section = header
                continue

            tokens = line.split()
            if not tokens or section is None:
                continue

            if section == "OBJSENSE":
                if tokens[0].upper().startswith("MAX"):
                    obj_sense = ObjectiveSense.MAXIMIZE
                continue

            if section == "ROWS":
                # Format: [Type] [RowName]
                r_type = tokens[0].upper()
                r_name = tokens[1]
                row_types[r_name] = r_type
                if r_type == "N" and objective_name is None:
                    objective_name = r_name
                elif r_type in ("L", "G", "E"):
                    con_coeffs[r_name] = {}

            elif section == "COLUMNS":
                # Check for integer marker
                if "MARKER" in line.upper():
                    if "INTORG" in line.upper():
                        is_integer_block = True
                    elif "INTEND" in line.upper():
                        is_integer_block = False
                    continue

                # Format: ColName Row1 Val1 [Row2 Val2]
                col_name = tokens[0]
                if col_name not in known_vars:
                    known_vars.add(col_name)
                    var_order.append(col_name)
                    v_type = VariableType.INTEGER if is_integer_block else VariableType.CONTINUOUS
                    # Default bounds: 0 <= x <= +inf
                    var_bounds[col_name] = (0.0, float("inf"), v_type)

                # Pairs of (row, val)
                idx = 1
                while idx < len(tokens):
                    r_name = tokens[idx]
                    val = float(tokens[idx + 1])
                    idx += 2
                    if r_name == objective_name:
                        obj_coeffs[col_name] = obj_coeffs.get(col_name, 0.0) + val
                    elif r_name in con_coeffs:
                        con_coeffs[r_name][col_name] = con_coeffs[r_name].get(col_name, 0.0) + val

            elif section == "RHS":
                # Format: [RHS_NAME] Row1 Val1 [Row2 Val2]; an odd token count means a set name is present
                idx = 1 if len(tokens) % 2 == 1 else 0
                while idx + 1 < len(tokens):
                    r_name = tokens[idx]
                    val = float(tokens[idx + 1])
                    idx += 2
                    if r_name == objective_name:
                        # RHS on the objective row is the negated objective constant
                        obj_offset = -val
                    elif r_name in row_types:
                        rhs_values[r_name] = val

            elif section in ("QUADOBJ", "QMATRIX"):
                # Format: Col1 Col2 Value. QUADOBJ lists each off-diagonal once (lower triangle);
                # QMATRIX lists both triangles. Objective term is 1/2 x^T Q x.
                if len(tokens) >= 3:
                    c1, c2, val = tokens[0], tokens[1], float(tokens[2])
                    if c1 == c2:
                        quad_terms[(c1, c1)] = quad_terms.get((c1, c1), 0.0) + val
                    else:
                        key = (c1, c2) if c1 <= c2 else (c2, c1)
                        # model convention: coefficient q on (i,j) contributes 1/2 q x_i x_j
                        weight = 2.0 if section == "QUADOBJ" else 1.0
                        quad_terms[key] = quad_terms.get(key, 0.0) + weight * val

            elif section == "RANGES":
                start_idx = 1 if len(tokens) % 2 == 1 else 0
                idx = start_idx
                while idx < len(tokens):
                    r_name = tokens[idx]
                    val = float(tokens[idx + 1])
                    idx += 2
                    if r_name in con_coeffs:
                        ranges[r_name] = val

            elif section == "BOUNDS":
                # Format: BoundType [BoundID] ColName [Value]
                b_type = tokens[0].upper()
                # The bound-set name is optional: "UP BND X1 4" or "UP X1 4"
                if b_type in ("FR", "MI", "PL", "BV") and len(tokens) == 3:
                    with_set = not (tokens[1] in known_vars and tokens[2] not in known_vars)
                elif b_type in ("FR", "MI", "PL", "BV"):
                    with_set = len(tokens) >= 3
                else:
                    with_set = len(tokens) >= 4
                col_name = tokens[2] if with_set else tokens[1]
                val_idx = 3 if with_set else 2
                val = float(tokens[val_idx]) if len(tokens) > val_idx else 0.0

                if col_name not in var_bounds:
                    if col_name not in known_vars:
                        known_vars.add(col_name)
                        var_order.append(col_name)
                    var_bounds[col_name] = (0.0, float("inf"), VariableType.CONTINUOUS)

                lb, ub, v_type = var_bounds[col_name]

                if b_type == "UP":  # Upper bound
                    ub = val
                    # MPS convention: a negative upper bound with default lower bound implies lb = -inf
                    if val < 0 and lb == 0.0 and col_name not in explicit_lower:
                        lb = float("-inf")
                elif b_type == "LO":  # Lower bound
                    lb = val
                    explicit_lower.add(col_name)
                elif b_type == "FX":  # Fixed variable
                    lb = val
                    ub = val
                elif b_type == "FR":  # Free variable (-inf, +inf)
                    lb = float("-inf")
                    ub = float("inf")
                elif b_type == "MI":  # Lower bound -inf
                    lb = float("-inf")
                elif b_type == "PL":  # Upper bound +inf
                    ub = float("inf")
                elif b_type == "BV":  # Binary variable
                    lb = 0.0
                    ub = 1.0
                    v_type = VariableType.BINARY
                elif b_type == "LI":  # Integer lower bound
                    lb = val
                    v_type = VariableType.INTEGER
                elif b_type == "UI":  # Integer upper bound
                    ub = val
                    v_type = VariableType.INTEGER

                var_bounds[col_name] = (lb, ub, v_type)

        # Build OptimizationModel
        model = OptimizationModel(name=model_name)

        for col_name in var_order:
            lb, ub, v_type = var_bounds.get(col_name, (0.0, float("inf"), VariableType.CONTINUOUS))
            model.add_variable(name=col_name, lower_bound=lb, upper_bound=ub, var_type=v_type)

        for r_name, r_type in row_types.items():
            if r_type == "N":
                continue  # Objective row
            coeffs = con_coeffs.get(r_name, {})
            rhs = rhs_values.get(r_name, 0.0)
            rng = ranges.get(r_name, None)

            if rng is not None:
                # Range constraint
                if r_type == "L":
                    lb = rhs - abs(rng)
                    ub = rhs
                elif r_type == "G":
                    lb = rhs
                    ub = rhs + abs(rng)
                else:
                    lb = rhs if rng > 0 else rhs + rng
                    ub = rhs + rng if rng > 0 else rhs
                model.add_constraint(name=r_name, coefficients=coeffs, sense=ConstraintSense.RANGE, lower_bound=lb, upper_bound=ub)
            else:
                sense_map = {"L": ConstraintSense.LE, "G": ConstraintSense.GE, "E": ConstraintSense.EQ}
                model.add_constraint(name=r_name, coefficients=coeffs, sense=sense_map[r_type], rhs=rhs)

        model.set_objective(
            linear_coefficients=obj_coeffs,
            sense=obj_sense,
            quadratic_coefficients={k: v for k, v in quad_terms.items() if v != 0.0},
            offset=obj_offset,
        )
        return model
