"""
CPLEX LP file format parser.
Reads linear and quadratic objective optimization problems written in .lp format.
"""
import re
from typing import Dict, List, Tuple
from sovereign_opt.model.model import OptimizationModel
from sovereign_opt.model.variable import VariableType
from sovereign_opt.model.constraint import ConstraintSense
from sovereign_opt.model.objective import ObjectiveSense


class LPParser:
    """
    Parses standard LP format files into an OptimizationModel.
    """

    @classmethod
    def parse_file(cls, filepath: str) -> OptimizationModel:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return cls.parse_string(content)

    @classmethod
    def parse_string(cls, content: str) -> OptimizationModel:
        # Strip comments (\ is comment delimiter in LP format)
        lines = []
        for line in content.splitlines():
            line = line.split("\\")[0].strip()
            if line:
                lines.append(line)

        text = " \n ".join(lines)

        # Section splitters
        section_pattern = re.compile(
            r"\b(MINIMIZE|MAXIMIZE|MIN|MAX|SUBJECT TO|SUCH THAT|S\.T\.|ST|BOUNDS|GENERALS|GENERAL|INTEGERS|INTEGER|BINARIES|BINARY|END)\b",
            re.IGNORECASE,
        )

        matches = list(section_pattern.finditer(text))
        sections = {}
        for i, m in enumerate(matches):
            header = m.group(1).upper()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sec_text = text[start:end].strip()
            sections[header] = sec_text

        # 1. Objective Sense & Expression
        obj_sense = ObjectiveSense.MINIMIZE
        obj_text = ""
        for h in ("MINIMIZE", "MIN"):
            if h in sections:
                obj_sense = ObjectiveSense.MINIMIZE
                obj_text = sections[h]
                break
        for h in ("MAXIMIZE", "MAX"):
            if h in sections:
                obj_sense = ObjectiveSense.MAXIMIZE
                obj_text = sections[h]
                break

        # Check for objective name (e.g. obj: 2 x1 + 3 x2)
        if ":" in obj_text:
            _, obj_text = obj_text.split(":", 1)

        # 2. Constraints text
        con_text = ""
        for h in ("SUBJECT TO", "SUCH THAT", "S.T.", "ST"):
            if h in sections:
                con_text = sections[h]
                break

        # 3. Bounds text
        bounds_text = sections.get("BOUNDS", "")

        # 4. Integer / Binary variables
        int_vars = set()
        bin_vars = set()
        for h in ("GENERALS", "GENERAL", "INTEGERS", "INTEGER"):
            if h in sections:
                int_vars.update(sections[h].split())
        for h in ("BINARIES", "BINARY"):
            if h in sections:
                bin_vars.update(sections[h].split())

        model = OptimizationModel(name="LP_Model")
        discovered_vars = set()

        # Helper to parse linear combination (e.g., "+ 2.5 x1 - x2 + 4.0 x3")
        term_regex = re.compile(r"([+-]?\s*\d*\.?\d+(?:[eE][+-]?\d+)?|[+-]?)\s*([a-zA-Z_][a-zA-Z0-9_#$]*)")

        def parse_terms(expr: str) -> Dict[str, float]:
            terms = {}
            # Replace spaces around signs
            clean_expr = expr.replace("+", " +").replace("-", " -")
            for coeff_str, var_name in term_regex.findall(clean_expr):
                var_name = var_name.strip()
                coeff_str = coeff_str.replace(" ", "").strip()
                if not var_name:
                    continue
                if not coeff_str or coeff_str == "+":
                    coeff = 1.0
                elif coeff_str == "-":
                    coeff = -1.0
                else:
                    coeff = float(coeff_str)
                terms[var_name] = terms.get(var_name, 0.0) + coeff
                discovered_vars.add(var_name)
            return terms

        obj_terms = parse_terms(obj_text)

        # Parse constraints
        # Each constraint may look like: "c1: 2 x1 + 3 x2 <= 10" or "2 x1 + 3 x2 <= 10"
        con_lines = con_text.split("\n")
        # Recombine multiline constraints
        combined_cons = []
        cur_con = ""
        for cl in con_lines:
            cl = cl.strip()
            if not cl:
                continue
            if cur_con:
                cur_con += " " + cl
            else:
                cur_con = cl
            if any(op in cur_con for op in ("<=", ">=", "=", "<", ">")):
                # Check if it has the RHS
                last_token = cur_con.split()[-1]
                try:
                    float(last_token)
                    combined_cons.append(cur_con)
                    cur_con = ""
                except ValueError:
                    pass
        if cur_con:
            combined_cons.append(cur_con)

        parsed_constraints = []
        for idx, con_str in enumerate(combined_cons):
            name = f"c{idx+1}"
            if ":" in con_str:
                name_part, con_str = con_str.split(":", 1)
                name = name_part.strip()

            if "<=" in con_str:
                lhs, rhs = con_str.split("<=", 1)
                sense = ConstraintSense.LE
            elif ">=" in con_str:
                lhs, rhs = con_str.split(">=", 1)
                sense = ConstraintSense.GE
            elif "=" in con_str:
                lhs, rhs = con_str.split("=", 1)
                sense = ConstraintSense.EQ
            else:
                continue

            rhs_val = float(rhs.strip())
            terms = parse_terms(lhs)
            parsed_constraints.append((name, terms, sense, rhs_val))

        # Parse bounds
        var_bounds: Dict[str, Tuple[float, float]] = {}
        for line in bounds_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Cases:
            # l <= x <= u
            # x <= u
            # x >= l
            # x = val
            # x free
            if "free" in line.lower():
                v = line.split()[0]
                var_bounds[v] = (float("-inf"), float("inf"))
                discovered_vars.add(v)
            elif "<=" in line and line.count("<=") == 2:
                parts = line.split("<=")
                l_val = float(parts[0].strip())
                v = parts[1].strip()
                u_val = float(parts[2].strip())
                var_bounds[v] = (l_val, u_val)
                discovered_vars.add(v)
            elif "<=" in line:
                parts = line.split("<=")
                v = parts[0].strip()
                u_val = float(parts[1].strip())
                var_bounds[v] = (var_bounds.get(v, (0.0, float("inf")))[0], u_val)
                discovered_vars.add(v)
            elif ">=" in line:
                parts = line.split(">=")
                v = parts[0].strip()
                l_val = float(parts[1].strip())
                var_bounds[v] = (l_val, var_bounds.get(v, (0.0, float("inf")))[1])
                discovered_vars.add(v)
            elif "=" in line:
                parts = line.split("=")
                v = parts[0].strip()
                val = float(parts[1].strip())
                var_bounds[v] = (val, val)
                discovered_vars.add(v)

        # Register all variables
        for v in sorted(discovered_vars):
            lb, ub = var_bounds.get(v, (0.0, float("inf")))
            if v in bin_vars:
                v_type = VariableType.BINARY
            elif v in int_vars:
                v_type = VariableType.INTEGER
            else:
                v_type = VariableType.CONTINUOUS
            model.add_variable(name=v, lower_bound=lb, upper_bound=ub, var_type=v_type)

        # Add constraints
        for name, terms, sense, rhs_val in parsed_constraints:
            model.add_constraint(name=name, coefficients=terms, sense=sense, rhs=rhs_val)

        # Set objective
        model.set_objective(linear_coefficients=obj_terms, sense=obj_sense)

        return model
