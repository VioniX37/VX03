"""
Netlib compressed-MPS ("EMPS") expander.

The Netlib LP collection (netlib.org/lp/data) distributes models in a compact text encoding
invented by David M. Gay (emps.c). This is a direct port of that expander: it turns the
encoded file into ordinary MPS text, which the sovereign MPS parser then reads.

Encoding summary:
- 92-character alphabet (TRTAB); every character maps to a digit 0..91.
- Indices: digits < 46 are "supersparse" references; long integers are written in base 46.
- Numbers are stored once in a table and referenced by index, or written inline either as
  integers (base 46) or as general floats (base 92 mantissa + decimal exponent).
- Every 71 data lines are followed by a checksum line, which is skipped here.

Names containing blanks are rewritten with '_' so the output parses as free-format MPS.
"""
from typing import List, Optional

TRTAB = "!\"#$%&'()*+,-./0123456789;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~"
_INV = [92] * 256
for _i, _ch in enumerate(TRTAB):
    _INV[ord(_ch)] = _i

_BOUND_TYPES = ["UP", "LO", "FX", "FR", "MI", "PL"]


def is_emps(text: str) -> bool:
    """A compressed Netlib file has a NAME line followed by two lines of integer statistics."""
    lines = [ln for ln in text.splitlines()[:4] if ln.strip()]
    if len(lines) < 3 or not lines[0].startswith("NAME"):
        return False
    try:
        return len([int(t) for t in lines[1].split()]) == 8 and len([int(t) for t in lines[2].split()]) == 3
    except ValueError:
        return False


class _Reader:
    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.pos = 0
        self.ncs = 1  # checksum position counter, mirrors emps.c

    def _raw(self) -> Optional[str]:
        if self.pos >= len(self.lines):
            return None
        s = self.lines[self.pos]
        self.pos += 1
        return s

    def rdline(self) -> Optional[str]:
        while True:
            s = self._raw()
            if s is None:
                return None
            self.ncs += 1
            if self.ncs >= 72:
                self.checkline()
            if s.startswith(":"):  # "mystery line" (unused in the collection)
                continue
            return s

    def checkline(self):
        while True:
            chk = self._raw()
            if chk is None:
                raise ValueError("EMPS: premature end of file in checksum line")
            if chk.startswith(":") and self.ncs <= 72:
                continue
            break
        self.ncs = 1


class _Cursor:
    """Position inside the current encoded line."""

    def __init__(self, reader: _Reader):
        self.r = reader
        self.s = ""
        self.i = 0

    def empty(self) -> bool:
        return self.i >= len(self.s)

    def load(self):
        line = self.r.rdline()
        if line is None:
            raise ValueError("EMPS: premature end of file")
        self.s, self.i = line, 0

    def rest(self) -> str:
        return self.s[self.i:]

    def next(self) -> int:
        c = _INV[ord(self.s[self.i])] if self.i < len(self.s) else 92
        self.i += 1
        return c


def _exindx(z: _Cursor) -> int:
    k = z.next()
    if k >= 46:
        raise ValueError("EMPS: bad index")
    if k >= 23:
        return k - 23
    x = k
    while True:
        k = z.next()
        x = x * 46 + k
        if k >= 46:
            return x - 46


def _exform(z: _Cursor, table: List[str]) -> str:
    """Expand one number; returns its decimal text."""
    k = z.next()
    if k < 46:  # reference into the number table
        z.i -= 1
        idx = _exindx(z)
        if idx < 1 or idx > len(table):
            raise ValueError(f"EMPS: number index {idx} out of range")
        return table[idx - 1]
    out = []
    k -= 46
    if k >= 23:
        out.append("-")
        k -= 23
        nelim = 11
    else:
        nelim = 12
    if k >= 11:  # integer-valued
        k -= 11
        if k >= 6:
            x = k - 6
        else:
            x = k
            while True:
                k = z.next()
                x = x * 46 + k
                if k >= 46:
                    x -= 46
                    break
        out.append(str(x) + ".")
        return "".join(out)

    # general floating point: digit string d (least significant first) and exponent ex
    ex = z.next() - 50
    x = z.next()
    y = 0
    while k > 0:
        k -= 1
        if x >= 100000000:
            y = x
            x = z.next()
        else:
            x = x * 92 + z.next()
    d: List[str] = []
    if y:
        while x > 1:
            d.append(chr(ord("0") + x % 10))
            x //= 10
        while True:
            d.append(chr(ord("0") + y % 10))
            if y < 10:
                break
            y //= 10
    elif x:
        while True:
            d.append(chr(ord("0") + x % 10))
            if x < 10:
                break
            x //= 10
    else:
        d.append("0")
    # value = (digit string) * 10^ex; emps.c only differs in how it pretty-prints this
    out.append("".join(reversed(d)) + "E" + str(ex))
    return "".join(out)


def _name(s: str) -> str:
    s = s[:8].rstrip()
    return s.replace(" ", "_") if s else ""


def expand_emps(text: str) -> str:
    """Expand a compressed Netlib file into free-format MPS text (first problem only)."""
    r = _Reader(text)
    buf = r.rdline()
    while buf is not None and not buf.startswith("NAME"):
        buf = r.rdline()
    if buf is None:
        raise ValueError("EMPS: no NAME line")
    r.ncs = 1
    out: List[str] = [buf.rstrip()]
    s1, s2 = r.rdline(), r.rdline()
    nrow, ncol, _colmx, nz, _nrhs, rhsnz, _nran, ranz = (int(t) for t in s1.split())
    _nbd, bdnz, ns = (int(t) for t in s2.split())
    r.ncs = 1

    z = _Cursor(r)
    table: List[str] = []
    for _ in range(ns):
        if z.empty():
            z.load()
        table.append(_exform(z, table))

    names: List[str] = [""]  # 1-based: rows then columns
    out.append("ROWS")
    for _ in range(nrow):
        line = r.rdline()
        nm = _name(line[1:]) or f"R{len(names)}"
        names.append(nm)
        out.append(f" {line[0]}  {nm}")

    def colout(head: str, count: int, what: int):
        if count == 0:
            if what <= 2:
                out.append(head)
            return
        out.append(head)
        z.s, z.i = "", 0
        cur = ""
        while count > 0:
            count -= 1
            if z.empty():
                z.load()
            while True:
                n = _exindx(z)
                if n != 0:
                    break
                cur = _name(z.rest()) or head
                if what == 1:
                    names.append(cur)
                z.load()
            if what >= 4:
                if n >= 7:
                    raise ValueError(f"EMPS: bad bound type index {n}")
                if z.empty():
                    z.load()
                col = names[nrow + _exindx(z)]
                btype = _BOUND_TYPES[n - 1]
                if n >= 4:
                    out.append(f" {btype} {cur}  {col}")
                    continue
                if z.empty():
                    z.load()
                out.append(f" {btype} {cur}  {col}  {_exform(z, table)}")
            else:
                row = names[n]
                if z.empty():
                    z.load()
                out.append(f"    {cur}  {row}  {_exform(z, table)}")

    colout("COLUMNS", nz, 1)
    colout("RHS", rhsnz, 2)
    colout("RANGES", ranz, 3)
    colout("BOUNDS", bdnz, 4)
    out.append("ENDATA")
    return "\n".join(out) + "\n"
