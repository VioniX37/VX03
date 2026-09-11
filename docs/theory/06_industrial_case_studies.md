# Industrial Case Studies & Problem Formulations

The codebase includes three production-relevant industrial scenarios formulated directly in Python:

---

## 1. Refinery Crude Oil & Blendstock Blending (LP / QP)

### 1.1 Engineering Background
Modern petroleum refineries distill crude oils into intermediate fractions and blend them with high-octane streams (e.g. Alkylate, Reformate) to satisfy market fuel standards:
- **Regular Gasoline**: Minimum Octane 87, Maximum Sulfur 0.010 (100 ppm).
- **Premium Gasoline**: Minimum Octane 93, Maximum Sulfur 0.005 (50 ppm).
- **Diesel Fuel**: Minimum Octane / Cetane 80, Maximum Sulfur 0.015 (150 ppm).

### 1.2 Mathematical Model
Let $x_{f, p} \ge 0$ denote barrels per day of feedstock $f \in \mathcal{F}$ blended into product $p \in \mathcal{P}$.
$$\max_{x} \sum_{p \in \mathcal{P}} \sum_{f \in \mathcal{F}} \left( \text{Price}_p - \text{Cost}_f \right) x_{f, p} - \sum_{f \in \text{Heavy}} \sum_{p \in \mathcal{P}} \frac{1}{2} \gamma x_{f, p}^2$$
subject to:
1. **Feedstock Availability**:
   $$\sum_{p \in \mathcal{P}} x_{f, p} \le \text{MaxSupply}_f \quad \forall f \in \mathcal{F}$$
2. **Product Minimum Demand**:
   $$\sum_{f \in \mathcal{F}} x_{f, p} \ge \text{MinDemand}_p \quad \forall p \in \mathcal{P}$$
3. **Octane Specification**:
   $$\sum_{f \in \mathcal{F}} (\text{Octane}_f - \text{MinOctane}_p) x_{f, p} \ge 0 \quad \forall p \in \mathcal{P}$$
4. **Sulfur Limit**:
   $$\sum_{f \in \mathcal{F}} (\text{Sulfur}_f - \text{MaxSulfur}_p) x_{f, p} \le 0 \quad \forall p \in \mathcal{P}$$

---

## 2. Power Generation Unit Commitment (MILP)

### 2.1 Engineering Background
Grid transmission operators must commit thermal and flexible generating units to balance fluctuating consumer demand across discrete time intervals $t \in \{1, \dots, T\}$. Thermal generators cannot start or shut down instantaneously and incur substantial amortized startup costs.

### 2.2 Mathematical Model
- Binary variable $u_{g, t} \in \{0, 1\}$ indicates whether generator $g$ is synchronized and online at hour $t$.
- Continuous variable $p_{g, t} \ge 0$ represents electric power produced (in MW).
$$\min_{u, p} \sum_{t=1}^T \sum_{g \in \mathcal{G}} \left( \text{MarginalCost}_g \cdot p_{g, t} + \text{StartupCost}_g \cdot u_{g, t} \right)$$
subject to:
1. **Demand Balance**:
   $$\sum_{g \in \mathcal{G}} p_{g, t} \ge \text{Demand}_t \quad \forall t \in \{1, \dots, T\}$$
2. **Operating Capacity Envelope**:
   $$\text{MinMW}_g \cdot u_{g, t} \le p_{g, t} \le \text{MaxMW}_g \cdot u_{g, t} \quad \forall g, t$$
3. **Binary Restriction**:
   $$u_{g, t} \in \{0, 1\}$$

---

## 3. Netlib Benchmark Suite (AFIRO)

Standard Netlib LP benchmark instance `AFIRO` tests basic LP solvers on real-world historical production planning equations. Known optimal objective value: $-464.75314286$.
