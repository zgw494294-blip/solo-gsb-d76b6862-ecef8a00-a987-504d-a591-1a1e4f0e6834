"""混合样本配比反推：由混合样本的观测条带份额反推各已知线性 DNA 的连续质量配比可行域。

问题
----
录入 2～5 份已知线性 DNA（各 4～5000 bp、仅 A/C/G/T）与全部样本共用的
1～8 条现有酶规则；识别串须适用于每份样本（不得长于任一样本）。各样本按
既有双向查找、允许重叠、同一边界切点合并规则做**完全酶切**（**不计单点
概率**，线性序列两端 0/n 为固定端点，端点命中不另增片段）。每份样本把
**同长度片段的碱基数相加**，再除以该样本全长，得到“单位 DNA 质量的条带
份额”：

    share_i(L) = （样本 i 完全酶切后长度恰为 L 的片段长度之和）/ 样本 i 全长

所有样本出现过的不同长度取并集 L_1..L_m（**m 超过 40 时拒绝**）。录入各
预测长度对应的观测份额：长度不得重复、份额非负且总和为 1；**遗漏长度视为
份额 0**；录入未出现过的长度视为参数错误。

设样本质量配比 w_i ≥ 0、Σw_i = 1（连续取值，**不做离散网格近似**），则
混合物在长度 L 上的预测份额为各样本条带份额的加权和：

    pred(L) = Σ_i share_i(L) · w_i

要求**逐行**绝对误差 |pred(L) − obs(L)| ≤ tolerance（0～0.05）。求该线性
可行域：返回一组可行配比、该配比下的预测条带，以及每份样本可行配比的
最小值、最大值及取到端点时对应的预测条带；无解时明确说明（并给出约束所
允许的最小最大行误差）。

实现
----
全部约束关于 w 为线性，用**两阶段单纯形法**（tableau + Bland 反循环规则）
在纯 Python 内求解，无第三方依赖：第一阶段求初始可行基（其任一基本可行解
即一组可行配比），第二阶段对每个样本分别最小化 / 最大化其权重，共
1 + 2k 次求解。无解时再解一个引入松弛标量 t 的线性规划，得到约束允许的
最小最大行误差（minimax），用于无解说明。规模上界：变量 ≤ 6、约束
≤ 2·40 + 1 = 81 行。
"""

from __future__ import annotations

from .digest import _collect_cut_sites, fragment_lengths

MIN_SAMPLES, MAX_SAMPLES = 2, 5
MIN_SAMPLE_LEN, MAX_SAMPLE_LEN = 4, 5000
MIN_ENZYMES, MAX_ENZYMES = 1, 8
MIN_OBSERVATIONS, MAX_OBSERVATIONS = 1, 40
MIN_TOLERANCE, MAX_TOLERANCE = 0.0, 0.05
MAX_PREDICTED_LENGTHS = 40

# 份额总和为 1 的判定容差（份额为 O(1) 量级，1e-9 远低于任何输入容差）
SUM_TOLERANCE = 1e-9
# 单纯形数值参数
_PIVOT_EPS = 1e-12  # 系数视为正的阈值
_FEAS_EPS = 1e-7  # 第一阶段最优值视为 0 的阈值
_OBJ_EPS = 1e-10  # 检验数视为负的阈值
_MAX_PIVOTS = 200_000  # Bland 规则下理论上有限，这里仅作兜底
# 输出清理：把浮点噪声截断到 0 / 1
_OUTPUT_EPS = 5e-9
_OUTPUT_ROUND = 10


class MixtureInputError(ValueError):
    """预测长度种数超过上限、观测长度未知等前提不满足（由上层转为 HTTP 400）。"""


# ---------------------------------------------------------------------------
# 两阶段单纯形法（纯 Python）
# ---------------------------------------------------------------------------

class _LpError(RuntimeError):
    """线性规划数值求解失败（迭代超过兜底上限等不应出现的情况）。"""


def _clean(value: float) -> float:
    """把 -1e-12 一类的浮点噪声截断为 0。"""
    return 0.0 if abs(value) < _PIVOT_EPS else value


def _pivot(tableau: list[list[float]], basis: list[int], entering: int,
           rows: int, col_count: int) -> bool:
    """以 entering 列为进基变量做一次单纯形转轴（离基行由最小比值选定）。

    返回 True；无合法离基行（无界）时不改动 tableau 并返回 False。
    """
    leaving = -1
    best_ratio = float("inf")
    for i in range(rows):
        coeff = tableau[i][entering]
        if coeff > _PIVOT_EPS:
            rhs = tableau[i][col_count]
            if rhs < 0.0 and abs(rhs) < _PIVOT_EPS:
                rhs = 0.0  # 退化情形的数值噪声
            ratio = rhs / coeff
            # Bland：比值并列时取下标最小的行
            if ratio < best_ratio - _PIVOT_EPS or (
                abs(ratio - best_ratio) <= _PIVOT_EPS and (leaving == -1 or i < leaving)
            ):
                best_ratio = ratio
                leaving = i
    if leaving == -1:
        return False  # 无界

    pivot_row = tableau[leaving]
    pivot_val = pivot_row[entering]
    for j in range(col_count + 1):
        pivot_row[j] /= pivot_val
    pivot_row[entering] = 1.0  # 消除除法噪声
    for i in range(rows + 1):
        if i == leaving:
            continue
        factor = tableau[i][entering]
        if factor == 0.0:
            continue
        row = tableau[i]
        for j in range(col_count + 1):
            row[j] -= factor * pivot_row[j]
        row[entering] = 0.0
    tableau[leaving][entering] = 1.0
    basis[leaving] = entering
    return True


def _read_solution(tableau: list[list[float]], basis: list[int],
                   n: int, col_count: int) -> list[float]:
    """从规范形 tableau 读出原始变量 x_0..x_{n-1} 的取值。"""
    x = [0.0] * n
    for i, basic_col in enumerate(basis):
        if basic_col < n:
            value = tableau[i][col_count]
            x[basic_col] = 0.0 if abs(value) < _OUTPUT_EPS else value
    return x


def solve_lp(c: list[float], constraints: list[tuple[str, list[float], float]]
             ) -> tuple[list[float], float]:
    """两阶段单纯形：最小化 c·x，x ≥ 0。

    constraints 为 [(op, a, b)]，op ∈ {"<=", ">=", "=="}，即 a·x ≤ b /
    a·x ≥ b / a·x == b；b 允许为负（内部整行取负规范化）。返回 (x, 最优值)；
    无可行解时返回 (None, None)。
    """
    n = len(c)
    m = len(constraints)
    # 列布局：x(0..n-1)、松弛/剩余变量 s_i(n+i)、人工变量 a_i(n+m+i)、RHS
    col_count = n + 2 * m
    rhs_col = col_count

    tableau: list[list[float]] = [[0.0] * (col_count + 1) for _ in range(m)]
    basis: list[int] = [-1] * m
    artificial_rows: list[int] = []

    for i, (op, a, b) in enumerate(constraints):
        row = tableau[i]
        sign = 1.0
        if b < 0.0:  # 规范化为 RHS ≥ 0，整行取负；≤ 与 ≥ 随之互换
            b = -b
            sign = -1.0
            if op == "<=":
                op = ">="
            elif op == ">=":
                op = "<="
        for j, coeff in enumerate(a):
            row[j] = sign * coeff
        if op == "<=":
            row[n + i] = 1.0
            basis[i] = n + i  # 松弛变量直接作初始基
        elif op == ">=":
            row[n + i] = -1.0
            row[n + m + i] = 1.0
            basis[i] = n + m + i
            artificial_rows.append(i)
        elif op == "==":
            row[n + m + i] = 1.0
            basis[i] = n + m + i
            artificial_rows.append(i)
        else:  # pragma: no cover - 上层不会传入
            raise _LpError(f"未知的约束算子：{op}")
        row[rhs_col] = b

    # ---- 第一阶段：最小化人工变量之和 ----
    # 采用最小化规范形 tableau：目标行 = Σ（人工基所在行），即
    # z − Σ(非人工)x = Σb；基变量所在的人工列系数清零。进基取正检验数列。
    obj = [0.0] * (col_count + 1)
    tableau.append(obj)
    for i in artificial_rows:
        for j in range(col_count + 1):
            obj[j] += tableau[i][j]
    for i in artificial_rows:
        obj[basis[i]] = 0.0

    non_art_cols = list(range(n + m))  # 人工变量永不进基
    pivots = 0
    while True:
        entering = -1
        for j in non_art_cols:  # Bland：取下标最小的正检验数列
            if obj[j] > _OBJ_EPS:
                entering = j
                break
        if entering == -1:
            break
        if not _pivot(tableau, basis, entering, m, col_count):
            raise _LpError("第一阶段出现无界解")  # 不可能，仅作防御
        pivots += 1
        if pivots > _MAX_PIVOTS:
            raise _LpError("第一阶段转轴次数超过上限")

    # 最小化规范形：RHS 即当前目标值 z（人工变量之和）
    phase1_value = _clean(obj[rhs_col])
    if phase1_value > _FEAS_EPS:
        return None, None  # 无可行解

    # 人工变量若仍在基中（取值必为 0，退化）：在该行任取一个非零系数的
    # 非人工列做一次高斯-约当消元（不做比值检验——RHS 为 0，不改变可行性），
    # 把人工变量逐出基；整行非人工系数全为 0 则是冗余约束，直接删除该行。
    for i in range(m - 1, -1, -1):
        if basis[i] >= n + m:
            pivot_col = -1
            for j in non_art_cols:
                if abs(tableau[i][j]) > _PIVOT_EPS:
                    pivot_col = j
                    break
            if pivot_col == -1:
                del tableau[i]
                del basis[i]
            else:
                pivot_val = tableau[i][pivot_col]
                for j in range(col_count + 1):
                    tableau[i][j] /= pivot_val
                tableau[i][pivot_col] = 1.0
                for r in range(m + 1):
                    if r == i:
                        continue
                    factor = tableau[r][pivot_col]
                    if factor == 0.0:
                        continue
                    for j in range(col_count + 1):
                        tableau[r][j] -= factor * tableau[i][j]
                    tableau[r][pivot_col] = 0.0
                tableau[i][pivot_col] = 1.0
                basis[i] = pivot_col

    # ---- 第二阶段：去掉人工列，换入真实目标 ----
    # 最小化目标行：z − c·x = 0（非基列初始检验数为 −c_j），再消去基变量列
    rows = len(basis)
    for row in tableau:
        del row[n + m:col_count]
    obj2 = tableau[-1]
    for j in range(n + m + 1):
        obj2[j] = 0.0
    for j, coeff in enumerate(c):
        obj2[j] = -coeff
    for i in range(rows):
        basic_col = basis[i]
        cb = obj2[basic_col]
        if cb != 0.0:
            for j in range(n + m + 1):
                obj2[j] -= cb * tableau[i][j]

    pivots = 0
    while True:
        entering = -1
        for j in range(n + m):
            if obj2[j] > _OBJ_EPS:  # 正检验数可使目标下降
                entering = j
                break
        if entering == -1:
            break
        if not _pivot(tableau, basis, entering, rows, n + m):
            raise _LpError("第二阶段出现无界解")
        pivots += 1
        if pivots > _MAX_PIVOTS:
            raise _LpError("第二阶段转轴次数超过上限")

    x = _read_solution(tableau, basis, n, n + m)
    objective = sum(c[j] * x[j] for j in range(n))
    return x, objective


# ---------------------------------------------------------------------------
# 混合样本配比反推
# ---------------------------------------------------------------------------

def _digest_sample_shares(sequence: str, enzymes: list[dict]
                          ) -> tuple[dict[int, float], dict]:
    """一份线性样本完全酶切，返回 ({长度: 单位质量条带份额}, 酶切明细)。"""
    n = len(sequence)
    cut_map = _collect_cut_sites(sequence, "linear", enzymes)
    boundaries = sorted(cut_map)
    lengths = fragment_lengths(n, "linear", boundaries)

    base_counts: dict[int, int] = {}
    for length in lengths:
        base_counts[length] = base_counts.get(length, 0) + length
    shares = {length: count / n for length, count in base_counts.items()}

    detail = {
        "length": n,
        "cut_count": len(cut_map),
        "cut_boundaries": boundaries,
        "fragment_count": len(lengths),
        "fragment_lengths": sorted(lengths),
        "base_counts": dict(sorted(base_counts.items())),
    }
    return shares, detail


def _weighted_shares(weights: list[float],
                     sample_shares: list[dict[int, float]],
                     lengths: list[int]) -> list[float]:
    """各长度的预测份额 = 各样本条带份额的加权和。"""
    return [
        sum(sample_shares[i].get(length, 0.0) * weights[i]
            for i in range(len(weights)))
        for length in lengths
    ]


def _round_value(value: float) -> float:
    """输出用：截断浮点噪声后四舍五入。"""
    rounded = round(value, _OUTPUT_ROUND)
    return 0.0 if abs(rounded) < _OUTPUT_EPS else rounded


def _round_weights(weights: list[float], count: int) -> list[float]:
    cleaned: list[float] = []
    for value in weights:
        if abs(value) < _OUTPUT_EPS:
            cleaned.append(0.0)
        elif abs(value - 1.0) < _OUTPUT_EPS:
            cleaned.append(1.0)
        else:
            cleaned.append(value)
    # 消除归一化噪声：把与 1 的残差补到绝对值最大的分量上
    residual = 1.0 - sum(cleaned)
    if abs(residual) <= 1e-7:
        idx = max(range(count), key=lambda i: abs(cleaned[i]))
        cleaned[idx] += residual
    return [_round_value(value) for value in cleaned]


def _build_band_rows(lengths: list[int], observed: list[float],
                     predicted: list[float], tolerance: float,
                     include_shares: bool = False,
                     sample_shares: list[dict[int, float]] | None = None
                     ) -> list[dict]:
    rows: list[dict] = []
    for idx, length in enumerate(lengths):
        error = abs(predicted[idx] - observed[idx])
        row = {
            "length": length,
            "observed_share": _round_value(observed[idx]),
            "predicted_share": _round_value(predicted[idx]),
            "abs_error": _round_value(error),
            "within_tolerance": error <= tolerance + 1e-9,
        }
        if include_shares and sample_shares is not None:
            row["sample_shares"] = [
                _round_value(shares.get(length, 0.0)) for shares in sample_shares
            ]
        rows.append(row)
    return rows


def _mixture_constraints(matrix: list[list[float]], observed: list[float],
                         tolerance: float) -> list[tuple[str, list[float], float]]:
    """可行域约束：w ≥ 0（求解器内置）、Σw = 1、逐行 |A·w − obs| ≤ tol。"""
    k = len(matrix)
    constraints: list[tuple[str, list[float], float]] = [
        ("==", [1.0] * k, 1.0)
    ]
    for col, obs in enumerate(observed):
        a = [matrix[i][col] for i in range(k)]
        constraints.append(("<=", a, obs + tolerance))
        constraints.append((">=", a, max(obs - tolerance, 0.0)))
    return constraints


def mixture_deconvolution(
    samples: list[dict],
    enzymes: list[dict],
    observations: list[dict],
    tolerance: float,
) -> dict:
    """求连续质量配比的可行域（两阶段单纯形，不做离散网格近似）。

    samples 为已通过校验的规范化线性样本：[{"sequence": str}]；enzymes 为
    已通过校验的共用酶规则（id/site/offset）；observations 为
    [{"length": int, "share": float}]（长度互异、非负、总和为 1 的校验在上层，
    未知长度与长度种数上限在本函数内判定）。
    """
    k = len(samples)

    # ---- 各样本完全酶切，汇总单位质量条带份额 ----
    sample_shares: list[dict[int, float]] = []
    sample_details: list[dict] = []
    for index, sample in enumerate(samples, start=1):
        shares, detail = _digest_sample_shares(sample["sequence"], enzymes)
        sample_shares.append(shares)
        detail["index"] = index
        sample_details.append(detail)

    length_set = set()
    for shares in sample_shares:
        length_set.update(shares)
    lengths = sorted(length_set)
    if len(lengths) > MAX_PREDICTED_LENGTHS:
        raise MixtureInputError(
            f"全部样本完全酶切后出现的不同片段长度共 {len(lengths)} 种，"
            f"超过上限 {MAX_PREDICTED_LENGTHS} 种；请减少样本数、酶条数或"
            "改用切点更少的识别串"
        )

    # ---- 观测份额：长度必须是预测长度，遗漏长度补 0 ----
    observed_map: dict[int, float] = {}
    for row in observations:
        length = row["length"]
        if length not in length_set:
            raise MixtureInputError(
                f"观测长度 {length} 不是任何样本完全酶切后的预测片段长度"
                f"（预测长度共 {len(lengths)} 种：{', '.join(map(str, lengths))}）；"
                "只能录入预测长度，未录入的预测长度其观测份额视为 0"
            )
        observed_map[length] = row["share"]
    observed = [observed_map.get(length, 0.0) for length in lengths]

    # 份额矩阵 A[i][col]：样本 i 在第 col 种长度上的单位质量条带份额
    matrix = [
        [sample_shares[i].get(length, 0.0) for length in lengths]
        for i in range(k)
    ]
    constraints = _mixture_constraints(matrix, observed, tolerance)

    result: dict = {
        "sample_count": k,
        "samples": [
            {
                "index": detail["index"],
                "length": detail["length"],
                "topology": "linear",
                "cut_count": detail["cut_count"],
                "cut_boundaries": detail["cut_boundaries"],
                "fragment_count": detail["fragment_count"],
                "fragment_lengths": detail["fragment_lengths"],
                "base_counts": detail["base_counts"],
            }
            for detail in sample_details
        ],
        "enzyme_ids": [enzyme["id"] for enzyme in enzymes],
        "enzyme_count": len(enzymes),
        "predicted_length_count": len(lengths),
        "predicted_lengths": lengths,
        "observations": [
            {"length": length, "observed_share": _round_value(value)}
            for length, value in zip(lengths, observed)
        ],
        "observation_row_count": len(observations),
        "tolerance": tolerance,
    }

    # ---- 第一组 LP：可行性（目标函数取 0，第一阶段的 BFS 即可行配比） ----
    feasible_weights, _ = solve_lp([0.0] * k, constraints)

    if feasible_weights is None:
        # ---- 无解：minimax LP 求约束允许的最小最大行误差 ----
        # 变量 (w_0..w_{k-1}, t)，t ≥ 0；增加 |A·w − obs| ≤ t 两行约束并最小化 t
        mm_constraints: list[tuple[str, list[float], float]] = [
            ("==", [1.0] * k + [0.0], 1.0)
        ]
        for col, obs in enumerate(observed):
            a = [matrix[i][col] for i in range(k)]
            mm_constraints.append(("<=", a + [-1.0], obs))
            mm_constraints.append(("<=", [-v for v in a] + [-1.0], -obs))
        _, min_max_error = solve_lp([0.0] * k + [1.0], mm_constraints)
        result["solvable"] = False
        result["minimal_max_error"] = (
            None if min_max_error is None else _round_value(min_max_error)
        )
        result["message"] = (
            f"无解：在非负且总和为 1 的连续质量配比（{k} 份样本）上，"
            f"不存在使全部 {len(lengths)} 种预测长度的条带份额逐行绝对误差"
            f"均不超过容差 ±{tolerance:g} 的配比"
            + (
                f"；把容差放宽到至少 {result['minimal_max_error']:g} 才可能有解。"
                if min_max_error is not None
                else "。"
            )
        )
        return result

    weights = _round_weights(feasible_weights, k)
    predicted = _weighted_shares(weights, sample_shares, lengths)
    max_error = max(
        (abs(predicted[idx] - observed[idx]) for idx in range(len(lengths))),
        default=0.0,
    )

    result["solvable"] = True
    result["feasible_mixture"] = {
        "weights": weights,
        "predicted_bands": _build_band_rows(
            lengths, observed, predicted, tolerance,
            include_shares=True, sample_shares=sample_shares,
        ),
        "max_error": _round_value(max_error),
    }

    # ---- 每份样本可行配比的最小值 / 最大值及对应预测条带 ----
    ranges: list[dict] = []
    for i in range(k):
        endpoint: dict = {}
        for kind, objective in (("min", 1.0), ("max", -1.0)):
            c = [0.0] * k
            c[i] = objective
            w, value = solve_lp(c, constraints)
            if w is None:  # 可行域已确认非空，不应发生
                raise _LpError(f"求解样本 {i + 1} 的配比{kind}值时意外失败")
            w_clean = _round_weights(w, k)
            pred = _weighted_shares(w_clean, sample_shares, lengths)
            endpoint[kind] = {
                "value": _round_value(value if kind == "min" else -value),
                "weights": w_clean,
                "predicted_bands": _build_band_rows(
                    lengths, observed, pred, tolerance
                ),
            }
        ranges.append({"sample_index": i + 1, "min": endpoint["min"],
                       "max": endpoint["max"]})
    result["sample_ranges"] = ranges
    return result
