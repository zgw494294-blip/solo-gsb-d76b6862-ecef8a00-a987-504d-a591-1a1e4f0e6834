"""混合样本配比反推：由混合物的条带份额反推各样本的连续质量配比可行域。

问题
----
录入 2～5 份已知**线性** DNA（各 4～5000 bp、仅 A/C/G/T）与共用的
1～8 条现有酶规则（识别串须适用于每份样本）。对每份样本按既有规则做
**完全酶切**（双向查找、允许重叠、同一边界切点合并；**不计单点概率**），
把**同长度片段的碱基数相加**，再除以该样本全长，得到“单位 DNA 质量的
条带份额” a_{i,ℓ}（每份样本各长度份额之和为 1）。

设样本质量配比 w_i ≥ 0、Σw_i = 1，则混合后长度 ℓ 的预测份额为
p_ℓ(w) = Σ_i w_i · a_{i,ℓ}。录入各预测长度对应的观测份额 o_ℓ
（非负、总和为 1、长度不重复，**遗漏长度视为 0**），要求**逐行**
|p_ℓ − o_ℓ| ≤ 容差（容差为 0～0.05 的输入值）。求连续配比的可行域：
一组可行配比，以及每份样本可行配比的最小值与最大值（并给出对应预测
条带）。全部样本预测长度的并集超过 40 种时拒绝计算。

实现
----
可行域是线性不等式组，用**两阶段单纯形法**（Bland 规则防退化循环）
精确求解连续线性规划，**不做离散网格近似**：

- 先解 min t（s.t. −t ≤ p_ℓ − o_ℓ ≤ t、Σw=1、w≥0、t≥0），t* 即
  理论上可达到的最小逐行最大误差；t* ≤ 容差才有可行域；
- 有解时以该最优解作为返回的一组可行配比；
- 再对每份样本分别解 min w_i 与 max w_i（t 固定为 ≤ 容差），
  得到该样本可行配比的最小值与最大值及取该值时的预测条带。

所有 LP 至多 6 个变量、约 3×40+1 行约束，规模很小。
"""

from __future__ import annotations

from .digest import _collect_cut_sites, fragment_lengths

MIN_SAMPLES, MAX_SAMPLES = 2, 5
MIN_SAMPLE_LEN, MAX_SAMPLE_LEN = 4, 5000
MIN_ENZYMES, MAX_ENZYMES = 1, 8
MIN_TOLERANCE, MAX_TOLERANCE = 0.0, 0.05
MIN_ROWS, MAX_ROWS = 0, 40
MAX_PREDICTED_LENGTHS = 40

# 观测份额总和须为 1 的校验容差
_SUM_TOLERANCE = 1e-9
# 单纯形数值容差（系数为 0/±1 与整数/全长 的有理数，1e-10 足够稳定）
_EPS = 1e-10
# 份额/输出截断位数，消除浮点噪声
_SHARE_ROUND_DIGITS = 12
_OUTPUT_ROUND_DIGITS = 10


class MixtureInputError(ValueError):
    """预测长度种类超过上限、观测了非预测长度等前提不满足（由上层转为 HTTP 400）。"""


# ---------------------------------------------------------------------------
# 两阶段单纯形法（纯 Python；Bland 规则；不做网格近似）
# ---------------------------------------------------------------------------

def _solve_lp(
    objective: list[float], constraints: list[tuple[list[float], str, float]]
) -> dict:
    """求 max  c·x  s.t.  A·x ≷ b（逐行 sense 为 "<=" / ">=" / "="）、x ≥ 0。

    返回 {"status": "optimal" / "infeasible" / "unbounded",
          "x": 最优顶点（无界时为可行方向上的可行解）, "objective": 目标值}。
    """
    n = len(objective)
    m = len(constraints)

    # 规范化：b < 0 的行整体取反（sense 翻转），保证初始右端非负
    norm_rows: list[tuple[list[float], str, float]] = []
    for a_raw, sense, rhs_raw in constraints:
        a = list(a_raw)
        rhs = float(rhs_raw)
        if rhs < 0.0:
            a = [-v for v in a]
            rhs = -rhs
            sense = {"<=": ">=", ">=": "<=", "=": "="}[sense]
        norm_rows.append((a, sense, rhs))

    # 变量列布局：x_1..x_n（表列 1..n），随后为松弛/剩余/人工变量
    aux_specs: list[tuple[str, int | None, int | None]] = []
    for _, sense, _ in norm_rows:
        if sense == "<=":
            aux_specs.append(("slack", None, None))
        elif sense == ">=":
            aux_specs.append(("surplus+art", None, None))
        else:
            aux_specs.append(("art", None, None))

    n_aux = sum(1 if t == "slack" else 2 if t == "surplus+art" else 1
                for t, _, _ in aux_specs)
    width = 1 + n + n_aux
    tab = [[0.0] * width for _ in range(m + 1)]
    basics: list[int] = [0] * m  # 每个约束行（表行 k = 下标+1）的基变量表列
    artificial_cols: set[int] = set()
    structural_cols: set[int] = set(range(1, n + 1))

    next_col = n + 1
    for k, (a, sense, rhs) in enumerate(norm_rows):
        row = k + 1
        tab[row][0] = rhs
        for j, v in enumerate(a):
            tab[row][j + 1] = float(v)
        if sense == "<=":
            tab[row][next_col] = 1.0
            basics[k] = next_col
            structural_cols.add(next_col)
            next_col += 1
        elif sense == ">=":
            tab[row][next_col] = -1.0  # 剩余变量
            structural_cols.add(next_col)
            next_col += 1
            tab[row][next_col] = 1.0  # 人工变量
            basics[k] = next_col
            artificial_cols.add(next_col)
            next_col += 1
        else:
            tab[row][next_col] = 1.0  # 人工变量
            basics[k] = next_col
            artificial_cols.add(next_col)
            next_col += 1

    def pivot(r: int, enter: int) -> None:
        piv = tab[r][enter]
        tab[r] = [v / piv for v in tab[r]]
        for i in range(m + 1):
            if i == r:
                continue
            factor = tab[i][enter]
            if factor != 0.0:
                tab[i] = [tab[i][j] - factor * tab[r][j] for j in range(width)]
        basics[r - 1] = enter

    def run_simplex(active_rows: list[int]) -> str:
        """Bland 规则单纯形；返回 "optimal" / "unbounded"。"""
        for _ in range(10000):
            enter = None
            for j in range(1, width):
                if tab[0][j] < -_EPS:  # max：最小下标负检验数入基
                    enter = j
                    break
            if enter is None:
                return "optimal"
            leave = None  # (比值, 基变量列, 行)
            for k in active_rows:
                a = tab[k][enter]
                if a > _EPS:
                    ratio = tab[k][0] / a
                    if (
                        leave is None
                        or ratio < leave[0] - _EPS
                        or (abs(ratio - leave[0]) <= _EPS
                            and basics[k - 1] < leave[1])
                    ):
                        leave = (ratio, basics[k - 1], k)
            if leave is None:
                return "unbounded"
            pivot(leave[2], enter)
        raise RuntimeError("单纯形迭代超过上限（Bland 规则下不应发生）")

    # ---- 第一阶段：max −Σ 人工变量（检验数行中人工变量系数为 +1） ----
    for col in artificial_cols:
        tab[0][col] = 1.0
    for k in range(1, m + 1):
        coeff = tab[0][basics[k - 1]]
        if coeff != 0.0:
            tab[0] = [tab[0][j] - coeff * tab[k][j] for j in range(width)]

    run_simplex(list(range(1, m + 1)))
    if tab[0][0] < -_EPS:  # −Σ人工 < 0，即人工变量之和 > 0
        return {"status": "infeasible", "x": [0.0] * n, "objective": float("inf")}

    # 把仍为基变量的人工变量逐出（其值必为 0）；找不到非零元则该行冗余
    deleted: set[int] = set()
    for k in range(1, m + 1):
        if basics[k - 1] not in artificial_cols:
            continue
        candidate = None
        for j in structural_cols:
            if abs(tab[k][j]) > _EPS:
                candidate = j
                break
        if candidate is None:
            deleted.add(k)
        else:
            pivot(k, candidate)

    # ---- 第二阶段：删去人工列与冗余行，装入真实目标行后继续 ----
    kept_rows = [k for k in range(1, m + 1) if k not in deleted]
    kept_cols = [0] + sorted(structural_cols)
    compact = [
        [tab[k][j] for j in kept_cols]
        for k in [0] + kept_rows
    ]
    compact_basics = [basics[k - 1] for k in kept_rows]
    # 真实目标行：max c·x → 检验数初始为 −c，辅助变量为 0
    col_index = {col: idx for idx, col in enumerate(kept_cols)}
    for j, cj in enumerate(objective):
        compact[0][j + 1] = -float(cj)
    for row_pos, k in enumerate(kept_rows, start=1):
        coeff = compact[0][col_index[compact_basics[row_pos - 1]]]
        if coeff != 0.0:
            compact[0] = [
                compact[0][j] - coeff * compact[row_pos][j]
                for j in range(len(kept_cols))
            ]

    # 用紧凑表继续迭代（局部替换 pivot/run 的表与列宽）
    cwidth = len(kept_cols)
    cm = len(kept_rows)

    def compact_pivot(r: int, enter: int) -> None:
        piv = compact[r][enter]
        compact[r] = [v / piv for v in compact[r]]
        for i in range(cm + 1):
            if i == r:
                continue
            factor = compact[i][enter]
            if factor != 0.0:
                compact[i] = [
                    compact[i][j] - factor * compact[r][j] for j in range(cwidth)
                ]
        compact_basics[r - 1] = kept_cols[enter]

    status = "optimal"
    for _ in range(10000):
        enter = None
        for j in range(1, cwidth):
            if compact[0][j] < -_EPS:
                enter = j
                break
        if enter is None:
            break
        leave = None
        for k in range(1, cm + 1):
            a = compact[k][enter]
            if a > _EPS:
                ratio = compact[k][0] / a
                if (
                    leave is None
                    or ratio < leave[0] - _EPS
                    or (abs(ratio - leave[0]) <= _EPS
                        and compact_basics[k - 1] < leave[1])
                ):
                    leave = (ratio, compact_basics[k - 1], k)
        if leave is None:
            status = "unbounded"
            break
        compact_pivot(leave[2], enter)

    # 从基变量列还原 x
    x = [0.0] * n
    for k in range(1, cm + 1):
        col = compact_basics[k - 1]
        if 1 <= col <= n:
            value = compact[k][0]
            x[col - 1] = 0.0 if abs(value) < _EPS else value
    return {"status": status, "x": x, "objective": compact[0][0]}


# ---------------------------------------------------------------------------
# 条带份额与配比反推
# ---------------------------------------------------------------------------

def _clean(value: float, digits: int = _OUTPUT_ROUND_DIGITS) -> float:
    """输出前截断浮点噪声。"""
    rounded = round(float(value), digits)
    if abs(rounded) < 0.5 * 10 ** -digits:
        return 0.0
    return rounded


def _sample_band_shares(
    sequence: str, enzymes: list[dict]
) -> tuple[dict[int, float], dict[int, int], int]:
    """单份线性样本完全酶切 → ({长度: 单位质量份额}, {长度: 片段条数}, 切点数)。

    同长度片段的碱基数相加后除以全长，即该长度“单位 DNA 质量的条带份额”。
    """
    n = len(sequence)
    cut_map = _collect_cut_sites(sequence, "linear", enzymes)
    lengths = fragment_lengths(n, "linear", sorted(cut_map))
    bases: dict[int, int] = {}
    counts: dict[int, int] = {}
    for length in lengths:
        bases[length] = bases.get(length, 0) + length
        counts[length] = counts.get(length, 0) + 1
    shares = {
        length: round(total / n, _SHARE_ROUND_DIGITS)
        for length, total in bases.items()
    }
    return shares, counts, len(cut_map)


def mixture_deconvolution(
    samples: list[dict],
    enzymes: list[dict],
    observations: list[dict],
    tolerance: float,
) -> dict:
    """由混合条带份额反推连续质量配比可行域。

    samples 为已通过结构校验的线性样本 [{"sequence": str}]；
    enzymes 为已通过校验的共用酶规则（id/site/offset）；
    observations 为 [{"length": int, "observed": float}]（输入序，长度互异）。
    """
    sample_count = len(samples)
    enzyme_ids = [enzyme["id"] for enzyme in enzymes]

    # 每份样本只随全部酶扫描一次
    per_sample: list[dict] = []
    union_lengths: set[int] = set()
    for index, sample in enumerate(samples, start=1):
        shares, counts, cut_count = _sample_band_shares(sample["sequence"], enzymes)
        per_sample.append(
            {
                "index": index,
                "length": len(sample["sequence"]),
                "cut_count": cut_count,
                "fragment_count": sum(counts.values()),
                "shares": shares,
                "counts": counts,
            }
        )
        union_lengths.update(shares)

    if len(union_lengths) > MAX_PREDICTED_LENGTHS:
        raise MixtureInputError(
            f"全部样本完全酶切后的不同预测长度共 {len(union_lengths)} 种，"
            f"超过上限 {MAX_PREDICTED_LENGTHS} 种，拒绝计算"
            "（线性规划的逐行误差约束随预测长度种类线性增长）。"
        )

    lengths = sorted(union_lengths)
    length_count = len(lengths)
    length_pos = {length: k for k, length in enumerate(lengths)}

    # 观测向量：录入值 + 遗漏长度视为 0
    observed_map: dict[int, float] = {length: 0.0 for length in lengths}
    for row in observations:
        length = row["length"]
        if length not in length_pos:
            raise MixtureInputError(
                f"观测长度 {length} bp 不是任何样本的预测片段长度；"
                f"全部预测长度（{length_count} 种，升序）为："
                + ", ".join(str(v) for v in lengths)
            )
        observed_map[length] = row["observed"]

    # 份额矩阵 A[i][k]：样本 i 在第 k 个长度上的单位质量条带份额
    matrix = [
        [info["shares"].get(length, 0.0) for length in lengths]
        for info in per_sample
    ]
    observed = [observed_map[length] for length in lengths]

    def band_rows(weights: list[float]) -> list[dict]:
        rows = []
        for k, length in enumerate(lengths):
            predicted = sum(weights[i] * matrix[i][k] for i in range(sample_count))
            rows.append(
                {
                    "length": length,
                    "observed": _clean(observed[k]),
                    "predicted": _clean(predicted),
                    "error": _clean(abs(predicted - observed[k])),
                }
            )
        return rows

    def feasibility_constraints(limit: str | float) -> list[tuple]:
        """配比约束 + 逐行 |Σ w_i·a_{i,ℓ} − o_ℓ| ≤ limit。

        limit 为数值时 t 固定 ≤ limit；为 "t" 时与变量 t 联动（用于求 t*）。
        变量顺序：w_1..w_s（, t）。
        """
        rows: list[tuple[list[float], str, float]] = [
            ([1.0] * sample_count, "=", 1.0)  # Σw_i = 1
        ]
        for k in range(length_count):
            coeff = [matrix[i][k] for i in range(sample_count)]
            rhs = observed[k]
            if limit == "t":
                # p − t ≤ o 且 p + t ≥ o
                rows.append((coeff + [-1.0], "<=", rhs))
                rows.append((coeff + [1.0], ">=", rhs))
            else:
                rows.append((list(coeff), "<=", rhs + limit))
                rows.append((list(coeff), ">=", rhs - limit))
        return rows

    sample_bands_payload = [
        {
            "index": info["index"],
            "length": info["length"],
            "cut_count": info["cut_count"],
            "fragment_count": info["fragment_count"],
            "bands": [
                {
                    "length": length,
                    "fragment_count": info["counts"].get(length, 0),
                    "share": _clean(info["shares"].get(length, 0.0)),
                }
                for length in lengths
            ],
        }
        for info in per_sample
    ]

    result: dict = {
        "sample_count": sample_count,
        "samples": [
            {
                "index": info["index"],
                "length": info["length"],
                "topology": "linear",
                "cut_count": info["cut_count"],
                "fragment_count": info["fragment_count"],
            }
            for info in per_sample
        ],
        "enzyme_ids": enzyme_ids,
        "enzyme_count": len(enzyme_ids),
        "tolerance": tolerance,
        "predicted_length_count": length_count,
        "predicted_lengths": lengths,
        "observations": [
            {"length": row["length"], "observed": float(row["observed"])}
            for row in observations
        ],
        "sample_bands": sample_bands_payload,
    }

    # ---- 求理论最小逐行最大误差 t*：min t（LP 中按 max −t 解） ----
    rows_t = feasibility_constraints("t")
    obj = [0.0] * sample_count + [-1.0]  # max −t
    solved = _solve_lp(obj, rows_t)
    if solved["status"] == "infeasible":
        # 仅 Σw=1、w≥0 不可能不可行；防御性分支
        result["solvable"] = False
        result["message"] = "无解：配比约束本身不可行。"
        return result
    weights_star = solved["x"][:sample_count]
    t_star = -solved["objective"]

    if t_star > tolerance + _EPS:
        result["solvable"] = False
        result["min_max_error"] = _clean(t_star)
        result["message"] = (
            f"无解：在容差 ±{tolerance:g} 下，不存在非负且总和为 1 的质量配比，"
            "能使每个预测长度的加权预测份额与观测份额之差均不超过容差；"
            f"线性规划给出的理论最小逐行最大误差为 {_clean(t_star):g}"
            "（遗漏长度按观测份额 0 计），仍超过容差。"
        )
        return result

    # ---- 一组可行配比：取 t* 最优点 ----
    feasible_weights = [max(0.0, w) for w in weights_star]
    feasible_bands = band_rows(feasible_weights)

    # ---- 每份样本可行配比的最小值 / 最大值（t ≤ 容差） ----
    rows_fixed = feasibility_constraints(float(tolerance))
    weight_ranges: list[dict] = []
    for i in range(sample_count):
        obj_i = [1.0 if j == i else 0.0 for j in range(sample_count)]
        min_solved = _solve_lp([-v for v in obj_i], rows_fixed)  # max −w_i
        max_solved = _solve_lp(obj_i, rows_fixed)               # max  w_i
        w_min = max(0.0, -min_solved["objective"])
        w_max = max(0.0, max_solved["objective"])
        weights_min = [max(0.0, v) for v in min_solved["x"][:sample_count]]
        weights_max = [max(0.0, v) for v in max_solved["x"][:sample_count]]
        weight_ranges.append(
            {
                "index": i + 1,
                "min": _clean(w_min),
                "max": _clean(w_max),
                "weights_at_min": [_clean(v) for v in weights_min],
                "weights_at_max": [_clean(v) for v in weights_max],
                "min_bands": band_rows(weights_min),
                "max_bands": band_rows(weights_max),
            }
        )

    result["solvable"] = True
    result["min_max_error"] = _clean(t_star)
    result["feasible_mix"] = {
        "weights": [_clean(v) for v in feasible_weights],
        "bands": feasible_bands,
    }
    result["weight_ranges"] = weight_ranges
    return result
