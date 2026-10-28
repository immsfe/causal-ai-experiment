from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ==============================
# 1. 项目路径
# ==============================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

for folder in [DATA_DIR, REPORT_DIR, FIGURE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "hillstrom.csv"

# ==============================
# 2. 获取真实数据
# ==============================

url = (
    "https://raw.githubusercontent.com/"
    "AdityaDabrase/ab-testing-email-marketing/"
    "main/data/raw/hillstrom.csv"
)

if not DATA_PATH.exists():
    print("正在下载Hillstrom数据...")
    df = pd.read_csv(url)
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

# 部分R数据源带有额外行号列
df = df.drop(
    columns=["rownames", "Unnamed: 0"],
    errors="ignore"
)

print("\n===== 数据基本信息 =====")
print("数据形状：", df.shape)
print("\n字段：", df.columns.tolist())
print("\n前5行：")
print(df.head().to_string())

# ==============================
# 3. 数据质量检查
# ==============================

print("\n===== 缺失值 =====")
print(df.isna().sum())

print("\n===== 实验分组 =====")
print(df["segment"].value_counts())

# 检查关键字段
required = ["segment", "conversion", "visit", "spend"]

for col in required:
    if col not in df.columns:
        raise ValueError(f"缺少关键字段：{col}")

# ==============================
# 4. 计算业务指标
# ==============================

summary = df.groupby("segment").agg(
    users=("conversion", "size"),
    conversions=("conversion", "sum"),
    conversion_rate=("conversion", "mean"),
    visit_rate=("visit", "mean"),
    revenue_per_user=("spend", "mean")
)

print("\n===== 各组业务指标 =====")
print(summary.round(4).to_string())

summary.to_csv(
    REPORT_DIR / "group_summary.csv",
    encoding="utf-8-sig"
)

# ==============================
# 5. A/B测试
# 男性营销组 vs 不发送邮件组
# ==============================

treatment = df.loc[
    df["segment"] == "Mens E-Mail", "conversion"
]

control = df.loc[
    df["segment"] == "No E-Mail", "conversion"
]

if len(treatment) == 0 or len(control) == 0:
    raise ValueError("未找到实验组或对照组，请检查segment取值")

p_t = treatment.mean()
p_c = control.mean()

n_t = len(treatment)
n_c = len(control)

# 平均处理效应的估计
effect = p_t - p_c

# 置信区间的非合并标准误
se = np.sqrt(
    p_t * (1 - p_t) / n_t
    + p_c * (1 - p_c) / n_c
)

# 零假设下的合并比例
p_pool = (
    treatment.sum() + control.sum()
) / (n_t + n_c)

se_null = np.sqrt(
    p_pool * (1 - p_pool)
    * (1 / n_t + 1 / n_c)
)

z = effect / se_null
p_value = 2 * stats.norm.sf(abs(z))

ci_low = effect - 1.96 * se
ci_high = effect + 1.96 * se

print("\n===== A/B实验结果 =====")
print(f"实验组购买率：{p_t:.4%}")
print(f"对照组购买率：{p_c:.4%}")
print(f"绝对提升：{effect * 100:.4f} 个百分点")
print(f"Z统计量：{z:.4f}")
print(f"P值：{p_value:.6f}")
print(
    f"95%置信区间："
    f"[{ci_low * 100:.4f}, "
    f"{ci_high * 100:.4f}] 个百分点"
)

# ==============================
# 6. 保存实验结果
# ==============================

results = pd.DataFrame([{
    "treatment": "Mens E-Mail",
    "control": "No E-Mail",
    "treatment_rate": p_t,
    "control_rate": p_c,
    "absolute_effect": effect,
    "z_statistic": z,
    "p_value": p_value,
    "ci_lower": ci_low,
    "ci_upper": ci_high
}])

results.to_csv(
    REPORT_DIR / "ab_test_results.csv",
    index=False
)

# ==============================
# 7. 可视化
# ==============================

plot_data = summary["conversion_rate"] * 100

plt.figure(figsize=(9, 5))
plt.bar(plot_data.index, plot_data.values)

plt.ylabel("Conversion Rate (%)")
plt.title("Hillstrom Experiment: Conversion Rate")

for i, value in enumerate(plot_data.values):
    plt.text(
        i, value,
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "conversion_rate.png",
    dpi=200
)
plt.close()

print("\n分析完成！")
print("结果已保存到reports文件夹。")
