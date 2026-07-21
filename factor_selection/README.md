# 第三阶段：因子有效性检验与特征工程

本项目用于完成 A 股因子的有效性评价、相关性分析、正交化处理和核心因子筛选。项目采用扁平化目录结构，`evaluation.py`、`selection.py`、`correlation_sensitivity.py`、`walk_forward.py` 与 `weights.py` 可以直接导入使用。

## 当前包含的功能

### evaluation.py

- 计算未来收益，并限制未来收益标签不能跨越训练期、验证期和测试期边界。
- 计算因子每日 Rank IC 时间序列。
- 计算 IC 均值、ICIR、正 IC 比例和 t 统计量。
- 计算因子分层收益、多空收益差和分层单调性。
- 计算因子覆盖率和缺失率。
- 支持剔除市场、行业和规模暴露后的残差收益评价。
- 支持按行业和市场环境比较因子有效性。

### selection.py

- 计算因子相关性矩阵。
- 使用 Ridge 回归计算特征重要性。
- 对高相关因子进行正交化处理。
- 综合训练期、验证期、残差收益和市场环境指标筛选因子。
- 支持负向有效因子的方向统一。
- 限制同逻辑因子的入选数量，默认每组最多保留2个。
- 按相关性阈值剔除冗余因子，默认阈值为0.65。
- 输出入选因子、因子权重、入选原因和未入选原因。

### correlation_sensitivity.py

- 保留相关性阈值网格、因子池稳定性、Jaccard 相似度和权重距离计算逻辑。
- 可以独立使用不依赖回测的因子池比较辅助函数。
- 完整的验证期策略敏感性测试接口已经保留，但当前会提示回测系统尚未接入。

### walk_forward.py

- 生成滚动训练期、验证期和测试期窗口。
- 检查各区间顺序、边界和测试窗口是否重叠。
- 窗口生成逻辑不依赖策略回测，可以正常使用。

### weights.py

- 支持等权、IC 加权和 ICIR 加权。
- 按月度或季度滚动估计因子权重。
- 支持历史窗口、权重上限、指数衰减、平滑和单次最大权重变化。
- 只使用生效日期之前的历史 IC，避免未来信息泄漏。

## 当前限制：敏感性测试暂不可运行

本项目没有接入第四阶段的策略构建与回测模块，因此不包含以下内容：

- `strategy` 目录；
- 组合评分与投资组合构建；
- 验证期策略回测；
- 交易成本、调仓和成交约束；
- 基于验证期 Alpha、Information Ratio、Sharpe、换手率和交易成本的相关性阈值敏感性测试。

因此，当前可以手动设置不同的 `max_correlation` 参数并比较因子筛选结果，也可以使用 `correlation_sensitivity.py` 中不依赖回测的因子池稳定性函数，但无法运行包含验证期策略表现比较的完整相关性阈值敏感性测试。后续将接入第四阶段的组合构建、交易执行、回测和绩效分析系统，接入后即可恢复完整敏感性分析功能。

普通的因子评价、相关性分析、正交化、因子筛选、Walk-Forward 窗口生成和滚动因子权重功能不受影响，可以正常运行。

## 环境安装

在本项目一级目录打开 PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

也可以使用 Conda：

```powershell
conda env create -f environment.yml
conda activate factor-selection
```

## 基本使用方法

```python
import pandas as pd

from evaluation import add_forward_returns, evaluate_factors
from selection import factor_correlation, ridge_feature_importance, select_factors_detailed

data = pd.read_csv("factors.csv", parse_dates=["date"])
factor_names = ["momentum_20", "volatility_20", "book_to_price", "roe"]

evaluated = add_forward_returns(data, period=1)
summary, ic_series, quantile_returns = evaluate_factors(
    evaluated,
    factor_names,
    quantiles=5,
    min_observations=30,
)

correlation = factor_correlation(evaluated, factor_names)
ridge = ridge_feature_importance(evaluated, factor_names)

selected, weights, details = select_factors_detailed(
    summary,
    correlation,
    ridge_importance=ridge,
    max_correlation=0.65,
    max_factors=12,
    max_factors_per_family=2,
    minimum_coverage=0.60,
)
```

## 输入数据要求

基础评价至少需要以下字段：

- `date`：交易日期；
- `symbol`：股票代码；
- `close`：收盘价；
- 需要评价的各因子列。

残差收益评价还需要行业、市值和市场暴露字段。

## 运行测试

```powershell
python -m pytest
```

测试覆盖当前已经接入的评价、筛选、Walk-Forward、滚动权重和敏感性辅助函数，不会调用尚未接入的策略回测；同时会验证完整敏感性测试能够返回清晰的“等待接入回测系统”提示。
