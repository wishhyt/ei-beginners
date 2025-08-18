数据集托管在 Hugging Face，上层评测命令行工具叫 `eai-eval`。

# 数据集里有什么？

在 Hugging Face 的数据集中，每一行就是一个“任务”，主要字段是：

* `scene_id`/`task_id`/`task_name`：任务与场景标识；
* `natural_language_description`：任务的自然语言描述；
* `original_goal`：PDDL 形式化目标与初始状态；
* `tl_goal`：将目标转成 LTL/一阶逻辑的可判定形式；
* `action_trajectory`：示例动作序列（供评测对比或提示）；
* `transition_model`：环境/动作的转移动力学（逻辑形式）。
  数据包含两个 split：`virtualhome`（338 条）与 `behavior`（100 条）。

# 两种最常见的用法

## ① 直接用 Hugging Face 加载做你自己的训练/推理

```python
from datasets import load_dataset

ds_vh = load_dataset("Inevitablevalor/EmbodiedAgentInterface", split="virtualhome")
row = ds_vh[0]

print(row["task_name"])
print(row["natural_language_description"])
print(row["original_goal"][:300])   # PDDL 片段
print(row["tl_goal"])               # 逻辑目标
print(row["action_trajectory"])     # 参考动作序列（列表/JSON 字符串）
```

如果只想 BEHAVIOR：`split="behavior"`。这非常适合把 `natural_language_description` 作为输入、用 `tl_goal`/`action_trajectory` 作为监督或评测目标，或把 `original_goal` 里的 PDDL 解析后做规划/检验。

## ② 用作者的评测工具 `eai-eval` 复现实验/做细粒度评测

1. 安装与环境：

```bash
conda create -n eai-eval python=3.8 -y
conda activate eai-eval
pip install eai-eval
# 可选：需要 BEHAVIOR 仿真时再装 iGibson 及资源
python -m behavior_eval.utils.install_igibson_utils
python -m behavior_eval.utils.download_utils
```

（如要评测“状态转移”模块，建议先跑 `python examples/pddl_tester.py` 验证 PDDL 规划器装好。）

2. 生成模型要用的 prompts（命令会按模块/数据集批量产出提示）：

```bash
eai-eval --dataset virtualhome --eval-type action_sequencing --mode generate_prompts
# 其他模块：transition_modeling / goal_interpretation / subgoal_decomposition
```
**这里生成上一步的json prompts,再调模型。from 本地/服务商**
你用自己的 LLM 批量生成输出后，放到某个目录（如 `./my_responses`）。

3. 喂回评测，得到成功率/缺失步骤/顺序错误/可供性错误等细粒度指标：

```bash
eai-eval --dataset virtualhome --eval-type action_sequencing \
         --mode evaluate_results --llm-response-path ./my_responses
```

评测会同时做“轨迹可执行性”和“目标满足度（含部分满足得分）”两类检查；文档里还展示了样例输入、输出与汇总指标的含义。

# 坑!

* **字段对齐**：如果你自己写评测脚本，注意把你的 LLM 输出格式与 `action_trajectory`/逻辑目标的字段对齐，方便自动评分（官方评测会解析这些字段并给出各类错误分解）。

# 一些说明
1. Goal Interpretation（目标解释）
   - 把自然语言指令落到环境里的符号目标：包含对象“状态”与对象间“关系”（可形成简化的 LTL 目标），用于后续规划。评分看状态/关系预测的 P/R/F1 与格式/幻觉错误。
2. Subgoal Decomposition（子目标分解）
   - 给定初始状态s_0与总目标g，产出按时间顺序的子目标序列（再由系统把子目标翻译成可执行动作，借模拟器检验）。评分分“轨迹可执行性”和“目标达成/部分达成”，并统计缺步/多步/顺序/可供性等错误。
3. Action Sequencing（动作排序）
   - 直接生成动作序列让模拟器执行，检查是否达成目标；同样统计轨迹执行与目标完成，以及细分错误类型。并计算部分成功（PartialSucc）。
4. Transition Modeling（转移建模）
   - 让模型写出PDDL 风格算子（operator）的前置条件与效果。评分既看逻辑匹配（前置/效果的 P/R/F1），也看用预测算子能否成功规划（planner success rate）。