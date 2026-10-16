# llm_code — 从零训练一个最小的可对话 LLM

一个**从零手写**的 decoder-only GPT 项目。目标不是做出一个能用的产品，而是**把 LLM 的完整流程亲手走一遍**，理解其中每一行。

代码总量约 **5692 行 Python**（`src/` 2916 + `scripts/` 1941 + `tests/` 547 + `_sandbox/`），不依赖 `transformers` 的模型实现——模型、训练循环、checkpoint、数据管线全部自己写。外部依赖只有 `torch` / `tokenizers` / `numpy` / `pyarrow`。

---

## 当前状态

| 里程碑 | 内容 | 状态 |
|---|---|---|
| **M0** | 环境（CUDA torch + sm_120 验证） | ✅ 完成 |
| **M1** | Tokenizer（BPE 训练 + 校验） | ✅ 完成 |
| **M2** | 模型 + 预训练管线（含中断恢复） | ✅ 完成 |
| **M3** | 中英混合 tokenizer + base 正式长训 | ✅ 完成 |
| **M4** | SFT（中英双语） | ✅ 完成 |
| **M4.5** | 领域自适应预训练（DAPT） | ✅ 完成 |
| M5 | 对话 CLI 打磨 + 自动评测 | ⏸ 部分 |

## 完整链路（已全线打通）

```
FineWeb-Edu 2GB + 中文维基 1.6GB
        ↓  词表 32k（中英混合）
        ↓  预训练 6.3 亿 token / 4.6 小时
   base_full   val 3.617
        ↓  DAPT：arXiv 21,119 篇摘要 / 5000 万 token / 20 分钟
   base_dapt   val 2.863
        ↓  SFT：2,145 条中英双语对话 / 3 epoch / 40 秒
   dapt_sft    val 3.717   ← 可以对话了
```

### 三个阶段各自的作用（实测对比）

| 阶段 | val loss | 说明 |
|---|---|---|
| 只有预训练 | — | 会续写，但不会回答问题 |
| 预训练 + SFT | 3.8398 | 会答了，但常答错 |
| **预训练 + DAPT + SFT** | **3.7168** | **答案开始有实质内容** |

DAPT 的价值最直观——它把模型的输出从"乱七八糟的网页文本"变成了**学术摘要的语域**：

> `Adversarial examples are distorted, so that no such adversaries can fool these classifiers. In this paper we propose a novel attack method named Reconnaissance-based Adversarial Training (RAT)...`

最终模型的实际问答：

```
Q: How do I evaluate robustness?
A: Measure clean accuracy under a fixed attack set and find the worst-case
   result from a different attacker who knows the perturbation.
```

### 两个诚实的限制

1. **中文明显弱于英文。** 中文侧预训练只有维基百科（正式书面语，无语料多样性），中文 SFT 只有 500 条问答。结果是中文能生成、但内容经常不通。
2. **31.46M 参数装不下知识。** 它学会了领域词汇和句式，但概念理解是有限的。想显著提升需要更大的模型或更多的领域文本。

---

## 快速开始

项目里有两个启动器，**用它们就不用记 venv 路径**：

| 文件 | 用法 |
|---|---|
| `run.cmd` | 通用启动器。切到项目目录、用项目 venv、设好编码，转发所有参数 |
| `status.cmd` | 只跑状态检查，末尾 `pause`，**可以直接双击** |

```powershell
# 1. 看全局状态（推荐第一步）—— 也可以直接双击 status.cmd
D:\py\llm_code\run.cmd scripts\status.py

# 2. 和最终模型对话（中英双语）
D:\py\llm_code\run.cmd -m src.generate --chat --ckpt runs/dapt_sft/best.pt --tokenizer data/tokenizer/base_tokenizer.json --system "You are a concise AI security tutor."


# 3. 交互式玩（:q 退出，:settings 看参数）
D:\py\llm_code\run.cmd -m src.generate --interactive

# 4. 核心自检（21 项）
D:\py\llm_code\run.cmd tests\test_core.py
```

用**完整路径**调用，在哪个目录都行。或者先 `cd D:\py\llm_code`，然后写 `.\run.cmd ...`（注意 `.\` 前缀，PowerShell 不会搜索当前目录）。

---

## 命令怎么读

拿最长的那条举例：

```
D:\py\llm_code\run.cmd   -m src.generate   --ckpt ... --tokenizer ... --prompt ... --temperature 0
└──── ① 程序 ────┘       └─ ② 模块 ─┘     └──────── ③ 传给脚本的参数 ────────┘
```

### ① 程序 `run.cmd`

批处理启动器，内部做四件事：

```bat
cd /d "%~dp0"                    :: 切到脚本所在目录（D:\py\llm_code）
set PYTHONIOENCODING=utf-8       :: 输出支持 UTF-8
set PYTHONPATH=%~dp0             :: 让 python 找得到 src 包
.venv\Scripts\python.exe %*      :: 用项目 venv，%* 原样转发所有参数
```

### ② `-m src.generate`

| 词 | 含义 |
|---|---|
| `-m` | Python 的开关，全称 **module**，"把后面这个名字当作模块运行" |
| `src.generate` | **模块名**（点分隔），不是文件名 |

```
src/generate.py   ← 文件路径（斜杠）
 src.generate     ← 模块名（点）      ← Python 只认这个
     └── 包名     └── 模块名
```

`src/` 下有 `__init__.py`，所以它是包。**为什么不用 `python src\generate.py`？** 因为 `generate.py` 开头是 `from src.config import ...`（绝对包导入），直接跑文件会被当成独立脚本、找不到 `src` 包。`-m` 会把当前目录加进 `sys.path`，导入才正常。

**分界线**：Python 自己的选项在模块名之前，模块名之后的**全部**归脚本解析。

### ③ 四个参数

| 参数 | 说明 |
|---|---|
| `--ckpt runs/tiny_smoke/best.pt` | 权重文件。里面**同时存了 `model_config`**，所以程序能重建出模型结构，不用你手写尺寸 |
| `--tokenizer data/tokenizer/tiny_tokenizer.json` | 词表。**必须和 ckpt 配套**——用错词表，输出直接是乱码 |
| `--prompt "Once upon a time"` | 开头文本。**引号必须加**，否则终端按空格切成 4 个参数 |
| `--temperature 0` | 0 = 贪心（每步选概率最高的，输出确定）；默认 0.8 = 采样 |

### 这条命令背后发生了什么

```
1. run.cmd 启动 → cd 到项目目录 → 用 .venv 的 python 执行
2. Python 加载 src/generate.py
3. 读词表文件   → 8192 个 token 及其 id
4. 读 ckpt     → 从 model_config 重建结构，载入 6.29M 权重
5. "Once upon a time" → 编码成 4 个 id: [438, 452, 264, 410]
6. 自回归生成：预测下一个 id → 追加 → 再预测 → ...
   （用 KV cache 加速，每步只算新 token，不重算整段）
7. 生成的 id → 解码回文本 → 打印
```

### 关于工作目录

`src/paths.py` 会把所有相对路径**锚定到项目根目录**，所以工作目录是什么都不影响：

```python
resolve("data/tokenizer/tiny_tokenizer.json")
# -> D:\py\llm_code\data\tokenizer\tiny_tokenizer.json
# 无论 cwd 是 C:\ 、src\ 还是项目根
```

绝对路径原样通过（`--ckpt D:\elsewhere\x.pt` 仍可用）。这让 `run.cmd`、PyCharm、裸 `python` 三种调用方式行为完全一致。

---

## 在 PyCharm 里运行

PyCharm 有三条路，从省事到正规：

### 方法 1：内置终端（什么都不用配）

PyCharm 底部的 **Terminal** 标签页（`Alt + F12`）就是真实终端：

```powershell
D:\py\llm_code\run.cmd scripts\status.py
D:\py\llm_code\run.cmd -m src.generate --interactive
```

`run.cmd` 自己会切目录、找 venv，所以零配置。

### 方法 2：配解释器 + Run Configuration

**① 让 PyCharm 认识 venv**

```
File → Settings → Project: llm_code → Python Interpreter
  → ⚙ → Add Local Interpreter... → Existing
  → D:\py\llm_code\.venv\Scripts\python.exe
```

**② 建 Run Configuration**

`Run → Edit Configurations... → + → Python`，关键三个字段：

| 字段 | 填什么 |
|---|---|
| Script path | `D:\py\llm_code\src\generate.py` |
| Parameters | `--ckpt runs/tiny_smoke/best.pt --tokenizer data/tokenizer/tiny_tokenizer.json --prompt "Once upon a time" --temperature 0` |
| Working directory | `D:\py\llm_code`（现在填错也不影响了，但建议填对） |

带子包导入的（如训练）改用 **Module name**：

| 字段 | 填什么 |
|---|---|
| Module name | `src.train_pretrain` |
| Parameters | 见「命令速查」的训练命令 |

**③ 现成的五个配置**

| 名字 | 类型 | 目标 | Parameters |
|---|---|---|---|
| `1-status` | Script | `scripts\status.py` | *(空)* |
| `2-generate` | Script | `src\generate.py` | `--ckpt runs/tiny_smoke/best.pt --temperature 0` |
| `3-interactive` | Script | `src\generate.py` | `--interactive` |
| `4-train` | Module | `src.train_pretrain` | `--preset tiny --out-dir runs/tiny_v2 --max-iters 3000 --micro-batch 32 --grad-accum 2` |
| `5-tests` | Script | `tests\test_core.py` | *(空)* |

### 三个 PyCharm 特有的坑

**坑 1：输出被缓冲，训练日志不刷新** ⚠️

每个 Run Configuration 里勾上 `☑ Emulate terminal in output console`。否则你会盯着空屏幕等很久，而且 `Ctrl+C` 的行为也不对（本项目的中断存盘逻辑依赖真实终端语义）。

**坑 2：中文乱码**

Run Configuration → Environment variables 加：

```
PYTHONIOENCODING=utf-8;PYTHONUTF8=1
```

（`run.cmd` 里就是设这些，PyCharm 不走 `run.cmd`，得自己填。）

**坑 3：索引 6.4 GB 数据会卡**

右键这些目录 → **Mark Directory as → Excluded**：

```
data/     runs/     _tmp/
```

`.venv/` 不要排除（排除了就没补全了）。

`.idea/` 已在 `.gitignore` 里，配置不会被提交。

---

## 数据流

### 预训练管线（已打通）

```
① 原始文本   data/raw/tinystories/TinyStories-train.txt              1853 MB
      │  scripts/download_tinystories.py            （断点续传下载）
      ▼
② 词表       data/tokenizer/tiny_tokenizer.json  (vocab 8192)         0.5 MB
      │  src/tokenizer.py + scripts/prepare_pretrain_data.py
      ▼
③ 训练数据   data/processed/tiny_train.bin   31,185,732 tokens       59.5 MB
             data/processed/tiny_val.bin            500,000 tokens     1.0 MB
      │  src/dataset.py                            （文档切分 + memmap）
      ▼
④ 训练       src/train_pretrain.py  ← 主循环
             src/model.py           ← 网络结构
             src/config.py          ← 尺寸与超参
             src/checkpoint.py      ← 原子存档 / 恢复
      ▼
⑤ 权重       runs/tiny_smoke/best.pt   6.29M 参数   val_loss 2.6214   24 MB
      │  src/generate.py                           （采样 / 交互）
      ▼
⑥ 通顺英文
```

### SFT 管线（已打通，中英双语）

```
src/template.py  ← 对话格式的唯一定义处
      │
      ├── scripts/synth_sft_data.py     （API 路线）
      └── 子代理 workflow               （免费路线，已产出 500 条）
      ▼
data/sft/*.jsonl   625 条对话
      │  scripts/validate_sft.py        （校验）
      ▼
（待写）src/train_sft.py
```

---

## 文件清单

### `src/` — 核心（2916 行）

| 文件 | 行数 | 职责 |
|---|---|---|
| `config.py` | 153 | 两档配置 + 参数量解析式估算（含宽深兑换、embedding 占比判据） |
| `paths.py` | 36 | **路径锚定**：所有相对路径按项目根解析，与工作目录无关 |
| `template.py` | 146 | **对话模板**。训练与推理唯一真源，防止字符串拼接不一致 |
| `tokenizer.py` | 233 | 字节级 BPE 的训练 / 保存 / 加载 / 校验 |
| `model.py` | 398 | **GPT 本体**：RMSNorm、RoPE、SwiGLU、QK-Norm、绑定权重、KV cache、采样 |
| `dataset.py` | 276 | 文档切分、编码成 bin、memmap 数据集、切片 |
| `checkpoint.py` | 210 | 原子写入、三种存档分工、轮转、恢复、RNG 保存 |
| `train_pretrain.py` | 498 | 预训练主循环 + `--init`（DAPT 用）。warmup+cosine、梯度累积、bf16、SIGINT 处理、周期采样 |
| `sft_data.py` | 215 | **SFT 数据管线**：对话 → (ids, loss mask)，含与模板一致性校验 |
| `train_sft.py` | 413 | SFT 主循环：掩码 loss、dropout、每轮打印真实问答样本 |
| `generate.py` | 287 | 采样 / 续写(`--interactive`) / **对话(`--chat`)** 三种模式 |
| `net.py` | 104 | 断点续传下载（两个下载脚本共用） |

### `scripts/` — 工具（1941 行）

| 文件 | 行数 | 职责 |
|---|---|---|
| `download_corpus.py` | 133 | 语料下载（注册表驱动：FineWeb-Edu / 中英维基 / Cosmopedia） |
| `download_arxiv.py` | 144 | **arXiv 摘要下载**——21,119 篇领域文本，DAPT 的语料来源 |
| `build_corpus.py` | 355 | parquet → 文本 → 词表 → 编码 → **按比例混编**（5 个子命令） |
| `download_tinystories.py` | 140 | 早期用的 TinyStories 下载器（保留） |
| `prepare_pretrain_data.py` | 220 | 单语料的 tokenizer + encode（`build_corpus` 的简化版） |
| `status.py` | 211 | 一键全局状态（环境/数据/词表/bin/SFT/训练记录/磁盘 + 建议下一步） |
| `synth_sft_data.py` | 545 | API 合成 SFT 数据（现在主要用子代理，此脚本保留为备份路线） |
| `validate_sft.py` | 193 | SFT 数据校验：JSON / 角色交替 / 长度 / 重复 / 模板往返 |

### `tests/` — 验证（547 行）

| 文件 | 内容 |
|---|---|
| `test_core.py` | 460 行，**30 项自检**，零额外依赖。环境 / 参数量核对 / 模板往返 / 前向与 loss / **KV cache 等价** / 采样控制 / **单 batch 过拟合** / embedding 梯度 / **SFT 掩码与模板一致性** |
| `test_cli.py` | 87 行，**CLI 冒烟测试**。因为同一个 argparse 错误犯了两次（共享选项放在顶层 parser 而非 parent parser，导致 `cmd sub --opt` 报"unrecognized arguments"），所以对每个入口点都测两种参数顺序 |

### 配置与环境

| 文件 | 说明 |
|---|---|
| `run.cmd` | **通用启动器**。切目录 + 用项目 venv + 设编码，转发所有参数 |
| `status.cmd` | **双击即可**的状态检查（末尾 `pause`，窗口不会闪退） |
| `.gitignore` | 保证 6.4 GB 数据与权重不进 git（递归规则，`data/` 下新目录自动覆盖） |
| `.env` | **你的 API key（已 gitignore）** |
| `.env.example` | API 配置模板，任何 OpenAI 兼容端点都能用 |
| `README.md` | 本文档 |
| `_sandbox/sitecustomize.py` | 沙箱适配补丁（见下文「环境坑」） |

### 数据与产物（均 gitignored）

```
data/raw/          1853 MB   原始语料
data/processed/     121 MB   编码后的 bin + meta
data/tokenizer/     0.5 MB   词表 + meta
data/sft/           0.3 MB   625 条对话（25 个 jsonl）
runs/tiny_smoke/    168 MB   best/final/last/step_*.pt + train_log.csv
.venv/             4407 MB   环境（其中 4210 MB 是 torch 的 CUDA 库）
_tmp/                 0 MB   TEMP 重定向目标
```

---

## 设计决策与理由

所有关键选择都写在了对应源文件的头部注释里。这里列最要的几条：

### 为什么是 384 宽 / 8 层，而不是 512 宽 / 6 层

```
每层参数 ≈ 16 · d_model²
    attention (q,k,v,o) = 4d²
    SwiGLU MLP          = 3·d·d_ff，取 d_ff=4d 得 12d²
```

**宽度是平方代价**：`d` 翻倍，层数只剩 1/4。而且 `d` 砍太狠时，`V·d` 的 embedding 会占掉总参数的 70%，模型变成"一张查表 + 一点点算力"。

参考现成的模型：**SmolLM2-135M 用 d=576 / 30 层**——比 GPT-2 small（768）更窄，但深得多。小模型要**深而窄**。

判定式：`embedding 占比 = V·d / (V·d + 16·L·d²) ≤ 45%`

| 档位 | V | d | L | H | d_ff | ctx | 参数量 | emb占比 |
|---|---|---|---|---|---|---|---|---|
| **tiny** | 8192 | 256 | 4 | 4 | 1024 | 256 | 6.29M | 33% |
| **base** | 32768 | 384 | 8 | 6 | 1536 | 1024 | 31.46M | 40% |

两者的 `head_dim` 都正好是 64。

### 为什么绑定权重（tie embeddings）

输入 embedding 和输出投影共用同一个 `[V, d]` 矩阵。省掉约 40% 的参数，且对小模型通常是正收益。

**副作用值得知道**：绑定后，"目标 token 和输入 token 相同时"loss 会明显低于 `ln(V)`（实测 7.04 vs 9.01），因为模型能从自己的 embedding 里"偷看"答案。所以测试里的初始 loss 检查必须用**错位目标**（预测 t+1）。

### 为什么不做 GQA / MoE

GQA 省的是推理时的 KV cache 显存。本项目 batch=1、ctx≤1024，cache 才几 MB。**不值得为此增加复杂度。**

### 为什么要自己训 tokenizer

用别人的词表（比如 Qwen 的 15 万）会让本项目 31M 的模型光 embedding 就 58M 参数——比整个模型还大。**词表大小必须和模型规模匹配。**

---

## 实测结果（tiny 档）

```
预设 tiny (6.29M)  |  device cuda  bf16  |  RTX 5060 Laptop
600 步 / 9,830,400 token / 62.6 秒 / ~160,000 tok/s

  step |   lr     | train    | val
  -----+----------+----------+---------
     1 | 3.00e-06 |   9.0633 |          ← 从 ln(8192)=9.01 起步 ✓
   100 | 3.00e-04 |   7.0287 |
   150 | 4.50e-04 |          |  3.8876
   200 | 6.00e-04 |   4.0487 |
   300 | 5.22e-04 |   3.2774 |  3.0118
   400 | 3.32e-04 |   2.9752 |  2.7263
   600 | 6.00e-05 |   2.7137 |  2.6214  ← best
```

**中断恢复实测**：在 step 600 停下，用 `--resume auto` 重启，正确从 600 续到 650，`best_val` 保留 2.6214，loss 与中断前连续无断层。

---

## 环境坑（换机器时会复现）

### 1. 沙箱下 `tempfile.mkdtemp()` 不可用

本机 DSH 沙箱的文件访问层**无法写入用 `0o700` 创建的目录**，而 `tempfile.mkdtemp()` 恰好用的就是 `0o700`：

```
os.mkdir(path, 0o700)  → 写不进去  ❌
os.mkdir(path)         → 正常      ✅
os.mkdir(path, 0o777)  → 正常      ✅
```

后果：`pip`、`ensurepip`、`huggingface_hub` **全部失败**。

解法：`_sandbox/sitecustomize.py` 重写 `mkdtemp`，去掉 mode 参数。venv 的 `Lib/site-packages/` 里也放了一份，所以用 venv python 时自动生效。

**顺带**：`python -m venv` 用 `-Im` 隔离模式跑 `ensurepip`，会**忽略 `PYTHONPATH`**，所以 shim 对它无效。本项目的 venv 是用 `--without-pip` 建、再手动把 pip 复制进去的。

### 2. HuggingFace 直连不通

`hosts` 里有 `127.0.0.1 huggingface.co`（加速器写的）。解决方案是走镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

实测可达：PyPI 官方 ✅ / PyTorch cu128 ✅ / hf-mirror ✅ / modelscope ✅ / huggingface.co 直连 ❌

### 3. venv 借用了 anaconda 的解释器

```
home = C:\Users\17944\anaconda3
include-system-site-packages = false
```

**解释器和标准库是借来的，第三方包完全隔离。**

- ⚠️ **别卸载/移动 anaconda3**，否则本项目 venv 立刻失效（和"Python 3.14 缺 tempfile.py 导致 `D:\py\.venv` 挂掉"是同一个机制）
- ✅ 附带好处：anaconda 里 MKL 版的 numpy + torch 自带的 OpenMP 会冲突（`libiomp5md.dll`），而 venv 里 pip 装的 numpy 走 OpenBLAS，**冲突自动消失**

---

## 开发中抓到的 bug（值得记住的教训）

### ① 参数量估算用错维度

QK-Norm 是**按 head** 归一化的，每组只有 `head_dim` 个权重，我误写成了 `d_model`。两个配置分别算多了 1,536 / 5,120 个参数。

**不报错、不影响训练，只会让你长期用错误的数字做架构决策**——最阴险的一类 bug。

### ② 生成会越界

上下文检查写在了 append 之后，序列能到 `ctx+1`。真跑长对话时会在 RoPE 缓存边界炸掉或静默错位。改成**喂入前检查**。

### ③ TinyStories 不是"一行一故事" ⚠️

文件结构实际是：

```
line 0: 'One day, a little girl named Lily found a needle...'
line 1: 'Lily went to her mom and said...'      ← 同一篇的第 2 行
line 2: 'Together, they shared the needle...'
line 3: '<|endoftext|>'                          ← 故事分隔符
```

按"行=文档"编码，等于**把故事切成句子碎片**：

| | 修之前 | 修之后 |
|---|---|---|
| 平均文档长度 | 35.7 token | **211.2 token** |
| 150k 文档产出 | 5.35M token | **31.7M token** |

**语料被低估 5.9 倍。** 现在 `--doc-separator` 可配置，且平均文档长度 < 50 token 时会**自动打警告**——这类错误必须让机器自己喊出来。

### ④ 编辑时弄丢了命令行参数

把 `--no-sample` 误删后没补回，训练跑到第 1 步就崩。**但错误处理器按预期工作，崩溃前自动存了 checkpoint** —— 中断安全机制在真实故障中验证通过。

### ⑤ `fh.read(n)` 数的是字符，不是字节

训练词表时设 `--sample-mb 200`，实际喂给 BPE 训练器的是 **706 MB**——因为中文一个字符占 2.5 个 UTF-8 字节，`read(n)` 按字符计数。结果训练跑 10 分钟不结束。

改成**二进制读 + 读完再解码**，采样量才准确。

### ⑥ 混编比例逻辑写错

我写了一段"clamping + rescaling"来处理某个语料不够的情况，但在 `scale = 1.0` 时它把目标值**直接覆盖成了"全部可用 token"**——产出 53.7:46.3 而不是要的 2:1。

改成最朴素的实现：按比例算目标、**clamp 到可用量、不做任何再分配**。如果某个源不够，诚实的做法是报告一个更小的总量，而不是悄悄改变语言比例。

### ⑦ Windows 上 memmap 会锁文件

`np.memmap` 还映射着 `train.bin` 时，又用 `"wb"` 打开它截断 → `OSError 22 (Invalid argument)`。

改成**写入前先决定 train/val 边界**，一次写完两个文件，全程不重开。

### ⑧ 恢复训练后吞吐虚高 3 倍

`tokens_done` 是从 step 0 累计的，但 `elapsed` 从恢复点算。两者相除，报出来的吞吐是整个运行除以本次运行——虚高约 3 倍。修成只算本次新增的 token。

### ⑨ 生成样本只写 stdout，不落文件

`--sample-interval` 的样本只打印到 stdout。用 `Get-Content -Wait` 盯 `train_log.csv` 的人**永远看不到**——而生成样本恰恰是长训练里最该看的东西。

现在同时写进 `runs/<name>/samples.log`。

### ⑩ 生成数据时文件命名撞车

第二次生成 prose 时用了同样的 `prose_01.txt ... prose_16.txt` 命名，**把第一批覆盖了**。多轮生成必须换前缀。

### ⑪ argparse 全局选项位置（犯了两次）

共享选项放在顶层 parser 而非 parent parser 时，`script.py <子命令> --选项 值` 会报 `unrecognized arguments`——而这正是所有人会打的写法。

修法是用 `parents=[common]`。并新增 `tests/test_cli.py` 对每个入口点测两种参数顺序，防止第三次。

---

## 已知限制

- **当前 tokenizer 是纯英文的。** 中文会退化成 UTF-8 字节编码（3.0 token/字，正常应 < 1）。中文必须等 M3 用混合语料重训词表。`status.py` 会主动警告。
- **模型是"续写器"，不是聊天机器人。** SFT 未做，问它问题它只会续写更多 TinyStories 风格的文本。
- **6.29M 参数只学会了"英文长什么样"**，没有知识、没有推理。这是规模的硬限制，不是调参能解决的。

---

## 未写的部分

| 文件 | 归属 | 作用 |
|---|---|---|
| `src/eval.py` | M5 | 自动评测：困惑度 + 固定 prompt 回归套件（目前靠人工看生成样本） |
| `src/chat.py` | M5 | 独立的对话 CLI（目前用 `src/generate.py --chat` 代替，够用） |

### 已知的改进方向（按收益排序）

1. **中文侧需要更多语料。** 现在中文预训练只有维基百科（语域单一），中文 SFT 只有 500 条问答。补中文口语语料 + 扩充中文问答是最直接的收益。
2. **领域文本可以更多。** arXiv 摘要给了 730 万 token，用来 DAPT 已经有效果；如果能拿到**论文正文**（而不仅是摘要），可以到上亿 token 量级。
3. **SFT 数据量。** 2145 条在 3 个 epoch 就好，说明数据量偏小、过拟合快。扩到 1 万条以上会更稳。
4. **模型规模。** 31.46M 是"能跑通"的下限；要真正理解领域概念，需要更大容量。

---

## 命令速查

所有命令都可以直接用完整路径调用，从任何目录都行：

```powershell
D:\py\llm_code\run.cmd <参数>
```

| 想看什么 | 命令 |
|---|---|
| 全局状态 | 双击 `status.cmd`，或 `run.cmd scripts\status.py` |
| 模型输出 | `run.cmd -m src.generate --ckpt runs/tiny_smoke/best.pt --temperature 0` |
| 换 prompt | 加 `--prompt "你的文本"`（可重复多次） |
| 采样多次 | `--num-samples 5 --temperature 1.0` |
| 交互模式 | `run.cmd -m src.generate --interactive` |
| 核心自检 | `run.cmd tests\test_core.py` |
| SFT 校验 | `run.cmd scripts\validate_sft.py data/sft/*.jsonl` |
| 续训 | 在训练命令后加 `--resume auto` |

```powershell
# 训练（tiny 档，约 1 分钟）
D:\py\llm_code\run.cmd -m src.train_pretrain --preset tiny --out-dir runs/tiny_v2 --train-bin data/processed/tiny_train.bin --val-bin data/processed/tiny_val.bin --tokenizer data/tokenizer/tiny_tokenizer.json --max-iters 3000 --micro-batch 32 --grad-accum 2 --eval-interval 300 --save-interval 500 --sample-interval 500
```

训练中途 `Ctrl+C` 一次 → 存盘后干净退出。再启动加 `--resume auto` 接着跑。

### 完整复现流程（从头到尾）

```powershell
# ① 下载语料（约 10 分钟，3.6 GB）
run.cmd scripts\download_corpus.py --source fineweb-edu-en --limit 1
run.cmd scripts\download_corpus.py --source wikipedia-zh

# ② parquet → 文本（2 分钟）
run.cmd scripts\build_corpus.py extract

# ③ 训 32k 中英混合词表（70 秒）
run.cmd scripts\build_corpus.py tokenizer --preset base --sample-mb 64

# ④ 编码（约 40 分钟）
run.cmd scripts\build_corpus.py encode --preset base --tokenizer data/tokenizer/base_tokenizer.json

# ⑤ 按 2:1 中英比例混编 → 6.3 亿 token（1 秒）
run.cmd scripts\build_corpus.py mix --preset base --target-tokens 630000000 --ratio 2:1 --tokenizer data/tokenizer/base_tokenizer.json

# ⑥ 预训练（约 4.6 小时）
run.cmd -m src.train_pretrain --preset base --out-dir runs/base_full --train-bin data/processed/train.bin --val-bin data/processed/val.bin --tokenizer data/tokenizer/base_tokenizer.json --max-iters 9600 --micro-batch 8 --grad-accum 8 --warmup-iters 300 --eval-interval 400 --eval-iters 50 --save-interval 800 --sample-interval 800

# ⑦ 下载领域文本并编码（6 分钟）
run.cmd scripts\download_arxiv.py --per-query 3000
run.cmd scripts\prepare_pretrain_data.py encode --tokenizer data/tokenizer/base_tokenizer.json --input data/domain/arxiv.txt --doc-separator newline --out-bin data/processed/arxiv_all.bin --out-train data/processed/arxiv_train.bin --out-val data/processed/arxiv_val.bin --val-tokens 100000

# ⑧ DAPT（约 20 分钟）——注意 --init 是"加载权重但重置步数和优化器"
run.cmd -m src.train_pretrain --preset base --init runs/base_full/best.pt --out-dir runs/base_dapt --train-bin data/processed/arxiv_train.bin --val-bin data/processed/arxiv_val.bin --tokenizer data/tokenizer/base_tokenizer.json --max-iters 768 --micro-batch 8 --grad-accum 8 --lr 1.5e-4 --warmup-iters 50 --eval-interval 96 --eval-iters 40 --save-interval 192 --sample-interval 192

# ⑨ SFT（40 秒）
run.cmd -m src.train_sft --init runs/base_dapt/best.pt --tokenizer data/tokenizer/base_tokenizer.json --data-glob "data/sft/*.jsonl" --out-dir runs/dapt_sft --epochs 3 --batch-size 16 --lr 1.5e-4

# ⑩ 对话
run.cmd -m src.generate --chat --ckpt runs/dapt_sft/best.pt --tokenizer data/tokenizer/base_tokenizer.json --system "You are a concise AI security tutor."
```

### 常见困惑

| 现象 | 原因 |
|---|---|
| 提示符是 `>>>` | 你进了 Python 交互解释器，输 `exit()` 退出 |
| `run.cmd : 无法将...识别为 cmdlet` | 要么路径不对（用完整路径），要么在当前目录但漏了 `.\` 前缀 |
| 引号里的内容被拆开 | `--prompt` 的值有空格，**必须加引号** |
| 中文输出乱码 | 先跑 `chcp 65001`；`run.cmd` 已设 `PYTHONIOENCODING=utf-8` |

---

## 环境信息

```
Python     3.13.9  (借自 C:\Users\17944\anaconda3)
torch      2.11.0+cu128
CUDA       12.8
GPU        NVIDIA GeForce RTX 5060 Laptop GPU (8 GB)
capability sm_120  (Blackwell) — arch_list 含 sm_120 ✅
tokenizers 0.23.3
numpy      2.5.4
```

重建环境：

```powershell
python -m venv --without-pip .venv
# 把 sitecustomize.py 放进 .venv\Lib\site-packages\
# 手动引导 pip 后：
pip install numpy tokenizers
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
