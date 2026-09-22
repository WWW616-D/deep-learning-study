# 杀戮尖塔 2（Slay the Spire 2）无色卡 & 特殊卡牌 研究报告（中文）

> 数据来源：slaythespire.wiki.gg 官方 Wiki（CARGO API `Cards` 表，均过滤 `Game="2"`，抓取于本次任务时点；游戏处于 Early Access 活跃开发期，内容可能变动）。
> 费用列格式：`基础/升级`（即 `(A;B)` = 基础 A、升级后 B；`/` 后为空表示升级不改变该值）；`(-2)` = 不可打出类卡牌（Unplayable）的特殊费用；`(-1)` = X 费用卡。
> **缺口与不确定性**：Wiki 部分页面标注"Under construction"；Buffs 页尾部（Machine Learning 之后）由 Defect 能力牌卡面数据重建，已注明；多处卡面图标在抓取中被剥离，涉及能量/星数值的条目已标注"图标缺失"。

---

## 1. 无色卡（Colorless Cards）

共 **150 张**（Game=2、Color=Colorless）：Ancient 9 / Rare 25 / Uncommon 40 / Event 28（含 9 张 Mad Science 变体）/ Quest 4 / Token 10 / Curse 18 / Status 16。**不存在 Basic / Common / Special 稀有度的无色卡**；且全游戏 Token/Quest/Event/Special 稀有度卡全部为无色（已用 `Color<>"Colorless"` 校验，结果为空）。

### 1.1 无色 · Ancient（远古，9 张）— 与 3 组"远古"牌获取渠道相关

| 英文名 | 类型 | 费用 | 效果（基础/升级） |
|---|---|---|---|
| Abundance 丰饶 | 技能 | 1/0 | 从 3 张**已升级**的能力牌（Power）中选择 1 张加入手牌，本回合免费打出。消耗 |
| Apotheosis 神化 | 技能 | 2/1 | 先制。升级**所有**卡牌。消耗 |
| Apparition 幽灵 | 技能 | 1 | （基础带虚影 Ethereal；升级移除虚影）获得 1 层无实体（Intangible）。消耗 |
| Brightest Flame 最亮火焰 | 技能 | 0 | 获得 2/3 点能量；抽 2/3 张牌；失去 2 点最大生命值 |
| Maul 重击 | 攻击 | 1 | 造成 5/6 点伤害两次；本场战斗中所有 Maul 的伤害提高 2/3 点 |
| Neow's Fury 涅奥之怒 | 攻击 | 1 | 造成 10/14 点伤害；将弃牌堆中最多 2/3 张牌放入手牌。消耗 |
| Relax 休憩 | 技能 | 3 | 获得 16/18 点格挡；下回合抽 2/3 张牌并获得 2/3 点能量。消耗 |
| Whistle 口哨 | 攻击 | 2 | 造成 33/44 点伤害；**眩晕（Stun）**敌人。消耗 |
| Wish 祈愿 | 技能 | 0 | 将抽牌堆中 1 张牌放入手牌（升级获得保留 Retain）。消耗 |

### 1.2 无色 · Rare（稀有，25 张）

| 英文名 | 类型 | 费用 | 效果（基础/升级） |
|---|---|---|---|
| Alchemize 炼金术 | 技能 | 1/0 | 获得一瓶随机药水。消耗 |
| Anointed 受膏 | 技能 | 1 | 将抽牌堆中**所有**稀有卡放入手牌（升级获得保留）。消耗 |
| Beacon of Hope 希望灯塔 | 能力 | 2 | 【多人】你回合内每获得格挡，其他玩家获得其一半的格挡（升级获得先制） |
| Beat Down 击倒 | 技能 | 3 | 从弃牌堆随机打出 3/4 张攻击牌 |
| Bolas 捕兽索 | 攻击 | 0 | 造成 3/4 点伤害；下回合开始时回到手牌 |
| Calamity 灾祸 | 能力 | 3/2 | 每当你打出攻击牌，将一张随机攻击牌加入手牌 |
| Entropy 熵 | 能力 | 1 | 回合开始时，转化手牌中 1 张牌（升级获得先制） |
| Eternal Armor 永恒护甲 | 能力 | 3 | 获得 9/12 层**镀层（Plating）** |
| Gold Axe 金斧 | 攻击 | 1 | 造成等同于本场战斗中已打出卡牌数量的伤害（升级获得保留） |
| Hand of Greed 贪婪之手 | 攻击 | 2 | 造成 20/25 点伤害；若**致命（Fatal）**，获得 20/25 金币 |
| Hidden Gem 隐藏宝石 | 技能 | 1 | 抽牌堆中 1 张没有**重放（Replay）**的随机卡牌获得 2/3 层重放 |
| Jackpot 头奖 | 攻击 | 3 | 造成 25/30 点伤害；将 3 张随机 0 费卡牌（升级后为升级版）加入手牌 |
| Knockdown 击倒 | 攻击 | 3 | 【多人】造成 10/14 点伤害；本回合敌人从其他玩家处受到的伤害变为 2/3 倍 |
| Master of Strategy 战略大师 | 技能 | 0 | 抽 3/4 张牌。消耗 |
| Mayhem 混乱 | 能力 | 2/1 | 回合开始时，自动打出抽牌堆顶的牌 |
| Mimic 模仿 | 技能 | 1 | 【多人】获得等同于另一名玩家当前格挡的格挡（升级：消耗） |
| Nostalgia 怀旧 | 能力 | 1/0 | 每回合你打出的第一张攻击或技能牌置于抽牌堆顶 |
| Rally 集结 | 技能 | 2 | 【多人】所有玩家获得 12/17 点格挡 |
| Rend 撕裂 | 攻击 | 1 | 造成 10/12 点伤害；敌人身上每有一种不同的减益，额外造成 5/8 点伤害 |
| Rolling Boulder 滚石 | 能力 | 3 | 回合开始时对所有敌人造成 5/10 点伤害，且该伤害每回合永久 +5 |
| Scrawl 涂鸦 | 技能 | 1 | 抽牌直到手牌全满。消耗（升级获得保留） |
| Secret Technique 秘技 | 技能 | 0 | 从抽牌堆选择 1 张技能牌放入手牌（升级：消耗） |
| Secret Weapon 秘武 | 技能 | 0 | 从抽牌堆选择 1 张攻击牌放入手牌（升级：消耗） |
| Splash 泼溅 | 技能 | 1 | 从 3 张随机**其他角色**的攻击牌中选 1 张加入手牌，本回合免费打出（升级后为升级版） |
| The Gambit 赌局 | 技能 | 0 | 获得 50/75 点格挡；若本场战斗中受到**未格挡的攻击伤害**，死亡 |

### 1.3 无色 · Uncommon（罕见，40 张）

| 英文名 | 类型 | 费用 | 效果（基础/升级） |
|---|---|---|---|
| Automation 自动化 | 能力 | 1/0 | 每抽 10 张牌，获得 1 点能量 |
| Believe in You 相信你 | 技能 | 0 | 【多人】另一名玩家获得 2/3 点能量 |
| Catastrophe 灾难 | 技能 | 2 | 从抽牌堆随机打出 2/3 张牌 |
| Coordinate 协同 | 技能 | 1 | 【多人】本回合给予另一名玩家 5/8 点力量 |
| Dark Shackles 黑暗枷锁 | 技能 | 0 | 本回合敌人失去 9/15 点力量。消耗 |
| Discovery 发现 | 技能 | 1 | 从 3 张随机卡牌中选择 1 张加入手牌，本回合免费打出（升级：消耗） |
| Dramatic Entrance 隆重登场 | 攻击 | 0 | 先制。对所有敌人造成 11/15 点伤害。消耗 |
| Equilibrium 平衡 | 技能 | 2 | 获得 13/16 点格挡；本回合保留手牌 |
| Fasten 加固 | 能力 | 1 | 防御（Defend）牌额外提供 4/6 点格挡 |
| Finesse 巧技 | 技能 | 0 | 获得 4/7 点格挡；抽 1 张牌 |
| Fisticuffs 拳斗 | 攻击 | 1 | 造成 7/9 点伤害；获得等同于造成伤害的格挡 |
| Flash of Steel 钢之闪光 | 攻击 | 0 | 造成 5/8 点伤害；抽 1 张牌 |
| Gang Up 群殴 | 攻击 | 1 | 【多人】造成 5 点伤害；本回合其他玩家每攻击该敌人一次，额外造成 5/7 点伤害 |
| Huddle Up 聚拢 | 技能 | 1 | 【多人】所有盟友抽 2/3 张牌。消耗 |
| Impatience 焦躁 | 技能 | 0 | 若手牌中没有攻击牌，抽 2/3 张牌 |
| Intercept 拦截 | 技能 | 1 | 【多人】获得 9/13 点格挡；本回合本应施加给其他玩家的所有攻击伤害改为由你承受 |
| Jack of All Trades 万事通 | 技能 | 0 | 将 1/2 张随机无色牌加入手牌。消耗 |
| Lift 抬举 | 技能 | 1 | 【多人】给予另一名玩家 11/16 点格挡 |
| Mind Blast 心灵冲击 | 攻击 | 1/0 | 先制。造成等同于抽牌堆中卡牌数量的伤害 |
| Omnislice 全切 | 攻击 | 0 | 造成 8/11 点伤害；对所有其他敌人造成等同于该伤害的伤害 |
| Panache 华丽 | 能力 | 0 | 每回合每打出 5 张牌，对所有敌人造成 10/14 点伤害 |
| Panic Button 恐慌按钮 | 技能 | 0 | 获得 30/40 点格挡；2 回合内你不能从卡牌获得格挡。消耗 |
| Prep Time 准备时间 | 能力 | 1 | 回合开始时获得 4/6 点**活力（Vigor）** |
| Production 量产 | 技能 | 0 | 获得 2/3 点能量。消耗 |
| Prolong 延长 | 技能 | 0 | 下回合开始时获得等同于你当前格挡的格挡（升级：消耗） |
| Prowess 武艺 | 能力 | 1 | 获得 1/2 点力量和 1/2 点敏捷 |
| Purity 净化 | 技能 | 0 | 保留。消耗手牌中最多 3/5 张牌。消耗 |
| Restlessness 躁动 | 技能 | 0 | 保留。若手牌为空，抽 2/3 张牌并获得 2/3 点能量 |
| Salvo 齐射 | 攻击 | 1 | 造成 12/16 点伤害；本回合保留手牌 |
| Seeker Strike 探索打击 | 攻击 | 1 | 造成 9/12 点伤害；从抽牌堆 3 张卡牌中选择 1 张加入手牌 |
| Shockwave 冲击波 | 技能 | 2 | 对所有敌人施加 3/5 层虚弱和 3/5 层易伤。消耗 |
| Stratagem 谋略 | 能力 | 1/0 | 每当你洗牌抽牌堆，从中选择 1 张牌放入手牌 |
| Tag Team 车轮战 | 攻击 | 2 | 【多人】造成 11/15 点伤害；另一名玩家对敌人打出的下一张攻击牌将额外打出一次 |
| The Ball 传球 | 攻击 | 1 | 【多人】造成 10 点伤害；本场战斗中此牌伤害 +10/15，并将其交给一名随机盟友 |
| The Bomb 炸弹 | 技能 | 2 | 3 回合后对所有敌人造成 40/50 点伤害 |
| Thinking Ahead 未雨绸缪 | 技能 | 0 | 抽 2 张牌；将手牌 1 张牌置于抽牌堆顶（升级：消耗） |
| Thrumming Hatchet 嗡鸣飞斧 | 攻击 | 1 | 造成 11/14 点伤害；下回合开始时回到手牌 |
| Ultimate Defend 终极防御 | 技能 | 1 | 获得 11/15 点格挡 |
| Ultimate Strike 终极打击 | 攻击 | 1 | 造成 14/20 点伤害 |
| Volley 齐发 | 攻击 | X（-1） | 对随机敌人造成 10/14 点伤害 X 次 |

### 1.4 无色 · Event（事件卡，28 条；含 9 张 Mad Science 变体）

| 英文名 | 类型 | 费用 | 效果（基础/升级） |
|---|---|---|---|
| Byrd Swoop 猛禽俯冲 | 攻击 | 0 | 造成 14/18 点伤害 |
| Caltrops 铁蒺藜 | 能力 | 1 | 每当你被攻击，反伤 3/5 点 |
| Clash 冲突 | 攻击 | 0 | 仅当手牌全部为攻击牌时可打出；造成 14/18 点伤害 |
| Distraction 干扰 | 技能 | 1/0 | 将 1 张随机技能牌加入手牌，本回合免费打出。消耗 |
| Dual Wield 双持 | 技能 | 1 | 选择 1 张攻击或能力牌，将 1/2 张复制加入手牌 |
| Enlightenment 顿悟 | 技能 | 0 | 本回合/本场战斗中，手牌中所有卡牌费用降为 1。消耗 |
| Entrench 掘壕 | 技能 | 2/1 | 你的格挡翻倍 |
| Exterminate 歼灭 | 攻击 | 1 | 对所有敌人造成 3/4 点伤害 4 次 |
| Feeding Frenzy 进食狂热 | 技能 | 0 | 本回合获得 5/7 点力量 |
| Hello World 你好世界 | 能力 | 1 | 回合开始时将 1 张随机普通（Common）卡加入手牌（升级获得先制） |
| Mad Science 疯狂科学 | 技能 | 1 | 可在事件「修补时间（Tinker Time）」中制作与自定义的卡牌（升级获得先制） |
| Mad Science (Chaos) | 技能 | 1 | 先制。获得 8 点格挡；将 1 张随机卡牌加入手牌，本回合免费打出 |
| Mad Science (Choking) | 攻击 | 1 | 先制。造成 12 点伤害；本回合每打出 1 张牌，敌人失去 6 点生命 |
| Mad Science (Curious) | 能力 | 1 | 先制。能力牌费用 -1 |
| Mad Science (Energized) | 技能 | 1 | 先制。获得 8 点格挡和 2 点能量 |
| Mad Science (Expertise) | 能力 | 1 | 先制。获得 2 点力量和 2 点敏捷 |
| Mad Science (Improvement) | 能力 | 1 | 先制。战斗结束时升级 1 张随机卡牌 |
| Mad Science (Sapping) | 攻击 | 1 | 先制。造成 12 点伤害；施加 2 层虚弱和 2 层易伤 |
| Mad Science (Violence) | 攻击 | 1 | 先制。造成 12 点伤害 3 次 |
| Mad Science (Wisdom) | 技能 | 1 | 先制。获得 8 点格挡；抽 3 张牌 |
| Metamorphosis 蜕变 | 技能 | 2 | 将 3/5 张随机攻击牌加入抽牌堆；本场战斗中免费打出。消耗 |
| Outmaneuver 迂回 | 技能 | 1 | 下回合获得 2/3 点能量 |
| Peck 啄击 | 攻击 | 1 | 造成 2 点伤害 3/4 次 |
| Rebound 回弹 | 攻击 | 1 | 造成 9/12 点伤害；本回合你打出的下一张牌置于抽牌堆顶 |
| Rip and Tear 撕裂 | 攻击 | 1 | 对随机敌人造成 7/9 点伤害两次 |
| Squash 压扁 | 攻击 | 1 | 造成 10/12 点伤害；施加 2/3 层易伤 |
| Stack 堆叠 | 技能 | 1 | 获得等同于弃牌堆中卡牌数量（+3）的格挡 |
| Toric Toughness 环面坚韧 | 技能 | 2 | 获得 5/7 点格挡；接下来 2 回合开始时各获得 5/7 点格挡 |

### 1.5 无色 · Quest（任务卡，4 张）与 Token（衍生物，10 张）

**Quest：**

| 英文名 | 费用 | 效果 |
|---|---|---|
| Byrdonis Egg 鸟蛋 | -2 | 不可打出。可在休息点孵化 |
| Dowsing 探水 | -2 | 不可打出。再进入 5 个「?」房间后转化为「丰饶（Abundance）」 |
| Lantern Key 灯笼钥匙 | -2 | 不可打出。解锁下一幕的特殊事件 |
| Spoils Map 战利品地图 | -2 | 不可打出。标记下一幕中一处 600 额外金币的地点 |

**Token：**

| 英文名 | 类型 | 费用 | 效果（基础/升级） |
|---|---|---|---|
| Fuel 燃料 | 技能 | 0 | 获得 1/2 点能量。消耗 |
| Giant Rock 巨石 | 攻击 | 1 | 造成 20/24 点伤害 |
| Luminesce 发光 | 技能 | 0 | 保留。获得 2/3 点能量。消耗 |
| Minion Dive Bomb 随从俯冲轰炸 | 攻击 | 0 | 造成 13/16 点伤害。消耗 |
| Minion Sacrifice 随从牺牲 | 技能 | 0 | 获得 7/10 点格挡。消耗 |
| Minion Strike 随从打击 | 攻击 | 0 | 造成 6/9 点伤害；抽 1 张牌。消耗 |
| Shiv 匕首 | 攻击 | 0 | 造成 4/6 点伤害。消耗 |
| Soul 灵魂 | 技能 | 0 | 抽 2/3 张牌。消耗 |
| Sovereign Blade 君主之刃 | 攻击 | 2/1 | 保留。造成 10 点伤害（Regent「锻造 Forge」机制创造） |
| Sweeping Gaze 扫视 | 攻击 | 0 | 虚影。Osty 对随机敌人造成 10/15 点伤害。消耗 |

### 1.6 通用泛用性标注（色外泛用/药水/支援/多人）

- **升级型（Apotheosis 上位泛用）**：Apotheosis（升级所有卡牌，Ancient）、Abundance（生成已升级能力牌）。
- **药水/资源型**：Alchemize（随机药水）、Automation（抽卡回能）、Production / Luminesce / Fuel / Believe in You（能量）、Master of Strategy / Scrawl / Thinking Ahead / Finesse / Flash of Steel / Impatience（抽牌）、Hand of Greed（金币）。
- **生存/防守型**：The Gambit、Panic Button、Eternal Armor（镀层）、Equilibrium（保留手牌）、Ultimate Defend、Purity。
- **随机生成/杂技型**：Discovery、Jack of All Trades、Splash、Mad Science 系、Calamity（攻击引擎）、Mayhem（自动打出）、The Bomb、Rolling Boulder、Panache。
- **多人（MultiplayerOnly=1）无色卡共 12 张**：Beacon of Hope、Believe in You、Coordinate、Gang Up、Huddle Up、Intercept、Knockdown、Lift、Mimic、Rally、Tag Team、The Ball。

---

## 2. 状态卡 / 诅咒卡 / 衍生物（Status / Curse / Token / Quest / Event）

（Token/Quest/Event 已并入第 1 节；此处仅列 Status 与 Curse。全部为 Colorless 色。）

### 2.1 状态卡 Status（16 张）

| 英文名 | 费用 | 效果 | 备注 |
|---|---|---|---|
| Beckon 呼唤 | 1 | 回合结束时若在手牌，失去 6 点生命 | **StS2 新状态** |
| Burn 灼烧 | -2 | 不可打出。回合结束时若在手牌，受到 2 点伤害 | 同 StS1 |
| Dazed 眩晕 | -2 | 不可打出。虚影 | 同 StS1 |
| Debris 残骸 | 1 | 消耗 | **StS2 新状态**（可打出，效果即消耗自身） |
| Disintegration 解体 | -2 | 回合结束时受到 6/7/8 点伤害 | **StS2 新状态**；伤害为 6/7/8 三档（疑按章节递增，Wiki 原文如此） |
| Frantic Escape 仓皇逃窜 | 1 | 离得更远：使「The Insatiable（饕餮）」的**沙坑（Sandpit）**+1；此牌费用 +1 | **StS2 新状态**，Boss 专属 |
| Infection 感染 | -2 | 不可打出。回合结束时若在手牌，受到 3 点伤害 | **StS2 新状态** |
| Mind Rot 心智腐化 | -2 | 每回合少抽 1 张牌 | 同 StS1（敌人「Knowledge Demon 知识恶魔」同名减益） |
| Slimed 粘液 | 1 | 抽 1 张牌。消耗 | 同 StS1 |
| Sloth 怠惰 | -2 | 每回合最多打出 3 张牌 | **StS2 新状态**（Normality 的状态版；敌人同名减益） |
| Soot 煤烟 | -2 | 不可打出 | **StS2 新状态** |
| Toxic 剧毒 | 1 | 回合结束时若在手牌，受到 5 点伤害。消耗 | **StS2 新状态** |
| Void 虚空 | -2 | 不可打出。虚影。抽到此牌时失去 1 点能量 | 相比 StS1 新增**虚影**词条 |
| Waste Away 衰败 | -2 | 每回合少获得 1 点能量 | **StS2 新状态**（敌人同名减益） |
| Wither 枯萎 | -2 | 不可打出。回合结束时若在手牌，受到 3/6 点伤害 | **StS2 新状态**；**有升级版本**（3→6） |
| Wound 伤口 | -2 | 不可打出 | 相比 StS1 无"治疗时移除"相关词条 |

### 2.2 诅咒卡 Curse（18 张）

| 英文名 | 费用 | 效果 | 备注 |
|---|---|---|---|
| Ascender's Bane 攀登者之灾 | -2 | 不可打出。虚影。**永恒（Eternal）** | 尖塔 10 层起自带，同 StS1 |
| Bad Luck 霉运 | -2 | 不可打出。回合结束时若在手牌，失去 13 点生命。永恒 | **StS2 新诅咒** |
| Clumsy 笨拙 | -2 | 不可打出。虚影 | 同 StS1 |
| Curse of the Bell 钟之诅咒 | -2 | 不可打出。永恒 | 同 StS1（钟铃遗物） |
| Debt 债务 | -2 | 不可打出。回合结束时若在手牌，失去 10 金币 | **StS2 新诅咒** |
| Decay 腐朽 | -2 | 不可打出。回合结束时若在手牌，受到 2 点伤害 | 同 StS1 |
| Doubt 疑虑 | -2 | 不可打出。回合结束时若在手牌，获得 1 层虚弱 | 同 StS1 |
| Enthralled 魅惑 | 2 | 若在手牌中，必须先于其他牌打出。永恒 | **StS2 新诅咒**（有费用、可打出！） |
| Folly 愚行 | -2 | 不可打出。虚影。先制。永恒 | 同 StS1 |
| Greed 贪婪 | -2 | 不可打出。永恒 | 同 StS1 |
| Guilty 内疚 | -2 | 不可打出。5 场战斗后从牌组移除 | **StS2 新诅咒**（自动消失） |
| Injury 创伤 | -2 | 不可打出 | 同 StS1 |
| Normality 常态 | -2 | 不可打出。本回合最多打出 3 张牌 | 同 StS1 |
| Poor Sleep 睡眠不佳 | -2 | 不可打出。**保留（Retain）** | **StS2 新诅咒** |
| Regret 悔恨 | -2 | 不可打出。回合结束时若在手牌，每有 1 张手牌失去 1 点生命 | 同 StS1 |
| Shame 羞愧 | -2 | 不可打出。回合结束时若在手牌，获得 1 层脆弱 | 同 StS1 |
| Spore Mind 孢子心智 | 1 | 消耗 | **StS2 新诅咒**（有费用、可打出） |
| Writhe 扭动 | -2 | 不可打出。先制 | 同 StS1 |

> StS1 的 Pain / Parasite / Pride / Necronomicurse / Icky 等诅咒**未**出现在 StS2 当前卡库中（可能未实现或随事件调整，Wiki 未收录）。

---

## 3. Ancient 稀有度（远古卡，全角色共 19 张）

| 英文名 | 角色 | 类型 | 费用（含星费） | 效果（基础/升级） |
|---|---|---|---|---|
| Break 破势 | Ironclad | 攻击 | 1 | 造成 20/30 点伤害；施加 5/7 层易伤 |
| Corruption 腐化 | Ironclad | 能力 | 3/2 | 技能牌费用为 0；每当你打出技能牌，消耗之 |
| Suppress 压制 | Silent | 攻击 | 0 | 先制。造成 11/17 点伤害；施加 3/5 层虚弱 |
| Wraith Form 幽灵形态 | Silent | 能力 | 3 | 获得 2/3 层无实体（Intangible）；回合开始时失去 1 点敏捷 |
| Meteor Shower 流星雨 | Regent | 攻击 | 0 能量 + **2 星** | 对所有敌人造成 14/21 点伤害；对所有敌人施加 2 层虚弱和 2 层易伤 |
| The Sealed Throne 封印王座 | Regent | 能力 | 1 能量 + **3 星** /0 | 每当你打出一张牌，获得 1 颗星（★） |
| Forbidden Grimoire 禁忌魔典 | Necrobinder | 能力 | 2/1 | 战斗结束时，可从牌组移除 1 张卡牌。永恒 |
| Protector 守护者 | Necrobinder | 攻击 | 1/0 | Osty 造成 10/15 点伤害；额外造成等同于 Osty 最大生命值的伤害 |
| Biased Cognition 偏执认知 | Defect | 能力 | 1 | 获得 5/6 点聚焦（Focus）；回合开始时失去 1 点聚焦 |
| Quadcast 四重激发 | Defect | 技能 | 1/0 | 激发（Evoke）最右侧球体 4 次 |
| Abundance 丰饶 | 无色 | 技能 | 1/0 | 见 1.1 |
| Apotheosis 神化 | 无色 | 技能 | 2/1 | 见 1.1 |
| Apparition 幽灵 | 无色 | 技能 | 1 | 见 1.1 |
| Brightest Flame 最亮火焰 | 无色 | 技能 | 0 | 见 1.1 |
| Maul 重击 | 无色 | 攻击 | 1 | 见 1.1 |
| Neow's Fury 涅奥之怒 | 无色 | 攻击 | 1 | 见 1.1 |
| Relax 休憩 | 无色 | 技能 | 3 | 见 1.1 |
| Whistle 口哨 | 无色 | 攻击 | 2 | 见 1.1 |
| Wish 祈愿 | 无色 | 技能 | 0 | 见 1.1 |

> 说明：Ancient（远古）为 StS2 新引入的卡牌稀有度层级，通过远古事件/遗物/首领等渠道获得；其中 Regent 的两张（Meteor Shower、The Sealed Throne）消耗**星（Stars）**资源而非普通能量。

---

## 4. 关键词表（Keywords / Buffs / Debuffs / Enchantments 全表）

### 4.1 卡牌关键词 Keywords（Wiki「Slay the Spire 2:Keywords」全量）

**通用：**

| 关键词 | 中文 | 说明 | 相对 StS1 |
|---|---|---|---|
| Block | 格挡 | 格挡持续到下一回合，先于生命承受伤害 | 沿用 |
| Energy | 能量 | 每回合开始获得 3 点能量；每个角色有专属图标（Ironclad/Silent/Regent/Necrobinder/Defect/无色），**Regent 另有"星（Stars）"作为替代能量** | 沿用（星为新增） |

**卡牌关键词：**

| 关键词 | 中文 | 说明 | 相对 StS1 |
|---|---|---|---|
| Eternal | 永恒 | 此牌**无法**从牌组移除或转化（战斗中仍可被 BEGONE!/Entropy 临时转化；Thieving Hopper 可偷走） | **新关键词**（取代 StS1 的"无法移除"描述） |
| Exhaust | 消耗 | 打出后移出牌组直到战斗结束，进入消耗堆 | 沿用 |
| Ethereal | 虚影 | 回合结束时若在手牌，自动消耗 | 沿用 |
| Fatal | 致命 | 击杀**非随从（Minion）**敌人时触发 | 沿用 |
| Innate | 先制 | 首回合必在起手牌中（占用首抽名额；先制牌超过首抽数时全部入手） | 沿用 |
| Retain | 保留 | 回合结束不弃掉，可无限保留在手 | 沿用 |
| Replay | 重放 | 此牌被连续额外打出一次；多层叠加 | **新关键词**（来源：Glam/Spiral 附魔、Soldier's Stew、Transfigure、Sword Sage、Hidden Gem） |
| Unplayable | 不可打出 | 无法打出、无能量费用；被强制打出时改为直接进弃牌堆（不算打出） | 沿用 |

**角色专属：**

| 关键词 | 角色 | 中文 | 说明 | 相对 StS1 |
|---|---|---|---|---|
| Poison | Silent | 中毒 | 回合开始时失去 X 生命，随后中毒 -1 | 沿用 |
| Sly | Silent | 狡诈 | 若此牌在**你的回合内**被弃掉，立即免费打出 | **新关键词** |
| Stars | Regent | 星 | Regent 打出部分卡牌需要消耗的替代资源 | **新关键词**（卡库字段 StarCost） |
| Forge | Regent | 锻造 | 每场战斗首次锻造在手牌生成「君主之刃（Sovereign Blade）」；每次锻造 X 为所有区域（手牌/抽牌堆/弃牌堆/消耗堆）的君主之刃 +X 伤害；若场上无君主之刃则新建 | **新关键词** |
| Doom | Necrobinder | 厄运 | 回合结束时若厄运 ≥ 生命值则死亡；无视格挡、Slippery、无实体；可对自己施加（Neurosurge） | **新关键词** |
| Summon | Necrobinder | 召唤 | 召唤 X 即召唤 Osty（最大生命 X）；Osty 存活时改为提高其最大生命 X | **新关键词** |
| Channel | Defect | 充能 | 将球体放入最左侧空槽；无空槽时自动激发最右侧球体 | 沿用 |
| Evoke | Defect | 激发 | 消耗球体并触发其爆发效果；多球同时激发时从右到左 | 沿用 |
| Focus | Defect | 聚焦 | 提高球体效果 X；等离子球不受聚焦影响 | 沿用 |
| Lightning Orb | Defect | 闪电球 | 被动：对随机敌人造成 3 伤害；激发：造成 8 伤害 | 沿用 |
| Frost Orb | Defect | 冰霜球 | 被动：获得 2 格挡；激发：获得 5 格挡 | 沿用 |
| Dark Orb | Defect | 暗黑球 | 被动：储存伤害 +6；激发：对生命最低的敌人造成储存伤害 | 沿用 |
| Plasma Orb | Defect | 等离子球 | 被动：下回合开始多 1 能量；激发：获得 2 能量；不受聚焦影响 | 沿用 |
| Glass Orb | Defect | 玻璃球 | 被动：对所有敌人造成 4 伤害，随后该伤害 -1；激发：对所有敌人造成 2 倍被动伤害 | **新球体** |
| Enchant | — | 附魔 | 给牌组中的卡牌添加一个永久正面效果（详见 4.4） | **新机制** |
| Stunned | — | 眩晕 | 被 Whistle 等效果施加；被眩晕的敌人无法行动（关键词页无独立条目，由卡面/减益引用） | **新机制**（StS1 无玩家可用眩晕） |

### 4.2 增益 Buffs（Wiki「Slay the Spire 2:Buffs」全表，~110 项）

> Wiki 标注 Under construction。图标在抓取中缺失的数值以「☐」标注。**「Machine Learning」之后的 Defect 段由 Defect 能力牌卡面数据重建（标注 *）**。

**通用/药水系：**

| 英文名 | 中文 | 效果 |
|---|---|---|
| Strength | 力量 | 攻击伤害 +X |
| Dexterity | 敏捷 | 卡牌获得的格挡 +X |
| Artifact | 人工制品 | 抵消接下来 X 个减益 |
| Block Next Turn | 下回合格挡 | 下回合开始时获得 X 格挡（受敏捷/脆弱修正） |
| Blur | 模糊 | 接下来 X 回合开始时格挡不清零 |
| Draw Cards Next Turn | 下回合抽牌 | 下回合开始多抽 X 张 |
| Energy Next Turn | 下回合能量 | 下回合多获得 X 能量 |
| Focus | 聚焦 | 球体效果 +X |
| Intangible | 无实体 | X 回合内受到的伤害与生命损失降为 1（敌人回合结束移除） |
| Plating | 镀层 | 回合结束时获得 X 格挡；回合开始时镀层 -1 |
| Retain Hand | 保留手牌 | 接下来 X 回合保留手牌 |
| Thorns | 荆棘 | 被攻击时反伤 X |
| Vigor | 活力 | 下一次攻击额外造成 X 伤害 |
| Buffer | 缓冲 | 抵消接下来 X 次生命损失 |
| Clarity | 澄明 | 接下来 X 回合开始各多抽 1 张 |
| Duplication | 复制 | 接下来 X 张牌额外打出一次 |
| Flex Potion | 灵活药水 | 本回合获得 X 力量（临时力量） |
| Gigantification | 巨大化 | 接下来 X 张攻击牌造成三倍伤害 |
| Radiance | 光辉 | 接下来 X 回合额外获得 ☐（图标缺失，疑为能量） |
| Regen | 再生 | 回合结束时治疗 X 并 -1 |
| Reptile Trinket | 爬虫饰品 | 使用药水时：本回合获得 X 力量（临时力量） |
| Ritual | 仪式 | 回合结束时获得 X 力量 |
| Speed Potion | 速度药水 | 本回合获得 X 敏捷（临时敏捷） |

**Ironclad 系：** Aggression 侵略（回合开始将 X 张随机攻击从弃牌堆入手并永久升级）、Barricade 壁垒（格挡不清零）、Colossus 巨像（X 回合内受易伤敌人伤害 -50%）、Corruption 腐化（技能 0 费且消耗）、Crimson Mantle 猩红斗篷（回合开始失去 X 生命获得 X 格挡，自伤递增）、Cruelty 残暴（易伤敌人受 X% 额外伤害）、Dark Embrace 黑暗拥抱（每消耗一张牌抽 X）、Demon Form 恶魔形态（回合开始 +X 力量）、Feel No Pain 无痛觉（每消耗一张牌 +X 格挡）、Flame Barrier 火焰屏障（本回合被攻击反伤 X）、Free Attack（Unrelenting：接下来 X 张攻击 0 费）、Guarded 被守护（受坦克保护，伤害减半；坦克死亡移除）、Hellraiser 地狱狂欢（抽到含"Strike"的牌时自动打出）、Inferno 地狱火（回合开始失去 X 生命；你回合内每失去生命对所有敌人造成 X 伤害，自伤递增）、Juggernaut 攻城锤（获得格挡时对随机敌人造成 X 伤害）、Juggling 抛接（每回合第 3 张打出的攻击牌复制 X 张入手）、One-Two Punch 连击（本回合接下来 X 张攻击额外打出一次）、Pyre 火葬（回合开始获得 ☐——图标缺失，疑为能量）、Rage 怒火（本回合打出攻击时 +X 格挡）、Rupture 破裂（你回合内失去生命时 +X 力量）、Self-Forming Clay 自塑黏土（下回合获得 X 格挡；当格挡被打空时触发）、Setup Strike 预备打击（本回合获得 X 力量）、Stampede 踩踏（回合结束时随机攻击牌自动打出）、Tank 坦克（自身受伤翻倍，盟友受伤减半）、Unmovable 不移（每回合前 X 次卡牌格挡翻倍）、Vicious 凶残（施加易伤时抽 X 张）。

**Silent 系：** Accelerant 助燃（中毒多触发 X 次）、Accuracy 精准（匕首 +X 伤害）、Afterimage 残影（打出牌时 +X 格挡）、Anticipate 预判（本回合 +X 敏捷）、Burst 爆裂（接下来 X 张技能额外打出一次）、Concoct 调配（造成未格挡攻击伤害时施加 X 中毒）、Corrosive Wave 腐蚀波（本回合抽牌时对全体施加 X 中毒）、Double Damage 双倍伤害（接下来 X 回合攻击翻倍）、Envenom 淬毒（未格挡攻击伤害施加 X 中毒）、Fade 消隐（回合结束时失去 X 敏捷）、Fan of Knives 飞刀（匕首攻击全体）、Free Skill（Pounce：接下来 X 张技能 0 费）、Helical Dart 螺旋镖（打出匕首时本回合 +X 敏捷）、Infinite Blades 无尽刀刃（回合开始 +X 匕首入手）、Master Planner 总策划（打出的技能获得狡诈 Sly）、Nightmare 噩梦（下回合将选定卡牌复制 X 张入手）、Noxious Fumes 恶毒烟雾（回合开始对全体施加 X 中毒）、Outbreak 爆发（施加中毒时对全体造成 X 伤害）、Phantom Blades 幻影刀刃（匕首获得保留；首张匕首 +X 伤害）、Serpent Form 蛇形（打出牌时对随机敌人造成 X 伤害）、Shadow Step 影步（接下来 X 回合攻击翻倍）、Shadowmeld 影遁（本回合格挡获得翻倍 X 次，乘法叠加 2^X）、Sneaky 鬼祟（【多人】其他玩家攻击时你 +X 格挡）、Speedster 极速者（你回合内抽牌时对全体造成 X 伤害，仅限常规起手抽牌之外）、The Hunt 狩猎（战斗结束额外 X 张卡牌奖励）、Tools of the Trade 行会工具（回合开始抽 X 弃 X）、Tracking 追踪（虚弱敌人受攻击伤害 X 倍）、Well-Laid Plans 运筹帷幄（回合结束不再弃牌）。

**Regent 系：** Arsenal 兵工厂（打出无色牌时 +X 力量）、Black Hole 黑洞（每花费或获得 1 ☐ 对全体造成 X 伤害——图标缺失，疑为星）、Child of the Stars 星辰之子（每花费 1 ☐ 获得 X 格挡——疑为星）、Foregone Conclusion 命中注定（下回合将 X 张抽牌堆牌入手）、Furnace 熔炉（回合开始锻造 X 次）、Genesis 创世（回合开始获得 ☐——疑为星）、Hammer Time 锤击时刻（你锻造时全体盟友一起锻造）、Monarch's Gaze 君主凝视（攻击敌人时其本回合 -X 力量）、Monologue 独白（本回合打出牌时 +1 力量，回合结束移除）、Orbit 轨道（每花费 4 ☐ 获得 X ☐——图标缺失）、Pale Blue Dot 暗淡蓝点（每回合打出 ≥5 张牌时，下回合开始多抽 X）、Parry 招架（君主之刃获得 X 格挡）、Pillar of Creation 创世之柱（每回合首次生成卡牌时 +X 格挡）、Reflect 反射（X 回合内格挡掉的伤害反射给攻击者）、Royalties 版税（战斗结束时获得 X 金币）、Seeking Edge 追猎之刃（君主之刃攻击全体）、Spectrum Shift 光谱偏移（回合开始将 X 张随机无色牌入手）、Star Next Turn 下回合星（下回合获得 X 颗星）、Sword Sage 剑圣（君主之刃获得 X 层重放）、The Sealed Throne 封印王座（打出牌时获得 1 颗星）、Tyranny 暴政（回合开始抽 X 张并消耗 X 张手牌）、Void Form 虚空形态（每回合前 X 张牌免费打出）。

**Necrobinder 系：** Cacophony 噪音（全体玩家累计抽 Y 张后对随机敌人造成 X 伤害）、Calcify 石化（Osty 攻击 +X 伤害）、Call of the Void 虚空呼唤（回合开始将 X 张随机牌入手并施加虚影）、Countdown 倒计时（回合开始对随机敌人施加 X 厄运）、Danse Macabre 死亡之舞（打出 ≥2 费牌时 +X 格挡）、Demesne 领地（回合开始获得 ☐ 并多抽 X——疑为能量）、Devour Life 吞噬生命（打出灵魂时召唤 X）、Die for You 为你而死（Osty 吸收所有未格挡攻击伤害）、Forbidden Grimoire 禁忌魔典（战斗结束移除 X 张牌）、Friendship 友谊（回合开始获得 ☐——疑为能量/灵魂）、Haunt 作祟（打出灵魂时随机敌人失去 X 生命）、Lethality 致命（每回合首张攻击 +X% 伤害）、Necro Mastery 死灵掌握（Osty 失去生命时全体敌人失去 X 倍等量生命）、Pagestorm 书页风暴（抽到虚影牌时抽 X 张）、Reaper Form 收割形态（攻击造成伤害时施加等量 X 倍厄运）、Sentry Mode 哨兵模式（回合开始将 X 张「扫视」入手）、Shroud 帷幕（施加厄运时 +X 格挡）、Sleight of Flesh 血肉戏法（对敌人施加减益时其受到 X 伤害，临时减益不触发）、Soulbound 灵魂绑定（【多人】玩家生成灵魂时向你的抽牌堆加入 X 张灵魂）、Spirit of Ash 灰烬之灵（打出虚影牌时 +X 格挡）、Summon Next Turn 下回合召唤（下回合开始召唤 X）、Underworld 冥界（【多人】本回合其他玩家造成攻击伤害时施加等量厄运）、Veilpiercer 破帷者（接下来 X 张虚影牌 0 费）。

**Defect 系：** Consuming Shadow 吞噬暗影（回合结束激发最左球 X 次）、Coolant 冷却剂（回合开始每种不同球体 +X 格挡）、Creative AI 创造型 AI（回合开始将 X 张随机能力牌入手）、Echo Form 回响形态（每回合前 X 张牌额外打出一次）、Feral 野性（每回合首次打出 0 费攻击后回到手牌）、Focused Strike 聚焦打击（本回合 +X 聚焦）、Free Power（Synthesis：接下来 X 张能力牌 0 费）、Hailstorm 冰雹（回合结束若有冰霜球，对全体造成 X 伤害）、Hello World 你好世界（回合开始将 X 张随机普通卡入手）、Hibernate 休眠（本回合你的冰霜球为全体盟友提供格挡）、Hotfix 热修复（本回合 +X 聚焦）、Imitation Learning 模仿学习（【多人】玩家打出能力牌时你打出其复制，Wiki 标注需核实）、Iteration 迭代（每回合首次抽到状态卡时抽 X 张）、Lightning Rod 避雷针（接下来 X 回合开始充能 1 闪电球）、Loop 循环（回合开始触发最右球被动 X 次）、Machine Learning 机器学习（回合开始多抽 1 张）、One for All* 以一敌众（【多人】所有人 0 费攻击 +X 伤害）、Smokestack* 烟囱（生成状态卡时对全体造成 X 伤害）、Spinner* 纺丝者（回合开始充能 1 玻璃球；升级：额外立即充能 1）、Storm* 风暴（打出能力牌时充能 1/2 闪电球）、Subroutine* 子程序（打出能力牌时获得 1 能量）、Thunder* 雷霆（激发闪电球时对每个被击中的敌人造成 X 伤害）、Trash to Treasure* 变废为宝（生成状态卡时充能 1 随机球）。

> 另被引用但未单列于 Buffs 表：**Minion（随从）**（Fatal 关键词：击杀随从不触发致命效果）。

### 4.3 减益 Debuffs（Wiki「Slay the Spire 2:Debuffs」全表）

**通用减益（玩家可施加）：**

| 英文名 | 中文 | 效果 |
|---|---|---|
| Vulnerable | 易伤 | X 回合内受到的攻击伤害 +50% |
| Weak | 虚弱 | X 回合内攻击伤害 -25% |
| Frail | 脆弱 | X 回合内卡牌格挡 -25% |
| Strength (Debuff) | 力量（减益） | 攻击伤害 -X |
| Dexterity (Debuff) | 敏捷（减益） | 卡牌格挡 -X |
| Poison | 中毒 | 回合开始失去 X 生命，随后 -1 |
| Doom | 厄运 | 敌人回合结束时若生命 ≤ X 则死亡 |
| Focus (Debuff) | 聚焦（减益） | 球体效果 -X |
| Confused | 困惑 | 抽牌时费用随机 0–3（Snecko Eye） |
| Demise | 灭亡 | 回合结束时失去 X 生命（药水） |
| Shackling Potion / Mangle / Piercing Wail / Crush Under / Dying Star / Enfeebling Touch / Monarch's Gaze Strength Down / Dark Shackles | 临时力量损失系 | 本回合失去 X 力量；多来源机制相同（临时力量减益） |
| Shrink | 缩小 | 攻击伤害 -30%；施加者死亡后移除（敌方）／3 回合后移除（药水） |
| No Energy Gain | 无法获得能量 | 本回合无法获得额外能量（Expect a Fight） |
| No Draw | 无法抽牌 | 本回合不能再抽牌（战斗狂怒/子弹时间） |
| Strangle | 绞杀 | 本回合每打出 1 张牌，该敌人失去 X 生命 |
| Wraith Form | 幽灵形态 | 回合开始失去 X 敏捷（自伤） |
| Borrowed Time | 借来的时间 | 本回合卡牌费用 +X（自伤，Necrobinder） |
| Debilitate | 衰弱 | 接下来 X 回合虚弱与易伤效果翻倍 |
| Hang | 悬吊 | 所有「Hang」牌对该敌人造成 X 倍伤害（乘法叠加，封顶 999） |
| Neurosurge | 神经手术 | 回合开始对自己施加 X 厄运（自伤） |
| Oblivion | 湮灭 | 每打出 1 张牌，该敌人获得 X 厄运 |
| Sic 'Em | 咬它 | 本回合 Osty 命中该敌人时召唤 X |
| Biased Cognition | 偏执认知 | 回合开始失去 X 聚焦（自伤） |
| Knockdown | 击倒 | 【多人】本回合其他盟友对该敌人伤害 X 倍 |
| No Block | 无法格挡 | X 回合内不能从卡牌获得格挡（恐慌按钮） |
| Tag Team | 车轮战 | 【多人】其他玩家对该敌人打出的下一张攻击额外打出 X 次 |
| The Gambit | 赌局 | 本场战斗受到未格挡攻击伤害则死亡 |
| Flanking | 侧翼夹击 | 【多人】本回合其他盟友对该敌人伤害 X 倍（乘法叠加：2 层 = 4 倍） |
| Conqueror | 征服者 | 接下来 X 回合君主之刃对其造成双倍伤害 |
| Demise/Shrink/… | （同上） | — |

**敌方专属减益（来源为敌人）：**

| 英文名 | 中文 | 效果 | 来源 |
|---|---|---|---|
| Chains of Binding | 束缚之链 | 每回合前 X 张抽到的牌被「束缚（Bound）」 | Queen |
| Constrict | 缠绕 | 活着的 Slithering Strangler 使你在回合结束受到 X 伤害 | Slithering Strangler |
| Dampen | 压制 | 活着的 Magi Knight 使你的所有卡牌被降级（Downgraded） | Magi Knight |
| Disintegration | 解体 | 回合结束时受到 X 伤害 | Knowledge Demon |
| Hex | 妖术 | 活着的 Spectral Knight 使你的所有卡牌获得虚影 | Spectral Knight |
| Imbalanced | 失衡 | 若敌人的攻击被完全格挡，其被眩晕 | Bowlbug (Rock) |
| Magic Bomb | 魔法炸弹 | 回合结束受到 X 伤害；Magi Knight 死亡后清除 | Magi Knight |
| Mind Rot | 心智腐化 | 每回合少抽 X 张 | Knowledge Demon |
| Plow | 犁地 | 该敌人生命首次降至 X 以下时被眩晕并失去全部力量 | Ceremonial Beast |
| Ringing | 鸣响 | 本回合只能打出 1 张牌 | Ceremonial Beast |
| Shriek | 尖啸 | 该敌人生命首次降至 X 以下时被眩晕 | Terror Eel |
| Sloth | 怠惰 | 每回合最多打出 X 张牌 | Knowledge Demon |
| Slow | 迟缓 | 每打出 1 张牌，该敌人本回合受到攻击伤害 +10% | Bygone Effigy |
| Smoggy | 烟雾 | 每回合只能打出 1 张技能牌 | Living Fog |
| Surrounded | 被包围 | 从背后受到攻击时伤害 +50%；可用指向性卡/药水转向 | Crusher Rocket |
| Tangled | 缠绕 | 2 回合内攻击牌费用 +1 | Vine Shambler |
| Tender | 温柔 | 每打出 1 张牌，本回合失去 X 力量与 X 敏捷 | Hunter Killer |
| Waste Away | 衰败 | 每回合少获得 X 能量 | Knowledge Demon |

> 另被引用但未单列：**Stunned（眩晕）**（Imbalanced/Plow/Shriek 触发；Whistle 施加）、**Sandpit（沙坑）**（Boss「The Insatiable 饕餮」机制：状态卡 Frantic Escape 使其 +1）、**Bound（束缚）**（Chains of Binding：前 X 张抽到的牌被"束缚"）。

### 4.4 附魔 Enchantments（新机制，共 21 种）

> 附魔 = 给牌组中（或卡牌奖励中）的卡牌添加**永久正面效果**；每张卡只能有一个附魔、不可移除/替换（除非该卡被转化）；诅咒与任务卡不可附魔。

| 英文名 | 中文 | 效果 |
|---|---|---|
| Adroit | 灵巧 | 获得 X 格挡（Kifuda 遗物：X=3） |
| Clone | 克隆 | 可在休息点复制此牌（每次翻倍） |
| Corrupted | 腐化 | 攻击伤害 +50%，但打出时失去 2 生命 |
| Glam | 魅惑 | 此牌每场战斗拥有一次重放（Replay） |
| Goopy | 黏糊 | 此牌获得消耗；打出时永久 +1 格挡 |
| Imbued | 灌注 | 每场战斗开始时自动打出此牌（不付费用；限技能牌） |
| Inky | 墨迹 | 此牌施加 1 层虚弱（Blade of Ink 生成 Ink Shiv） |
| Instinct | 本能 | 此牌攻击伤害翻倍 |
| Momentum | 动量 | 打出时本场战斗此牌攻击伤害 +X |
| Nimble | 敏捷 | 此牌获得的格挡 +X |
| Perfect Fit | 严丝合缝 | 洗入抽牌堆时改为置于牌堆顶 |
| Royally Approved | 御准 | 此牌获得先制与保留 |
| Sharp | 锋利 | 此牌伤害 +X |
| Slither | 滑行 | 抽到此牌时费用随机 0–3 |
| Swift | 迅捷 | 每场战斗首次打出此牌时抽 X 张 |
| Soul's Power | 灵魂之力 | 此牌失去消耗 |
| Sown | 播种 | 每场战斗首次打出此牌时获得 1 能量 |
| Spiral | 螺旋 | 此牌获得 1 层重放（限 Strike/Defend） |
| Steady | 稳健 | 此牌获得保留 |
| Tezcatara's Ember | 特斯卡特拉余烬 | 费用 0、额外 3 伤害、获得永恒（限 Strike） |
| Vigorous | 强健 | 每场战斗首次打出时额外造成 X 伤害 |

---

## 5. 缺口与不确定项汇总

1. **Buffs 页尾部**：渲染页在「Machine Learning」处被 50k 截断；其后的 Defect 增益（One for All、Smokestack、Spinner、Storm、Subroutine、Thunder、Trash to Treasure）由 Defect 能力牌卡面数据重建（表中以 * 标注），效果文本与 Buffs 表其余条目一致可靠，但"来源/备注"列信息不完整。
2. **图标剥离**：多处增益/卡面的能量、星、金币图标在抓取中丢失（☐ 标注处），数值语义按上下文推断（如 Genesis/Friendship/Pyre 疑为能量或星），**未凭空编造数值**。
3. **Freeze**：Wiki 的关键词、增益、减益、附魔页面中**均未发现 "Freeze" 关键词或机制**（用户提及的 Freeze 未在官方 Wiki 出现；冰霜相关为 Frost Orb / Hailstorm / Hibernate / Coolant）。此结论基于 Wiki 全表检索，外部搜索因限流未完成交叉验证。
4. **Stunned / Minion / Sandpit / Bound / Downgraded**：被卡面与减益文本引用，但 Wiki 未为其建立独立词条（Stunned 在关键词页有锚点引用 `Keywords#Stunned`，页面正文暂无小节；Sandpit 为 Boss The Insatiable 专属计数）。
5. **Disintegration 状态**的 "6/7/8" 三档伤害与 Sloth/Beckon 等的具体来源敌人未经确认（Wiki 原文如此）。
6. 数据抓取时点：Firecrawl 云端通道限流后全部为本地直连抓取（2026 年 9 月 2 日最后编辑的 Debuffs 页），游戏为 Early Access 活跃开发版本，后续补丁可能改动数值与卡池。

---

## 参考来源

- 卡牌数据（CARGO API，Game="2" 过滤）：`https://slaythespire.wiki.gg/api.php?action=cargoquery&tables=Cards&...`
- 关键词：https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Keywords
- 增益：https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Buffs
- 减益：https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Debuffs
- 附魔：https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Enchantments
- 主页：https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Main
- 数据模块：`Module:Powers/StS2 data/Debuff`（Lua 数据，已抓取）
