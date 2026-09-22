# 杀戮尖塔2：摄政王（The Regent）全卡图鉴与构筑指南

> 数据来源：slaythespire.wiki.gg 官方 Wiki（Cargo 数据库，截至 2026-05，Early Access 当前版本）；构筑部分参考社区攻略。所有数值以 Wiki 为准；指南内容为玩家经验，标注了来源。
> 说明：官方简体中文译名为 **储君**（ZHS），"摄政王"为常见社区译名。英文原名保留以便对照。

---

## 一、角色基础

| 项目 | 内容 |
|---|---|
| 解锁方式 | 用 **Silent（静默猎手）** 完成一局游戏即可解锁（**无需胜利**）；用 Regent 完成一局（输了也算）可解锁下一位角色 **Necrobinder** |
| 最大生命 | **75**（Ascension 2 及以上开局为 **60**） |
| 资源 | 常规能量（每回合 3 点）+ 独特的第二资源「**星（Stars）**」 |
| 初始遗物 | **Divine Right 天赋神权**：每场战斗开始时获得 **3 颗星**（第 2 幕开始时遇到远古生物 Orobas 选择「Touch of Orobas」可将其升级替换为 **Divine Destiny 神圣天命**：每场战斗开始时获得 **7 颗星**，V0.108.0 由 6 加强至 7） |
| 初始牌组 | 打击×4、防御×4、**坠星 Falling Star**×1、**崇敬 Venerate**×1（基础牌无法通过常规卡牌奖励获得） |
| 角色定位 | 卡面描述："众星王座的继承人。掌握宇宙之力，但杂活都是仆从干。"——靠星星引擎与锻造巨剑输出 |

### 核心机制一：星（Stars）⭐

- 星是独立于能量的第二资源，由**卡牌/遗物/药水生成**，有**星费（Star cost）**的卡牌需消耗星才能打出（有些同时还要能量，如 Devastate 需 1 能量+4 星）。
- **星不会在回合结束时清零，且无上限**（官方 Wiki 遗物页 + untapped.gg 攻略确认），可以跨回合囤积。
- 初始遗物只给一次性 3 星，因此**星必须靠卡牌持续生产**（Venerate、Glow、Hidden Cache、Solar Strike、Shining Strike、Gather Light、Royal Gamble、Genesis 等），这是 Regent 最大的运营课题。
- 星相关关键词：**获得星**（generation）、**消耗星**（spending，可被 Black Hole/Child of the Stars/Galactic Dust/Mini Regent 等联动）、X 费星卡（Stardust，StarCost=-1，投入 X 颗星）。

### 核心机制二：锻造（Forge）与至高之刃（Sovereign Blade）

- **Sovereign Blade 至高之刃**：无色 Token 攻击牌，2 费（升级后 1 费）、保留（Retain）、基础伤害 **10**。**无法**在战斗外加入牌组，只能通过锻造创造。
- **锻造（Forge X）**：每场战斗**第一次锻造**时，将一张至高之刃置入手牌（伤害 = 10 + 本次锻造数值）；此后每次锻造都会给**所有区域（含消耗堆）中所有至高之刃**永久（本场战斗）增加 X 点伤害。若至高之刃已离开牌组（被消耗/转变），再锻造会生成一把新的（伤害仅 10+本次数值）。
- 锻造来源卡牌：精炼之刃(8/12)、战利品(6/9)、战火锻造(7/9)、壁垒(10/13)、征服者(3/5)、熔炉(每回合 5/7)、召来(8/11)、千锤百炼(5/7+每多命中一次+5/7)、大爆炸(5)、追锋(7/11)、铁匠(30/40)；遗物：Fencing Manual（战斗开始锻造 10）；药水：King's Courage（锻造 15）。
- 联动：Seeking Edge（打全体）、Sword Sage（重放 1=打两次）、Parry（获得格挡）、Conqueror（本回合双倍伤害）、Heirloom Hammer（复制之刃）、Summon Forth（任意区域回收）。
- 注意：Wiki 至高之刃页面所列 Refine Blade "9(13)"、Spoils of Battle "5(8)" 为**旧版本数据**，当前版本（V0.111.0 调整）为 **8(12)** 与 **6(9)**，本图鉴以 Cargo 当前数据为准。

### 核心机制三：转变仆从（Transform / Minions）

- BEGONE!（手牌→仆从打击）、CHARGE!!（抽牌堆→仆从俯冲炸弹）、GUARDS!!!（手牌→仆从献祭）将卡牌**转变为 0 费仆从牌**：仆从打击（6/9 伤害+抽 1）、仆从俯冲炸弹（13/16 伤害）、仆从献祭（7/10 格挡），均带消耗。
- 用途：剔除废牌、精简牌组、配合 Vitruvian Minion 遗物翻倍、支撑无限循环。

### 核心机制四：碎片（Debris）

- 撞击航向/坠毁着陆等牌会往手牌塞**碎片**（0 费状态牌，打出后消耗）。目前与之互动的牌很少，属于小众机制，多数构筑不建议围绕它（社区共识）。

### 专属遗物

| 稀有度 | 遗物 | 效果 |
|---|---|---|
| 初始 | Divine Right 天赋神权 | 每场战斗开始获得 3 颗星（可升级为 Divine Destiny，7 星） |
| 普通 | Fencing Manual 剑术手册 | 每场战斗开始时锻造 10 |
| 罕见 | Galactic Dust 银河尘埃 | 每消耗 10 颗星，获得 10 点格挡 |
| 罕见 | Regalite 王权水晶 | 每回合第一次创造卡牌时，获得 4 点格挡（V0.111.0 由 6 削弱为 4） |
| 稀有 | Lunar Pastry 月饼 | 每回合结束时获得 1 颗星 |
| 稀有 | Mini Regent 迷你摄政王 | 每回合第一次消耗星时，获得 1 点力量 |
| 稀有 | Orange Dough 橙色面团 | 每场战斗开始时将 2 张随机无色牌加入手牌 |
| 商店 | Vitruvian Minion 维特鲁威仆从 | 名称含"Minion"的牌伤害和格挡翻倍（作用于三种仆从牌，仆从打击的抽牌不翻倍） |

### 专属药水

| 稀有度 | 药水 | 效果 |
|---|---|---|
| 普通 | Star Potion 星之药水 | 获得 3 颗星 |
| 罕见 | King's Courage 王者之勇 | 锻造 15 |
| 稀有 | Cosmic Concoction 宇宙合剂 | 将 3 张已升级的无色牌加入手牌 |

### 纪元（Epoch）解锁奖励

| 纪元 | 解锁条件 | 奖励 |
|---|---|---|
| I Arrive 抵达 | 击败第一幕 | 卡牌：Spoils of Battle 战利品、Furnace 熔炉、Sword Sage 剑圣 |
| We Need a Hero! 我们需要英雄！ | 击败第二幕 | 遗物：Fencing Manual、Galactic Dust、Lunar Pastry |
| Grand Strategy 宏大战略 | 击败第三幕 | 药水：Star Potion、King's Courage、Cosmic Concoction |
| Little King 小国王 | 击败进阶 1 | 卡牌：Patter 絮语、Lunar Blast 月华冲击、Heavenly Drill 天界钻击 |
| Friends 朋友们 | 击杀 15 名精英 | 卡牌：BEGONE!、Supermassive 超大质量、Arsenal 武库 |
| Discontent 不满 | 击杀 15 名首领 | 遗物：Regalite、Mini Regent、Orange Dough |

（附：用 Regent 完成一局——即使失败——解锁 Necrobinder。）

---

## 二、完整卡牌图鉴（86 张 + 衍生牌）

> 格式：**名称（英文）**｜类型｜能量费（基础→升级）｜星费｜效果（基础/升级）
> 注：星费栏"—"表示无星费；"X"表示投入 X 颗星/点能量的 X 费牌。所有"伤害/格挡"数值格式为 基础/升级。

### 基础卡（Basic，仅初始牌组，不可从奖励获得）— 4 张

| 名称 | 类型 | 费 | 星费 | 效果 |
|---|---|---|---|---|
| 打击 Strike (Regent) | 攻击 | 1 | — | 造成 6/9 点伤害 |
| 防御 Defend (Regent) | 技能 | 1 | — | 获得 5/8 点格挡 |
| 坠星 Falling Star | 攻击 | 0 | 2 | 造成 8/12 点伤害；施加 1 层虚弱；施加 1 层易伤 |
| 崇敬 Venerate | 技能 | 1 | — | 获得 2/3 颗星 |

### 普通卡（Common）— 20 张

| 名称 | 类型 | 费 | 星费 | 效果 |
|---|---|---|---|---|
| 星界脉冲 Astral Pulse | 攻击 | 0 | 3 | 对所有敌人造成 6/8 点伤害，共两次 |
| 退散！BEGONE! | 技能 | 1 | — | 选择手牌中一张牌，转变为「仆从打击」（升级后转为升级版） |
| 天界威能 Celestial Might | 攻击 | 2 | — | 造成 6 点伤害，重复 3/4 次 |
| 星之斗篷 Cloak of Stars | 技能 | 0 | 1 | 获得 7/10 点格挡 |
| 撞击航向 Collision Course | 攻击 | 0 | — | 造成 10/14 点伤害；将 1 张「碎片」加入手牌 |
| 宇宙冷漠 Cosmic Indifference | 技能 | 1 | — | 获得 6/9 点格挡；将弃牌堆中一张牌置于抽牌堆顶 |
| 新月长矛 Crescent Spear | 攻击 | 1 | 1 | 造成 8 点伤害；你每有一张带星费的牌，额外造成 2/3 点伤害 |
| 碾碎 Crush Under | 攻击 | 1 | — | 对所有敌人造成 8/9 点伤害；所有敌人本回合失去 1/2 点力量 |
| 汇聚光芒 Gather Light | 技能 | 1 | — | 获得 8/11 点格挡；获得 1 颗星 |
| 流光 Glitterstream | 技能 | 2 | — | 获得 11/13 点格挡；下一回合获得 5/7 点格挡 |
| 辉光 Glow | 技能 | 1 | — | 获得 1/2 颗星；抽 1 张牌；下一回合抽 1 张牌 |
| 指引之星 Guiding Star | 技能 | 1 | 1 | 造成 12/13 点伤害；下一回合抽 2/3 张牌 |
| 隐藏宝库 Hidden Cache | 技能 | 1 | — | 获得 1 颗星；下一回合获得 3/4 颗星 |
| 认清本分 Know Thy Place | 技能 | 0 | — | 施加 1 层虚弱；施加 1 层易伤（基础版消耗；升级版不再消耗） |
| 絮语 Patter | 技能 | 1 | — | 获得 8/10 点格挡；获得 2/3 点活力 |
| 光子斩 Photon Cut | 攻击 | 1 | — | 造成 10/13 点伤害；抽 1/2 张牌；将手牌中 1 张牌置于抽牌堆顶 |
| 精炼之刃 Refine Blade | 技能 | 1 | — | 锻造 8/12；下一回合获得 1 点能量 |
| 烈阳打击 Solar Strike | 攻击 | 1 | — | 造成 9/10 点伤害；获得 1/2 颗星 |
| 战利品 Spoils of Battle | 技能 | 1 | — | 锻造 6/9；抽 2 张牌 |
| 战火锻造 Wrought in War | 攻击 | 1 | — | 造成 7/9 点伤害；锻造 7/9 |

### 罕见卡（Uncommon）— 35 张

| 名称 | 类型 | 费 | 星费 | 效果 |
|---|---|---|---|---|
| 校准 Alignment | 技能 | 0 | 2 | 获得 2/3 点能量 |
| 黑洞 Black Hole | 能力 | 1 | — | 每当你消耗或获得星时，对所有敌人造成 3/4 点伤害 |
| 壁垒 Bulwark | 技能 | 2 | — | 获得 12/15 点格挡；锻造 10/13 |
| 冲锋！！CHARGE!! | 技能 | 1 | — | 选择抽牌堆中 2 张牌，转变为「仆从俯冲炸弹」（升级后为升级版） |
| 星辰之子 Child of the Stars | 能力 | 1 | — | 每当你消耗星时，每颗星获得 2/3 点格挡 |
| 征服者 Conqueror | 技能 | 1 | — | 锻造 3/5；本回合「至高之刃」对该敌人造成双倍伤害 |
| 汇聚 Convergence | 技能 | 1 | — | 下一回合获得 1 点能量和 1/2 颗星；本回合保留手牌 |
| 毁灭 Devastate | 攻击 | 1 | 4 | 造成 35/45 点伤害 |
| 熔炉 Furnace | 能力 | 1 | — | 每回合开始时锻造 5/7 |
| 伽马爆破 Gamma Blast | 攻击 | 0 | 3 | 造成 13/18 点伤害；施加 2 层虚弱；施加 2 层易伤 |
| 微光 Glimmer | 技能 | 1 | — | 抽 3/4 张牌；将手牌中 1 张牌置于抽牌堆顶 |
| 霸权 Hegemony | 攻击 | 2 | — | 造成 15/18 点伤害；下一回合获得 2/3 点能量 |
| 王者踢击 Kingly Kick | 攻击 | 4 | — | 造成 27/35 点伤害；每当抽到这张牌时费用 -1 |
| 王者重拳 Kingly Punch | 攻击 | 1 | — | 造成 8/10 点伤害；每当抽到这张牌时，本场战斗伤害 +4/6 |
| 击倒重击 Knockout Blow | 攻击 | 3 | — | 造成 30/38 点伤害；若以此击杀敌人，获得 5 颗星 |
| 月华冲击 Lunar Blast | 攻击 | 0 | — | 本回合每打过一张技能牌，造成 4/5 点伤害 |
| 彰显权威 Manifest Authority | 技能 | 1 | — | 获得 7/8 点格挡；将 1 张随机无色牌加入手牌（升级后为已升级的无色牌） |
| 独白 Monologue | 技能 | 0 | — | 本回合每打出一张牌，获得 1 点临时力量（升级版获得保留） |
| 轨道 Orbit | 能力 | 2→1 | — | 每消耗 4 点能量，获得 1 点能量 |
| 暗淡蓝点 Pale Blue Dot | 能力 | 1 | — | 若你本回合打出至少 5 张牌，下回合开始时抽 1/2 张牌 |
| 招架 Parry | 能力 | 1 | — | 「至高之刃」现在获得 10/14 点格挡 |
| 粒子之墙 Particle Wall | 技能 | 0 | 2 | 获得 9/12 点格挡；将这张牌收回手牌 |
| 创世之柱 Pillar of Creation | 能力 | 1 | — | 每当你创造一张卡牌，获得 2/3 点格挡 |
| 预言 Prophesize | 技能 | 2 | — | 抽 6/9 张牌 |
| 类星体 Quasar | 技能 | 0 | 2 | 从 3 张随机无色牌中选择 1 张加入手牌（升级后为升级版） |
| 辐射 Radiate | 攻击 | 0 | — | 本回合每获得 1 颗星，对所有敌人造成 3/4 点伤害 |
| 反射 Reflect | 技能 | 1 | 3 | 获得 15/20 点格挡；本回合被格挡的伤害反射给攻击者 |
| 共振 Resonance | 技能 | 1 | 2 | 获得 1/2 点力量；所有敌人失去 1 点力量 |
| 皇家豪赌 Royal Gamble | 技能 | 0 | 5 | 获得 9 颗星；消耗（升级版获得保留） |
| 闪耀打击 Shining Strike | 攻击 | 1 | — | 造成 8/11 点伤害；获得 2 颗星；将这张牌置于抽牌堆顶 |
| 光谱位移 Spectrum Shift | 能力 | 2→1 | — | 每回合开始时将 1 张随机无色牌加入手牌 |
| 星尘 Stardust | 攻击 | 0 | X | 对随机敌人造成 5/7 点伤害，重复 X 次（X=投入的星数，StarCost=-1 按 X 费牌解读） |
| 召来 Summon Forth | 技能 | 1 | — | 将「至高之刃」从任意区域置入手牌；锻造 8/11 |
| 超大质量 Supermassive | 攻击 | 1 | — | 造成 5 点伤害；本场战斗每创造过一张卡牌，额外造成 3/4 点伤害 |
| 星球改造 Terraforming | 技能 | 1 | — | 获得 7/10 点活力 |

### 稀有卡（Rare）— 25 张

| 名称 | 类型 | 费 | 星费 | 效果 |
|---|---|---|---|---|
| 武库 Arsenal | 能力 | 1 | — | 每当你创造一张卡牌，获得 1 点力量（升级版获得固有） |
| 千锤百炼 Beat into Shape | 攻击 | 1 | — | 造成 5/7 点伤害；锻造 5/7；本回合每额外命中该敌人一次，再锻造 5/7 |
| 大爆炸 Big Bang | 技能 | 0 | — | 抽 1 张牌；获得 1 点能量；获得 1 颗星；锻造 5；消耗（升级版获得固有） |
| 轰击 Bombardment | 攻击 | 3 | — | 造成 18/24 点伤害；每回合开始时若此牌在消耗堆中则自动打出；消耗 |
| 欢乐礼包 Bundle of Joy | 技能 | 1 | — | 将 3/4 张随机无色牌加入手牌 |
| 彗星 Comet | 攻击 | 0 | 5 | 造成 33/44 点伤害；施加 3 层虚弱；施加 3 层易伤 |
| 坠毁着陆 Crash Landing | 攻击 | 1 | — | 对所有敌人造成 21/26 点伤害；手牌填满「碎片」 |
| 抉择，抉择 Decisions, Decisions | 技能 | 0 | 6 | 抽 3/5 张牌；选择手牌中一张技能牌，打出 3 次；消耗 |
| 垂死之星 Dying Star | 攻击 | 1 | 3 | 虚无；对所有敌人造成 9/11 点伤害；所有敌人本回合失去 9/11 点力量 |
| 命中注定 Foregone Conclusion | 技能 | 1 | — | 下一回合将抽牌堆中 2/3 张牌置入手牌 |
| 创世纪 Genesis | 能力 | 2 | — | 每回合开始时获得 2/3 颗星 |
| 护驾！！！GUARDS!!! | 技能 | 2 | — | 将手牌中任意数量卡牌转变为「仆从献祭」（升级后为升级版）；消耗 |
| 天界钻击 Heavenly Drill | 攻击 | X | — | 造成 8/10 点伤害，重复 X 次；若 X≥4 则 X 翻倍（Cost=-1 的 X 费牌） |
| 传家之锤 Heirloom Hammer | 攻击 | 2 | — | 造成 20/25 点伤害；选择手牌中一张无色牌，将一张复制品加入手牌 |
| 我乃无敌 I Am Invincible | 技能 | 1 | — | 获得 10/13 点格挡；回合结束时若此牌位于抽牌堆顶则自动打出 |
| 如你所愿 Make It So | 攻击 | 0 | — | 造成 6/9 点伤害；每回合每打出 3 张技能牌，将这张牌置入手牌 |
| 君主凝视 Monarch's Gaze | 能力 | 2→1 | — | 每当你攻击一名敌人，其本回合失去 1 点力量 |
| 中子神盾 Neutron Aegis | 能力 | 1 | 5 | 获得 8/11 点镀层 |
| 王室贡金 Royalties | 能力 | 1 | — | 战斗结束时获得 30/40 金币 |
| 追锋 Seeking Edge | 能力 | 1 | — | 锻造 7/11；「至高之刃」现在对所有敌人造成伤害 |
| 七星 Seven Stars | 攻击 | 2→1 | 7 | 对所有敌人造成 7 点伤害，重复 7 次 |
| 剑圣 Sword Sage | 能力 | 2→1 | — | 「至高之刃」获得重放 1（每回合打出两次） |
| 铁匠 The Smith | 技能 | 1 | 4 | 锻造 30/40 |
| 暴政 Tyranny | 能力 | 1 | — | 每回合开始时抽 1 张牌并从手牌消耗 1 张（升级版获得固有） |
| 虚空形态 Void Form | 能力 | 3 | — | 结束你的回合；此后每回合前 2 张牌免费打出（升级版获得虚无） |

### 远古卡（Ancient，无法以常规途径获得）— 2 张

| 名称 | 类型 | 费 | 星费 | 效果 |
|---|---|---|---|---|
| 流星雨 Meteor Shower | 攻击 | 0 | 2 | 对所有敌人造成 14/21 点伤害；对所有敌人施加 2 层虚弱和 2 层易伤 |
| 封印王座 The Sealed Throne | 能力 | 1 | 3 | 每当你打出一张牌，获得 1 颗星 |

### 多人模式专属（Multiplayer）— 5 张

| 名称 | 类型 | 费 | 星费 | 效果 |
|---|---|---|---|---|
| 慷慨赏赐 Largesse | 技能 | 0 | — | 另一名玩家将 1 张随机无色牌加入手牌（升级后为升级版） |
| 星座 Constellation | 技能 | 0 | 2 | 另一名玩家抽 1 张牌、获得 1 点能量和 9/12 点格挡 |
| 密谋 Plot | 技能 | 1 | — | 下一回合所有玩家抽 2/3 张牌 |
| 锤击时刻 Hammer Time | 能力 | 2→1 | — | 每当你锻造时，所有队友也锻造同等数值 |
| 导师 Tutor | 技能 | 1→0 | — | 另一名玩家选择抽牌堆中一张牌置入手牌 |

### 衍生牌（Token）— 仆从与至高之刃

| 名称 | 类型 | 费 | 效果 |
|---|---|---|---|
| 仆从打击 Minion Strike | 攻击 | 0 | 造成 6/9 点伤害；抽 1 张牌；消耗（BEGONE! 创造） |
| 仆从俯冲炸弹 Minion Dive Bomb | 攻击 | 0 | 造成 13/16 点伤害；消耗（CHARGE!! 创造） |
| 仆从献祭 Minion Sacrifice | 技能 | 0 | 获得 7/10 点格挡；消耗（GUARDS!!! 创造） |
| 至高之刃 Sovereign Blade | 攻击 | 2→1 | 保留；造成 10 点伤害。由首次「锻造」创造，每次锻造永久（本场战斗）增加其伤害 |

---

## 三、流派构筑

> 以下为社区攻略归纳（intoindiegames、untapped.gg、Caleb Gannon 视频等，见文末来源），供参考；Early Access 数值可能随版本调整。

### 1. 星星流（Star Build）— 最稳定、公认的主力流派 ⭐

- **思路**：大量产星 → 囤星或当回合花星 → 用高星费爆发牌终结。星不跨回合清零且无上限，是囤积流成立的根本。
- **核心输出**：**辐射 Radiate**（每获得 1 星对全体造成 3/4 伤害——主流 AOE 终结技）、**星尘 Stardust**（X 星随机打 X 次——囤星爆发）、**毁灭 Devastate**（4 星 35/45 单体）、**彗星 Comet**（5 星 33/44+3 虚弱+3 易伤）、**七星 Seven Stars**（7 星全体 7×7）、**伽马爆破 Gamma Blast**（3 星 13/18+2 虚弱+2 易伤）、**星界脉冲 Astral Pulse**（3 星全体 6/8×2）。
- **产星引擎**：**辉光 Glow**（1/2 星+过牌）、**隐藏宝库 Hidden Cache**（1 费下回合 3/4 星，性价比之王）、**崇敬 Venerate**（2/3 星）、**烈阳打击 Solar Strike**（9/10 伤+1/2 星）、**闪耀打击 Shining Strike**（8/11 伤+2 星回顶）、**汇聚光芒 Gather Light**（格挡+1 星）、**皇家豪赌 Royal Gamble**（5 星换 9 星+消耗）、**创世纪 Genesis**（每回合 2/3 星的能力引擎）、**击倒重击 Knockout Blow**（击杀得 5 星）。
- **联动**：**黑洞 Black Hole**（花/得星都对全体造成 3/4 伤害，被动星伤）、**星辰之子 Child of the Stars**（每花 1 星 2/3 格挡——星流主力防御）、**校准 Alignment**（2 星换 2/3 能量，升级后驱动无限）。
- **活力加成**：**絮语 Patter**（格挡+活力）、**星球改造 Terraforming**（7/10 活力）→ Radiate 每星伤害翻倍（指南称 Patter+Radiate 可使每星伤害翻倍，攒足活力每星可打 50+）。
- **关键遗物/药水**：Lunar Pastry（每回合末 +1 星）、Galactic Dust（每花 10 星 10 格挡）、Mini Regent（每回合首次花星 +1 力量）、Star Potion（+3 星）、King's Courage（锻造 15，锦上添花）。
- **花星方式两种**：①即花即走（Solar Strike/Gather Light 现产现用，配 Astral Pulse、粒子之墙、星之斗篷）；②囤星一波（Hidden Cache → 下回合 Radiate/Stardust 清场）。
- **前期注意事项**：初始只有 3 星+每轮 Venerate 的 2 星，星费牌不宜贪多（untapped.gg：星费牌与产星牌的比例必须平衡，否则强力卡打不出去）。

### 2. 锻造流（Forge / Sovereign Blade 流派）— 一刀流终结

- **思路**：叠锻造把至高之刃养到一击必杀。Sovereign Blade 2 费+保留，适合配合易伤/虚弱、征服者等倍率打大数字。
- **核心卡**：**壁垒 Bulwark**（格挡+锻造 10/13，公认最佳锻造牌——"为格挡价值拿它，锻造是赠品"）、**熔炉 Furnace**（每回合自动锻造 5/7，过 Boss 的廉价成长）、**战火锻造 Wrought in War**、**精炼之刃 Refine Blade**（锻造 8/12+下回合能量）、**战利品 Spoils of Battle**（锻造 6/9+抽 2）、**千锤百炼 Beat into Shape**（多段命中额外锻造）、**铁匠 The Smith**（4 星锻造 30/40，大跳板）、**大爆炸 Big Bang**（0 费全能启动）。
- **关键支援**：**追锋 Seeking Edge**（锻造 7/11+刀打全体）、**征服者 Conqueror**（刀本回合双倍）、**剑圣 Sword Sage**（刀重放 1=伤害翻倍，且 Parry 格挡也翻倍）、**招架 Parry**（刀获得 10/14 格挡）、**召来 Summon Forth**（任意区域回收再打一刀）、**传家之锤 Heirloom Hammer**（复制带伤害的刀）。
- **遗物/药水**：Fencing Manual（开局锻造 10，直接送刀）、King's Courage（锻造 15）。
- **弱点（社区观点）**：刀通常一战斗只打一两次；多敌战斗（未升级刀时）效率低；被降低伤害类敌人克制（Slippery 滑溜、Hardened Shell 硬化壳、Hard to Kill 难杀）；intoindiegames 建议避免剑圣（Replay 使刀的费用压力变大）、铁匠（星费太高）、招架（收益小）——但这些是个人观点，Wiki 认为 Sword Sage/Parry 是强联动。锻造流"情境依赖"，适合自然抓到几张锻造牌时顺路养刀，不宜强行放弃星卡。

### 3. 无色流（Colorless Build）— RNG 高但上限爆炸

- **思路**：靠随机无色牌制造与"创造卡牌"联动。
- **核心**：**超大质量 Supermassive**（每创造一张牌 +3/4 伤害，可轻松 100+）、**光谱位移 Spectrum Shift**（每回合一张随机无色牌）、**彰显权威 Manifest Authority**、**类星体 Quasar**、**欢乐礼包 Bundle of Joy**、**传家之锤 Heirloom Hammer**、**创世之柱 Pillar of Creation**（创造得格挡）、**武库 Arsenal**（创造得力量）。
- **遗物/药水**：Orange Dough（开局 2 张随机无色牌）、Cosmic Concoction（3 张升级无色牌）。
- **评价**：有趣但依赖随机；Bundle of Joy/Quasar 被部分指南评为不值（除非走无色流）。

### 4. 过牌/手牌控制流（Hand & Draw Control）— 循环同一套牌

- **思路**：反复循环核心牌收益。**王者重拳 Kingly Punch**（每次抽到 +4/6 伤害）、**王者踢击 Kingly Kick**（每次抽到 -1 费，4 费可降到 0）、**月华冲击 Lunar Blast**（每打一张技能 +4/5 伤害）、**如你所愿 Make It So**（每 3 张技能回手）。
- **调度牌**：**宇宙冷漠 Cosmic Indifference**（弃牌堆顶→抽牌堆顶）、**微光 Glimmer**、**光子斩 Photon Cut**（把手牌置顶）、**命中注定 Foregone Conclusion**（下回合定向置入手牌）、**预言 Prophesize**（抽 6/9）、**辉光 Glow**。
- 指南建议：Kingly Kick 配合 Photon Cut/Cosmic Indifference 置顶可快速降费。

### 5. 无限流（Alignment 无限 / 牌组精简）— 上限最高、最吃构筑

- **需求**：**校准 Alignment+**（升级）与**辉光 Glow+**（升级），牌组压到 **≤10 张**（商店删牌、事件、消耗、BEGONE!/CHARGE!! 转变）。
- **循环**：打 Alignment（2 星→3 能量）→ 打 Glow（得星+抽牌）→ 抽回 Alignment → 无限循环，星星无限增长。
- **终结**：**辐射 Radiate+**（每得星全队伤害——无限星=秒杀）；也可用星尘 Stardust。
- **精简工具**：**退散！BEGONE!**、**暴政 Tyranny**（每回合固定消耗 1 张）、**冲锋！！CHARGE!!**、**护驾！！！GUARDS!!!**、**轰击 Bombardment**（Tyranny 消耗后每回合仍自动打出，指南点名配合）。
- **遗物**：**维特鲁威仆从 Vitruvian Minion**（商店遗物，仆从牌伤害/格挡翻倍，强化转变流）。

### 6. 力量/强化流（Strength 体系）

- **核心**：**共振 Resonance**（1/2 力量+全体 -1 力量）、**独白 Monologue**（当回合每牌 +1 临时力量）、**武库 Arsenal**（创造牌 +1 力量）、**君主凝视 Monarch's Gaze**（攻击使敌人 -1 力量）、**垂死之星 Dying Star**（全体 9/11 伤害+全体 -9/11 力量）、**碾碎 Crush Under**。
- **遗物**：Mini Regent（每回合首次花星 +1 力量）。
- 注：Sovereign Blade 是单发高伤，力量/活力等**加法**加成对它收益小，**易伤/征服者等乘法**加成收益大（Wiki 明确指出）。

### 7. 能量流（Energy 体系，副流派）

- **校准 Alignment**（星→能量）、**轨道 Orbit**（每 4 能量返 1，升级后 1 费很强）、**霸权 Hegemony**（15/18 伤+下回合 2/3 能量）、**汇聚 Convergence**（下回合 1 能量+1/2 星+保留手牌）、**虚空形态 Void Form**（结束回合，之后每回合前 2 张免费——指南评价"近乎破坏游戏"，需配合过牌）。
- 能量与星双资源并管是 Regent 高手向的核心课题。

### 指南公认"无脑强/必抓"卡（跨流派）

- **流星雨 Meteor Shower**（远古，坠星强化版+全体）、**垂死之星 Dying Star**、**轰击 Bombardment**（每回合自动打）、**暴政 Tyranny**（精简）、**冲锋！！CHARGE!!**（精简+仆从）、**大爆炸 Big Bang**（能量+星+锻造+抽牌四合一）、**护驾！！！GUARDS!!!**（废牌变格挡）、**命中注定 Foregone Conclusion**、**宇宙冷漠 Cosmic Indifference**、**封印王座 The Sealed Throne**（与黑洞/星联动极强）。
- **公认偏弱/慎拿**（intoindiegames 观点）：欢乐礼包 Bundle of Joy（塞随机牌）、类星体 Quasar（贵）、碎片相关牌（小众）；untapped.gg 提醒**指引之星 Guiding Star** 表面强但 1 星费会挤压其他星卡（注意：untapped 称其 2 星费，Cargo 当前数据为 1 星，可能存在版本差）。

### 开局与总体策略

- **第 1 幕**优先抓：星界脉冲 Astral Pulse、撞击航向 Collision Course、光子斩 Photon Cut、认清本分 Know Thy Place、退散！BEGONE!、宇宙冷漠 Cosmic Indifference、烈阳打击 Solar Strike（untapped.gg 推荐）；intoindiegames 补充：伽马爆破、辐射、粒子之墙、星辰之子、熔炉、击倒重击（30/38 伤+击杀 5 星，精英战利器）。
- **格挡不能丢**：专注星输出时也要拿 1-2 张格挡（Patter、Cloak of Stars、Particle Wall、Reflect、Bulwark 为最佳格挡牌）。
- **综合优先级**（intoindiegames）：① 建星引擎 → ② 抓辐射 Radiate → ③ 补活力加成 → ④ 精简牌组。混合构筑（星+锻造、星+无限）完全可行。

---

## 四、数据可靠性与已知差异说明

- 卡牌数据全部来自 Wiki Cargo 数据库（`Cards` 表，`Game="2" AND Color="Regent"`），共 86 张唯一卡（基础 4 + 普通 20 + 罕见 35 + 稀有 25 + 远古 2）+ 多人专属 5 张（含于上表）+ 衍生牌 3 张 + 至高之刃，与角色页统计（80 张常规卡池 + 5 多人 + 2 远古）吻合。
- **版本差异标记**：
  - Refine Blade 锻造 8(12)、Spoils of Battle 锻造 6(9)：Cargo 与卡牌页当前一致（V0.111.0 调整后）；至高之刃页面中 9(13)/5(8) 为旧数据。
  - Divine Destiny 现为 **7 星**（V0.108.0 由 6 加强）；intoindiegames 攻略写 6 星，已过时。
  - Regalite 现为每回合首次创造 +4 格挡（V0.111.0 由 6 削弱）。
  - untapped.gg 称 Guiding Star 星费 2，Cargo 当前为 1（可能有版本差，以 Cargo 为准）。
- **推断项**：Stardust（StarCost=-1）与 Heavenly Drill（Cost=-1）按 Wiki 惯例解读为 X 费牌（X=投入的星/能量），Wiki 未直接写明"X"字样。
- 卡牌效果中"创造卡牌"（create）指任何将新卡牌加入手牌/牌组的效果（无色牌、仆从、碎片、至高之刃等）。

---

## 来源（Sources）

**官方 Wiki（权威，slaythespire.wiki.gg）：**
- 角色页：https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Regent
- 至高之刃/锻造机制：https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Sovereign_Blade
- 天赋神权：https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Divine_Right ｜ 神圣天命：https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Divine_Destiny
- 卡牌数据（Cargo API，Game="2" AND Color="Regent"）：https://slaythespire.wiki.gg/api.php?action=cargoquery&tables=Cards&fields=Name,Type,Rarity,Cost,CostPlus,StarCost,Description,MultiplayerOnly,Tags&where=Game%3D%222%22%20AND%20Color%3D%22Regent%22&format=json&limit=200
- 各遗物/药水页：Fencing Manual、Galactic Dust、Regalite、Lunar Pastry、Mini Regent、Orange Dough、Vitruvian Minion、Star Potion、King's Courage、Cosmic Concoction（https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:遗物名）
- 精炼之刃/战利品数值核对：https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Refine_Blade 、 https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Spoils_of_Battle

**社区攻略（构筑部分，观点类）：**
- Into Indie Games《The Regent Ultimate Build Guide》：https://intoindiegames.com/walkthroughs/tips-tricks/slay-the-spire-2-the-regent-ultimate-build-guide/
- Untapped.gg《Regent Guide》：https://sts2.untapped.gg/en/characters/regent
- Caleb Gannon Gaming《How to WIN with the Regent》：https://www.youtube.com/watch?v=RSAXZ2K75xo （2026-03，构筑章节：Starfall/Radiate、Value Stars、Lunar Blast、Forge、Colorless、Deck Thin Infinites）
- Reddit 构筑哲学帖：https://www.reddit.com/r/slaythespire/comments/1s210if/a_guide_on_build_philosophy_for_regent/
- Mobalytics《Regent Guide》：https://mobalytics.gg/slay-the-spire-2/characters/regent-guide
- Steam 社区讨论：https://steamcommunity.com/app/2868840/discussions/0/798964766335901878/
- TheGamer《Best Build And Relics For The Regent》（未能抓取全文，仅检索摘要）：https://www.thegamer.com/slay-the-spire-2-best-builds-cards-relics-the-regent-stars-forge-guide/
