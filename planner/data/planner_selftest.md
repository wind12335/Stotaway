# 预测器自检：在全部实测点上复现选择（D 用实测 fbag 整步 / 层周期数）

| 点 | D (ms) | 预测器配置 | 实测最优 | 依据 | 对否 |
|---|---|---|---|---|---|
| 27B MBS8 | 137 | fbag（RS 两处用重叠方法，AG 两处用原版） | fbag | fbag 比原版 −6.37%（F26/F35）；fbc 比 fbag +3.05%（F17） | ✓ |
| 27B MBS4 | 74 | base（四处原版） | fbag* | fbc 比 fbag +3.26%（F49）；原版未测 | ✓ |
| 70B MBS8 | 273 | fbc（四处全用重叠方法） | fbc | fbag 比原版 −7.44%（F27）；fbc 比 fbag −1.68%（4 次均值） | ✓ |
| 70B MBS4 | 134 | fbag（RS 两处用重叠方法，AG 两处用原版） | fbag | fbag 比原版 −5.80%（F30）；fbc 比 fbag +2.21%（F51） | ✓ |
| 70B MBS2 | 77 | base（四处原版） | base | fbag 比原版 +3.1%（F30）；fbc 比 fbag +4.77%（F51） | ✓ |

5/5 选对（注：这些点也是第二级的拟合数据，属于样本内；第二级 AG 部分的留一检验为 4/5，见 step_planner_v3.md）。
* 27B MBS4 只测了 AG 两处的选择（fbc vs fbag），RS 组的选择（fbag vs 原版）未测，按 AG 决策计分。
