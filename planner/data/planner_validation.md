# 代价模型 + 规划器：留一模型验证（CPU，仅用已测的算子级数据）

## 计时方式：单次（97 个调用点，11 个模型）

| 指标 | 值 |
|---|---|
| dx 时间预测误差 MAPE | 6.7%（最大 19.5%） |
| 选对最快方案 | 69/97 |
| 相对最优的平均损失 | 1.93%（最差 37.18%） |
| 选中方案比原版慢的格子 | 0 |
| 几何均值加速：规划器 | 1.142× |
| 几何均值加速：总用 dx | 1.029× |
| 几何均值加速：静态 M 阈值 | 1.100× |
| 几何均值加速：事后最优 | 1.163× |

选错的格子：

- ag_gu dsv3-d M16384：选 stock（5.308 ms），最优 dx（4.995 ms），损失 6.3%
- ag_gu dsv3-d M32768：选 stock（10.118 ms），最优 dx（9.129 ms），损失 10.8%
- ag_up dsv3-d M8192：选 split2（1.703 ms），最优 split4（1.648 ms），损失 3.3%
- ag_up dsv3-d M16384：选 split2（3.150 ms），最优 split4（2.986 ms），损失 5.5%
- ag_up gemma3-27b M8192：选 stock（1.573 ms），最优 split2（1.549 ms），损失 1.5%
- ag_up gemma3-27b M16384：选 split2（2.606 ms），最优 split4（2.518 ms），损失 3.5%
- ag_gu llama3-70b M32768：选 stock（15.265 ms），最优 dx（14.046 ms），损失 8.7%
- ag_up llama3-70b M16384：选 split2（4.432 ms），最优 split4（4.227 ms），损失 4.9%
- ag_gu llama3-8b M32768：选 stock（4.670 ms），最优 dx（4.503 ms），损失 3.7%
- ag_gu llama31-405b M32768：选 stock（57.831 ms），最优 dx（56.652 ms），损失 2.1%
- ag_up llama31-405b M8192：选 split2（8.578 ms），最优 split4（8.213 ms），损失 4.5%
- ag_up llama31-405b M16384：选 split4（16.797 ms），最优 split2（15.691 ms），损失 7.1%
- ag_up llama31-405b M32768：选 stock（33.345 ms），最优 dx（32.534 ms），损失 2.5%
- ag_up llama4-mav-d M8192：选 stock（1.333 ms），最优 split2（1.227 ms），损失 8.7%
- ag_up llama4-mav-d M16384：选 split2（2.159 ms），最优 split4（2.077 ms），损失 3.9%
- rs llama4-mav-d M8192：选 stock（1.575 ms），最优 dx（1.385 ms），损失 13.7%
- ag_up mistral-l-2407 M16384：选 split2（6.732 ms），最优 split4（6.698 ms），损失 0.5%
- ag_gu mistral3-d M16384：选 stock（4.892 ms），最优 dx（4.834 ms），损失 1.2%
- ag_up mistral3-d M16384：选 split2（2.912 ms），最优 split4（2.696 ms），损失 8.0%
- ag_gu qwen25-72b M16384：选 stock（8.078 ms），最优 dx（7.398 ms），损失 9.2%
- ag_gu qwen25-72b M32768：选 stock（16.191 ms），最优 dx（14.282 ms），损失 13.4%
- ag_up qwen25-72b M16384：选 split2（5.168 ms），最优 split4（5.041 ms），损失 2.5%
- ag_up qwen25-72b M32768：选 stock（12.662 ms），最优 dx（9.230 ms），损失 37.2%
- ag_gu qwen3-32b M32768：选 stock（5.028 ms），最优 dx（4.590 ms），损失 9.5%
- ag_up qwen3-32b M16384：选 split4（1.894 ms），最优 split2（1.841 ms），损失 2.9%
- rs qwen3-32b M8192：选 stock（1.444 ms），最优 dx（1.333 ms），损失 8.3%
- ag_gu qwen3-8b M32768：选 stock（4.336 ms），最优 dx（4.273 ms），损失 1.5%
- ag_up qwen3-8b M32768：选 stock（3.220 ms），最优 dx（3.159 ms），损失 1.9%

## 计时方式：连发（97 个调用点，11 个模型）

| 指标 | 值 |
|---|---|
| dx 时间预测误差 MAPE | 9.2%（最大 27.6%） |
| 选对最快方案 | 66/97 |
| 相对最优的平均损失 | 2.58%（最差 24.58%） |
| 选中方案比原版慢的格子 | 0 |
| 几何均值加速：规划器 | 1.139× |
| 几何均值加速：总用 dx | 1.101× |
| 几何均值加速：静态 M 阈值 | 1.120× |
| 几何均值加速：事后最优 | 1.167× |

选错的格子：

- ag_gu dsv3-d M8192：选 stock（2.656 ms），最优 dx（2.132 ms），损失 24.6%
- ag_gu dsv3-d M16384：选 stock（5.246 ms），最优 dx（4.526 ms），损失 15.9%
- ag_up dsv3-d M8192：选 split2（1.588 ms），最优 split4（1.550 ms），损失 2.4%
- ag_up dsv3-d M16384：选 split2（3.050 ms），最优 dx（2.799 ms），损失 9.0%
- ag_up gemma3-27b M8192：选 stock（1.519 ms），最优 split2（1.440 ms），损失 5.5%
- ag_up llama3-70b M16384：选 split2（4.345 ms），最优 split4（4.124 ms），损失 5.4%
- ag_gu llama3-8b M16384：选 stock（2.390 ms），最优 dx（2.311 ms），损失 3.4%
- ag_up llama3-8b M32768：选 stock（3.383 ms），最优 dx（3.036 ms），损失 11.4%
- rs llama3-8b M8192：选 stock（1.263 ms），最优 split2（1.231 ms），损失 2.6%
- ag_gu llama31-405b M32768：选 stock（57.874 ms），最优 dx（56.164 ms），损失 3.0%
- ag_up llama31-405b M8192：选 split2（8.488 ms），最优 split4（8.190 ms），损失 3.6%
- ag_up llama31-405b M16384：选 dx（15.626 ms），最优 split2（15.588 ms），损失 0.2%
- ag_gu llama4-mav-d M8192：选 stock（1.735 ms），最优 dx（1.607 ms），损失 8.0%
- ag_gu llama4-mav-d M16384：选 stock（3.264 ms），最优 dx（3.135 ms），损失 4.1%
- ag_up llama4-mav-d M8192：选 stock（1.283 ms），最优 split2（1.116 ms），损失 15.0%
- ag_up llama4-mav-d M16384：选 split2（2.050 ms），最优 dx（1.892 ms），损失 8.4%
- rs llama4-mav-d M8192：选 stock（1.541 ms），最优 split2（1.523 ms），损失 1.2%
- ag_gu mistral-l-2407 M16384：选 stock（11.284 ms），最优 dx（11.251 ms），损失 0.3%
- ag_up mistral-l-2407 M8192：选 split2（3.405 ms），最优 dx（3.256 ms），损失 4.6%
- ag_up mistral-l-2407 M16384：选 dx（6.935 ms），最优 split4（6.531 ms），损失 6.2%
- ag_gu mistral3-d M8192：选 stock（2.478 ms），最优 dx（2.147 ms），损失 15.4%
- ag_up mistral3-d M16384：选 split2（2.813 ms），最优 split4（2.622 ms），损失 7.3%
- ag_gu qwen25-72b M16384：选 stock（7.970 ms），最优 dx（6.776 ms），损失 17.6%
- ag_gu qwen25-72b M32768：选 stock（15.785 ms），最优 dx（13.364 ms），损失 18.1%
- ag_up qwen25-72b M8192：选 split2（2.414 ms），最优 dx（2.206 ms），损失 9.4%
- ag_up qwen25-72b M16384：选 split2（5.036 ms），最优 dx（4.714 ms），损失 6.8%
- ag_gu qwen3-32b M16384：选 stock（2.510 ms），最优 dx（2.334 ms），损失 7.5%
- ag_gu qwen3-32b M32768：选 stock（4.945 ms），最优 dx（4.138 ms），损失 19.5%
- ag_up qwen3-32b M16384：选 split4（1.788 ms），最优 split2（1.737 ms），损失 3.0%
- rs qwen3-32b M8192：选 stock（1.397 ms），最优 dx（1.327 ms），损失 5.3%
- ag_up qwen3-8b M32768：选 stock（3.168 ms），最优 dx（2.999 ms），损失 5.6%

