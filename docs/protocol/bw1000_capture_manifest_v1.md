# BW1000 实验捕获清单 v1

本清单用于 C3 latency、native 对照和最小 profiler probe。必须在 BW1000 分配到设备的
节点执行；本地运行不能生成正式结果。

## 必须保存的元数据

```text
timestamp:
hostname:
slurm_job_id:
slurm_job_nodelist:
device_name:
device_uuid:
device_arch:
driver_or_runtime:
python:
torch:
triton:
hip_or_cuda:
source_revision:
probe_tree_sha256:
input_manifest_sha256:
```

## C3 latency

- warmup/repeat：与 C4 baseline 明确记录，并使用设备同步；
- batch：至少 `9, 65, 257, 4099, 65537`；
- 输出：min/median/mean/max、标准差、每次返回状态；
- 计时范围：仅 C3 Newton kernel，不包含 C4 状态更新；
- 结果文件：`result/c3-latency-<job_id>.json`；
- 原始输出：`logs/c3-latency-<job_id>.stdout/.stderr`。

## native backend 对照

- 使用相同输入、FP64、batch 和 warmup/repeat；
- 若 BW1000 没有可运行的 SeisSol native backend，保存构建命令、失败日志和源码版本，
  明确记录 `NO_NATIVE_BACKEND`，不得用其他设备代替；
- 分别记录 kernel、数据搬运、同步和端到端时间；
- 结果文件：`result/native-comparison-<job_id>.json`。

## 最小 profiler probe

- 只采集一个已通过 correctness 的 C3 或 C4 kernel；
- 先单一规模、少量 repeat，确认 profiler 不改变结果；
- 保存 occupancy/register、指令组成、launch、内存访问或平台等价指标；
- profiler 失败或指标缺失也要保留 stderr；
- 结果文件：`result/profile-<job_id>/`，不得覆盖 latency baseline。
