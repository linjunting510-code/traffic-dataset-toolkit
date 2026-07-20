# traffic-dataset-toolkit

交通事件视觉数据集构建的 CLI 工具包。面向"视频抽帧 → 自动标注 → 转 YOLO → 人工校正 → 训练"的数据集流水线,把零散脚本收敛成一个可安装、可测试、可开源的小工具库。

> 本仓库**只包含代码**,数据集与视频不入库(见 `.gitignore`)。

## 数据集流水线

```
原始视频(单个或整个目录)
  │  tds extract(ffmpeg 抽帧,.part 未下载完成的自动跳过)
  ▼
<视频名>_frame_000001.jpg 连续帧图片
  │  (需要改成 000001.jpg 风格时)tds rename ──► 000001.jpg 起的连续编号(保帧序)
  │  自动标注(二选一或结合):
  │   ① tds autolabel(ultralytics YOLO,覆盖 6/9 类)
  │   ② X-AnyLabeling 自动标注(每张图生成同名 .json)── tds convert 转 YOLO
  ▼
labels/*.txt(YOLO 格式)+ classes.txt + autolabel_report.txt
  │  人工校正(X-AnyLabeling 里补漏、改错;Traffic Cone/Barrier/Tree 必须人工补)
  ▼
tds validate ──► 配对/classes.txt/格式/统计 体检报告
  │
  ▼
YOLO 训练
```

## 检测类别(9 类)

唯一定义在 [`configs/classes.yaml`](configs/classes.yaml),所有子命令都从这里读取,**不要在代码里硬编码类别**:

| id | 名称          | 常见别名(大小写不敏感)        |
|----|---------------|-------------------------------|
| 0  | Car           | car                           |
| 1  | Bus           | bus                           |
| 2  | Truck         | truck                         |
| 3  | Motorcycle    | motorbike                     |
| 4  | Pedestrian    | person                        |
| 5  | Traffic Cone  | cone                          |
| 6  | Barrier       | barrier                       |
| 7  | Tree          | tree                          |
| 8  | Traffic light | trafficlight                  |

转换时不在 9 类里的目标(bicycle / airplane / suitcase 等误检)会被自动跳过并统计。

## 安装

```bash
pip install -e .
```

只依赖 `pyyaml`(代码内置了极简 YAML 解析兜底,没有 pyyaml 也能跑)。要求 Python ≥ 3.9。

不安装也可以直接用(仓库根目录下):

```bash
# Windows cmd
set PYTHONPATH=src
python -m traffic_dataset.cli --help

# PowerShell
$env:PYTHONPATH = "src"

# bash
export PYTHONPATH=src
```

## 使用

### tds extract —— 视频抽帧(ffmpeg)

```bash
# 目标总帧数模式(默认):按视频时长自动算帧率,短视频密采、长视频稀采
tds extract "D:\dataset_work\raw_videos\clip001.mp4" -o "D:\dataset_work\frames"

# 整个目录批量(不递归,自动跳过 .part 未完成下载文件)
tds extract "D:\dataset_work\raw_videos" -o "D:\dataset_work\frames" --frames 120

# 固定帧率模式
tds extract "D:\dataset_work\raw_videos" -o "D:\dataset_work\frames" --fps 2
```

- 输出命名:`<视频文件名去后缀>_frame_000001.jpg`(6 位连续编号)
- 两种采样模式互斥:`--fps N` 固定帧率;`--frames M` 目标总帧数(ffprobe 测时长,`fps = M / 时长`)。都不给时默认 `--frames 120`——约能覆盖一个事件片段全过程,标注量可控
- 抽帧前打印计划(每个视频的时长/采用 fps/预计帧数),抽完打印统计
- 输出目录已有同名帧时跳过该视频,避免覆盖重抽
- 需要本机安装 ffmpeg / ffprobe:`winget install ffmpeg` 或[官网](https://ffmpeg.org/download.html)

### tds rename —— 图片批量重命名

```bash
tds rename "D:\dataset_work\Traffic incidents\2\images" --dry-run   # 先预览
tds rename "D:\dataset_work\Traffic incidents\2\images"             # 确认后执行
```

按文件名排序后重命名为 `000001.jpg` 起的连续编号(保连续帧顺序);只处理 jpg/jpeg/png,不碰 json/classes.txt;两步改名防覆盖冲突。

### tds autolabel —— 检测模型自动打标

```bash
# 对事件目录的 images/ 跑 YOLO 检测,labels 写到同目录 labels/
tds autolabel "D:\dataset_work\pilot\国庆白天"

# 固定监控机位建议开跟踪:按帧序 ByteTrack 跟踪,减少漏标抖动
tds autolabel "D:\dataset_work\pilot\国庆夜间" --track --conf 0.3
```

- **类别覆盖**:COCO 预训练模型只能覆盖项目 9 类中的 6 类 —— person→Pedestrian、car→Car、motorcycle→Motorcycle、bus→Bus、truck→Truck、traffic light→Traffic light;其余 COCO 类(bicycle 等)自动跳过并统计
- **局限**:**Traffic Cone / Barrier / Tree 三类 COCO 没有对应类别,模型永远检不出,必须人工补标**;小目标、夜间、遮挡场景检出率低,autolabel 的定位是"减少人工量",不是替代人工复核
- **输出**:`labels/*.txt`(与 convert 同款 YOLO 格式)、`labels/classes.txt`、`labels/autolabel_report.txt`(类别分布、每帧平均置信度最低的 10 帧、零检出帧清单)
- **环境**:需要在打标环境 `pip install ultralytics`(torch 随之安装,CPU 可跑);`--model` 默认 yolo11n.pt(首次自动下载),`--conf` 默认 0.25,`--device` 默认 cpu
- 打标完建议接着跑 `tds validate` 体检(validate 会自动忽略 autolabel_report.txt)

### tds convert —— X-AnyLabeling JSON 转 YOLO

```bash
tds convert "D:\dataset_work\Traffic incidents\2"
```

读取 `<事件目录>/images/*.json`,写出 `<事件目录>/labels/*.txt`(与图片同名)和 `labels/classes.txt`(9 类规范名)。类别名/别名大小写不敏感;9 类以外的目标自动跳过并统计。

### tds validate —— 数据集体检

```bash
tds validate "D:\dataset_work\Traffic incidents\2"
```

检查项:

1. **图片-标注配对**:有图无 txt 记警告;有 txt 无图(孤儿标注)记错误
2. **classes.txt 一致性**:与 `configs/classes.yaml` 的 9 类逐行对比,不一致逐行列出差异
3. **标注格式**:每行 5 列、class_id 在 0-8、坐标/宽高在 (0,1] 区间,越界报错(带文件名+行号)
4. **统计**:图片数、标注文件数、每类框数分布、零样本类别提醒

退出码:有 error 返回 `1`,仅 warning 返回 `0`,可直接接入 CI / pre-commit。

### 类别定义覆盖

默认读取仓库 `configs/classes.yaml`。可用 `--classes <path>` 或环境变量 `TDS_CLASSES_YAML` 指定其他类别文件。

## 目录结构

```
traffic-dataset-toolkit/
├── configs/classes.yaml   # 9 类唯一定义(id→名称 + 别名)
├── src/traffic_dataset/
│   ├── cli.py             # argparse 入口:tds extract/rename/autolabel/convert/validate
│   ├── classes.py         # 类别加载与解析(带 YAML 极简兜底)
│   ├── extract.py         # ffmpeg 视频抽帧
│   ├── rename.py          # 图片重命名
│   ├── autolabel.py       # 检测模型自动打标(惰性 import ultralytics)
│   ├── convert.py         # JSON -> YOLO
│   └── validate.py        # 数据集校验
└── tests/                 # unittest
```

## 测试

```bash
# 仓库根目录,PYTHONPATH 指向 src
export PYTHONPATH=src        # Windows: set PYTHONPATH=src
python -m unittest discover tests
```

## License

MIT,见 [LICENSE](LICENSE)。
