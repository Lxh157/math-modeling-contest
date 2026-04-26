## 数学建模课程工作区

本目录用于整理本学期数学建模竞赛的资料、代码、数据与报告。

### 目录约定
- `01_notes/`：课程笔记与知识点沉淀
- `02_template/`：可复用模板（报告、作业、代码骨架、通用函数/画图脚本/常见模型等）
- `03_goingon/`：上手开始做，里面是具体比赛，比赛相关的信息都放在其中
  - `/code`: 代码
  - `/data`: 数据，含赛题提供的原始数据、数据处理后存放的数据
  - `/information`: 赛事相关信息
  - `/paper`: 论文

### 当前环境配置
- `03_goingon/北邮校赛/code` 已配置独立 Python 虚拟环境：`03_goingon/北邮校赛/code/.venv`
- Python 版本：`3.12.4`
- 依赖锁定文件：`03_goingon/北邮校赛/code/requirements.txt`
- 已安装的基础数模常用库包括：`numpy`、`pandas`、`matplotlib`、`scipy`、`scikit-learn`、`seaborn`、`statsmodels`、`sympy`、`openpyxl`、`networkx`、`pulp`、`tqdm`、`jupyterlab`
- PowerShell 激活命令：
  ```powershell
  .\03_goingon\北邮校赛\code\.venv\Scripts\Activate.ps1
  ```
- 直接使用解释器：
  ```powershell
  .\03_goingon\北邮校赛\code\.venv\Scripts\python.exe
  ```
- 如需重建环境，可执行：
  ```powershell
  py -3.12 -m venv .\03_goingon\北邮校赛\code\.venv
  .\03_goingon\北邮校赛\code\.venv\Scripts\python.exe -m pip install -r .\03_goingon\北邮校赛\code\requirements.txt
  ```



