# Jupyter Notebook 批量转 PDF 工具

这是一个用于批量将 Jupyter Notebook 文件转换为 PDF 格式的 Python 脚本。

## 功能特性

- 🔍 **递归搜索**: 自动搜索指定目录及其子目录中的所有 `.ipynb` 文件
- 📄 **批量转换**: 一次性转换多个 notebook 文件
- 🛡️ **错误处理**: 完善的错误处理和日志记录
- 📊 **进度显示**: 实时显示转换进度和结果统计
- 📝 **详细日志**: 生成详细的转换日志文件
- ⚙️ **命令行支持**: 支持命令行参数和选项

## 安装依赖

在使用脚本之前，请确保安装了必要的依赖：

```bash
# 安装 Jupyter 和 nbconvert
pip install jupyter nbconvert

# 安装 LaTeX (用于 PDF 转换)
# Windows: 下载并安装 MiKTeX 或 TeX Live
# macOS: brew install --cask mactex
# Ubuntu/Debian: sudo apt-get install texlive-xetex texlive-fonts-recommended
```

## 使用方法

### 基本用法

```bash
# 转换当前目录中的所有 notebook
python jupyter_to_pdf_converter.py

# 转换指定目录
python jupyter_to_pdf_converter.py homework4

# 转换指定路径
python jupyter_to_pdf_converter.py "d:\code\python\deeplearning\homework4"
```

### 命令行选项

```bash
# 显示详细输出
python jupyter_to_pdf_converter.py --verbose

# 查看帮助信息
python jupyter_to_pdf_converter.py --help
```

## 使用示例

### 示例 1: 转换当前项目的所有 notebook

```bash
cd d:\code\python\deeplearning
python jupyter_to_pdf_converter.py
```

### 示例 2: 转换特定作业目录

```bash
python jupyter_to_pdf_converter.py homework4
```

### 示例 3: 转换并显示详细信息

```bash
python jupyter_to_pdf_converter.py homework4 --verbose
```

## 输出说明

脚本运行时会显示：
- 搜索到的 notebook 文件数量
- 每个文件的转换进度
- 转换成功/失败的统计信息
- 详细的错误信息（如果有）

转换完成后，PDF 文件会保存在与原 notebook 文件相同的目录中。

## 日志文件

脚本会生成 `jupyter_to_pdf.log` 日志文件，包含：
- 转换过程的详细记录
- 错误信息和堆栈跟踪
- 时间戳和操作历史

## 注意事项

1. **LaTeX 环境**: PDF 转换需要 LaTeX 环境，首次使用可能需要下载相关包
2. **网络连接**: 首次转换时可能需要网络连接来下载 LaTeX 包
3. **文件权限**: 确保对目标目录有读写权限
4. **中文支持**: 如果 notebook 包含中文内容，确保 LaTeX 环境支持中文字体

## 常见问题

### Q: 转换失败，提示找不到 LaTeX

**A**: 请安装 LaTeX 环境：
- Windows: 安装 [MiKTeX](https://miktex.org/) 或 [TeX Live](https://tug.org/texlive/)
- macOS: `brew install --cask mactex`
- Linux: `sudo apt-get install texlive-xetex texlive-fonts-recommended`

### Q: 转换中文内容时出现乱码

**A**: 确保 LaTeX 环境支持中文，可能需要安装额外的中文字体包。

### Q: 某些 notebook 转换失败

**A**: 检查 notebook 文件是否损坏，或者包含不支持的内容（如某些特殊的 widget）。

## 脚本结构

```
jupyter_to_pdf_converter.py
├── JupyterToPDFConverter 类
│   ├── find_jupyter_notebooks()     # 搜索 notebook 文件
│   ├── convert_notebook_to_pdf()    # 转换单个文件
│   ├── convert_all()                # 批量转换
│   └── print_summary()              # 输出结果统计
├── check_dependencies()             # 检查依赖
└── main()                          # 主函数和命令行处理
```

## 许可证

此脚本为开源项目，可自由使用和修改。