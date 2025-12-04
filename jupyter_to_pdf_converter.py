#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jupyter Notebook 批量转换为 PDF 脚本

该脚本可以递归遍历指定文件夹，找到所有的 .ipynb 文件并将其转换为 PDF 格式。

使用方法:
    直接运行脚本，在 main() 函数中修改 target_directory 参数来指定要转换的目录
    
配置参数:
    - target_directory: 要转换的目录路径
    - verbose_mode: 是否显示详细输出

依赖:
    - nbconvert
    - jupyter
    
安装依赖:
    pip install nbconvert jupyter
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import subprocess
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jupyter_to_pdf.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class JupyterToPDFConverter:
    """Jupyter Notebook 到 PDF 转换器"""
    
    def __init__(self, root_dir: str = None):
        """
        初始化转换器
        
        Args:
            root_dir: 根目录路径，默认为当前目录
        """
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.success_count = 0
        self.error_count = 0
        self.errors = []
        
    def find_jupyter_notebooks(self) -> List[Path]:
        """
        递归查找所有 Jupyter notebook 文件
        
        Returns:
            包含所有 .ipynb 文件路径的列表
        """
        notebooks = []
        logger.info(f"正在搜索目录: {self.root_dir}")
        
        for notebook_path in self.root_dir.rglob("*.ipynb"):
            # 跳过 .ipynb_checkpoints 目录中的文件
            if ".ipynb_checkpoints" not in str(notebook_path):
                notebooks.append(notebook_path)
                
        logger.info(f"找到 {len(notebooks)} 个 Jupyter notebook 文件")
        return notebooks
    
    def convert_notebook_to_pdf(self, notebook_path: Path) -> Tuple[bool, str]:
        """
        将单个 notebook 转换为 PDF
        
        Args:
            notebook_path: notebook 文件路径
            
        Returns:
            (成功标志, 错误信息)
        """
        try:
            # 生成 PDF 文件路径
            pdf_path = notebook_path.with_suffix('.pdf')
            
            # 使用 nbconvert 命令行工具进行转换 (直接使用 python -m nbconvert)
            cmd = [
                sys.executable, '-m', 'nbconvert',
                '--to', 'pdf',
                '--output', str(pdf_path),
                str(notebook_path)
            ]
            
            logger.info(f"正在转换: {notebook_path.name}")
            
            # 执行转换命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                logger.info(f"✓ 成功转换: {notebook_path.name} -> {pdf_path.name}")
                return True, ""
            else:
                error_msg = f"转换失败: {result.stderr}"
                logger.error(f"✗ {notebook_path.name}: {error_msg}")
                return False, error_msg
                
        except Exception as e:
            error_msg = f"转换过程中发生异常: {str(e)}"
            logger.error(f"✗ {notebook_path.name}: {error_msg}")
            return False, error_msg
    
    def convert_all(self) -> None:
        """转换所有找到的 notebook 文件"""
        notebooks = self.find_jupyter_notebooks()
        
        if not notebooks:
            logger.warning("未找到任何 Jupyter notebook 文件")
            return
        
        logger.info(f"开始批量转换 {len(notebooks)} 个文件...")
        logger.info("=" * 60)
        
        for i, notebook_path in enumerate(notebooks, 1):
            logger.info(f"[{i}/{len(notebooks)}] 处理文件: {notebook_path}")
            
            success, error_msg = self.convert_notebook_to_pdf(notebook_path)
            
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
                self.errors.append((str(notebook_path), error_msg))
        
        # 输出转换结果统计
        self.print_summary()
    
    def print_summary(self) -> None:
        """打印转换结果摘要"""
        logger.info("=" * 60)
        logger.info("转换完成！")
        logger.info(f"成功转换: {self.success_count} 个文件")
        logger.info(f"转换失败: {self.error_count} 个文件")
        
        if self.errors:
            logger.error("\n失败的文件详情:")
            for file_path, error_msg in self.errors:
                logger.error(f"  - {file_path}: {error_msg}")
        
        logger.info(f"\n详细日志已保存到: jupyter_to_pdf.log")


def check_dependencies() -> bool:
    """检查必要的依赖是否已安装"""
    try:
        # 检查 jupyter 命令是否可用 (使用 python -m jupyter)
        result = subprocess.run([sys.executable, '-m', 'jupyter', '--version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("未找到 jupyter 命令，请先安装 jupyter")
            return False
            
        # 检查 nbconvert 是否可用 (直接使用 python -m nbconvert)
        result = subprocess.run([sys.executable, '-m', 'nbconvert', '--version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("未找到 nbconvert，请先安装 nbconvert")
            return False
            
        logger.info("依赖检查通过")
        return True
        
    except FileNotFoundError:
        logger.error("未找到 Python 解释器或相关模块")
        logger.error("安装命令: pip install jupyter nbconvert")
        return False


def main():
    """主函数"""
    # ==================== 配置参数 ====================
    # 在这里直接设置要转换的目录和选项
    target_directory = r"D:\code\python\deeplearning\homework6"  # 使用原始字符串避免转义问题
    verbose_mode = False    # 是否显示详细输出
    
    # 可选的其他目录示例：
    # target_directory = "homework4"                                    # 转换 homework4 目录（相对路径）
    # target_directory = r"d:\code\python\deeplearning\homework4"      # 转换指定完整路径（原始字符串）
    # target_directory = "d:/code/python/deeplearning/homework4"       # 使用正斜杠
    # target_directory = "homework5"                                    # 转换 homework5 目录
    # ================================================
    
    # 设置日志级别
    if verbose_mode:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 检查目录是否存在
    target_dir = Path(target_directory)
    if not target_dir.exists():
        logger.error(f"目录不存在: {target_dir}")
        sys.exit(1)
    
    if not target_dir.is_dir():
        logger.error(f"路径不是目录: {target_dir}")
        sys.exit(1)
    
    # 检查依赖
    if not check_dependencies():
        logger.error("依赖检查失败，请安装必要的依赖后重试")
        sys.exit(1)
    
    # 开始转换
    logger.info(f"Jupyter Notebook 批量转换工具")
    logger.info(f"目标目录: {target_dir.absolute()}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    converter = JupyterToPDFConverter(str(target_dir))
    converter.convert_all()


if __name__ == "__main__":
    main()