import argparse
import sys
from analyzer import CodeSommelier
from reporter import MarkdownReporter

def main():
    # 1. 定义命令行参数
    parser = argparse.ArgumentParser(description="Code Sommelier - 代码优雅度评分工具 🍷")
    
    parser.add_argument(
        '--project_path', 
        type=str, 
        required=True, 
        help='需要评判的项目文件夹路径'
    )
    
    parser.add_argument(
        '--language', 
        type=str, 
        default=None, 
        choices=['python', 'cpp'],
        help='指定评判语言 (如 python, cpp)。不指定则分析所有支持的语言。'
    )

    args = parser.parse_args()

    # 2. 初始化品鉴师
    sommelier = CodeSommelier(args.project_path, args.language)

    # 3. 开始品鉴
    success, message = sommelier.taste()
    
    if not success:
        print(message)
        sys.exit(1)

    # 4. 生成报告
    reporter = MarkdownReporter()
    reporter.generate(sommelier.results, sommelier.get_file_tree_str())

if __name__ == "__main__":
    main()