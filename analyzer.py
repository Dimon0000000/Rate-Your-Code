import os
import fnmatch
from pathlib import Path
from flavors import get_analyzer_for_file

class CodeSommelier:
    # 默认忽略的目录和文件模式
    IGNORE_PATTERNS = {
        '.git', '.svn', '.hg', '.idea', '.vscode', 
        '__pycache__', 'node_modules', 'venv', '.env', 
        'dist', 'build', 'target', '*.egg-info',
        '*.min.js', '*.min.css', 'jquery*.js'
    }

    def __init__(self, project_path, target_language=None):
        self.root = Path(project_path)
        self.target_language = target_language.lower() if target_language else None
        self.results = []
        self.file_tree = []

    def taste(self):
        """开始品鉴流程"""
        if not self.root.exists():
            return False, "❌ 庄园入口未找到，请检查路径。"

        print(f"🍷 正在通过嗅觉辨识代码风味... (扫描: {self.root})")
        self._scan_and_analyze(self.root)
        return True, "品鉴完成"

    def _is_ignored(self, name):
        """检查文件或目录是否应该被忽略"""
        for pattern in self.IGNORE_PATTERNS:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    def _scan_and_analyze(self, current_path, prefix=""):
        """递归遍历目录，生成树并分析文件"""
        try:
            # 排序以保持输出美观
            items = sorted(os.listdir(current_path))
        except PermissionError:
            return

        # 过滤掉不需要的文件和目录
        filtered_items = [
            i for i in items 
            if not i.startswith('.') and not self._is_ignored(i)
        ]
        
        count = len(filtered_items)
        for i, item in enumerate(filtered_items):
            full_path = current_path / item
            is_last = (i == count - 1)
            connector = "└── " if is_last else "├── "
            
            # 1. 记录文件树结构
            self.file_tree.append(f"{prefix}{connector}{item}")

            if full_path.is_dir():
                new_prefix = prefix + ("    " if is_last else "│   ")
                self._scan_and_analyze(full_path, new_prefix)
            else:
                # 2. 核心分发逻辑：根据文件获取对应的分析器
                analyzer = get_analyzer_for_file(full_path, self.target_language)
                
                if analyzer:
                    # 如果找到了对应语言的分析器，开始分析
                    score_data = analyzer.analyze(full_path)
                    self.results.append(score_data)

    def get_file_tree_str(self):
        return "\n".join(self.file_tree)