from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

@dataclass
class AnalysisResult:
    file_name: str
    language: str
    score: float
    rating: str  # S, A, B, C, D
    issues: List[str] = field(default_factory=list)

class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, file_path) -> AnalysisResult:
        pass

    def calculate_rating(self, score):
        if score >= 95: return "S (神品)"
        if score >= 85: return "A (珍藏)"
        if score >= 70: return "B (优良)"
        if score >= 50: return "C (餐酒)"
        return "D (劣质)"