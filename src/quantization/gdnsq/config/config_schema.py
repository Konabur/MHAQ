from pydantic import BaseModel

from typing import Optional

class CurriculumConfig(BaseModel):
    enable: bool = True
    mean: float = 0.9
    std: float = 0.5


class GDNSQQuantizerParams(BaseModel):
    distillation: Optional[bool] = False    
    distillation_loss: Optional[str] = "Cross-Entropy"
    distillation_teacher: Optional[str] = None
    qnmethod: str = "STE"
    curriculum: CurriculumConfig = CurriculumConfig()

