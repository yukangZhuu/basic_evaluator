from .base_adaptor import BaseAdaptor
from .gsm8k_adaptor import GSM8KAdaptor
from .math500_adaptor import Math500Adaptor
from .mmlu_adaptor import MMLUAdaptor
from .teacher_traces_adaptor import TeacherTracesAdaptor


class AdaptorFactory:
    @staticmethod
    def create_adaptor(benchmark_type: str, data_path: str, thinking_mode: bool = False) -> BaseAdaptor:
        adaptor_map = {
            'gsm8k': GSM8KAdaptor,
            'math-500': Math500Adaptor,
            'math500': Math500Adaptor,
            'mmlu': MMLUAdaptor,
            'teacher_traces_12k': TeacherTracesAdaptor,
            'teacher-traces': TeacherTracesAdaptor,
        }
        
        adaptor_class = adaptor_map.get(benchmark_type.lower())
        if adaptor_class is None:
            raise ValueError(f"Unsupported benchmark type: {benchmark_type}. "
                           f"Supported types: {list(adaptor_map.keys())}")
        
        return adaptor_class(data_path, thinking_mode)
