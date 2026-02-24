from .base_adaptor import BaseAdaptor
from .aime24_adaptor import AIME24Adaptor
from .aime25_adaptor import AIME25Adaptor
from .math500_adaptor import Math500Adaptor
from .teacher_traces_adaptor import TeacherTracesAdaptor


class AdaptorFactory:
    @staticmethod
    def create_adaptor(benchmark_type: str, data_path: str, thinking_mode: bool = False) -> BaseAdaptor:
        adaptor_map = {
            'aime24': AIME24Adaptor,
            'aime25': AIME25Adaptor,
            'math500': Math500Adaptor,
            'teacher_traces_12k': TeacherTracesAdaptor,
            'teacher-traces': TeacherTracesAdaptor,
        }
        
        adaptor_class = adaptor_map.get(benchmark_type.lower())
        if adaptor_class is None:
            raise ValueError(f"Unsupported benchmark type: {benchmark_type}. "
                           f"Supported types: {list(adaptor_map.keys())}")
        
        return adaptor_class(data_path, thinking_mode)
