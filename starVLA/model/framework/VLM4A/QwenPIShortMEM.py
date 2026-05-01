from typing import Optional

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenPIShortMEM")
class QwenPIShortMEM(Qwen_PI):
    """QwenPI variant that selects the ShortMEM VLM through framework.qwenvl.base_vlm."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
