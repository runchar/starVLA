from typing import Optional

from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("QwenOFTShortMEM")
class QwenOFTShortMEM(Qwenvl_OFT):
    """QwenOFT variant that selects the ShortMEM VLM through framework.qwenvl.base_vlm."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
