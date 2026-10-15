"""Historical hardware names; normal modules import their owners directly."""
from aka.legacy.task.hardware import _hardware_token
from aka.legacy.task.hardware import hardware_vendor
from aka.legacy.task.hardware import supported_frameworks
from aka.legacy.task.hardware import _workspace_slug
from aka.legacy.task.hardware import framework_workspace_suffix
from aka.legacy.workflow.hardware import _is_triton_family
from aka.legacy.task.source.inspection import kernel_is_gluon
from aka.legacy.task.source.inspection import head_kernel_is_gluon
from aka.legacy.workflow.hardware import should_convert_to_gluon
from aka.legacy.workflow.hardware import hardware_directive
from aka.legacy.task.hardware import AMD_FRAMEWORKS
from aka.legacy.task.hardware import DEFAULT_FRAMEWORKS
from aka.legacy.task.hardware import NVIDIA_FRAMEWORKS
from aka.legacy.task.hardware import PPU_FRAMEWORKS
from aka.legacy.task.source.inspection import source_uses_gluon
