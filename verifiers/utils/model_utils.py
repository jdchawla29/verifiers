import importlib
import inspect
from importlib.util import find_spec
from importlib import import_module
from typing import Any, Callable

import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoProcessor,
    AutoConfig,
    PreTrainedModel,
)  # type: ignore


def accepts_logits_to_keep(model: PreTrainedModel) -> bool:
    """
    Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it.
    We inspect the model's forward method to determine this.

    Args:
        model: The model to check
        
    Returns:
        True if the model accepts logits_to_keep, False otherwise
    """
    forward = (
        model.get_base_model().forward
        if hasattr(model, "get_base_model")
        else model.forward
    )
    try:
        inspect.signature(forward).bind_partial(**{"logits_to_keep": None})
        return True
    except TypeError:
        return False


class _ForwardRedirection:
    """Implements the `forward-redirection`.

    Taken from Pytorch-lightning: https://github.com/Lightning-AI/pytorch-lightning/blob/02311d03fb982560246eead7c08104481fac9579/src/lightning/pytorch/strategies/strategy.py#L602

    A method call to a wrapped module gets rerouted through the wrapper's `forward` method instead.

    """

    def __call__(
        self,
        wrapper_module: nn.Module,
        original_module: nn.Module,
        method: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Reroutes a method call through the `wrapper_module`'s `forward` method.

        Args:
            wrapper_module: The module that has `original_module` wrapped.
            original_module: The module that was wrapped inside `wrapper_module`.
            method_name: The name of the method that should be called on the `original_module` after inputs get
                redirected through the `wrapper_module`'s `forward` method.
            *args: The positional arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.

        """
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            original_module.forward = original_forward  # type: ignore[method-assign]
            # Call the actual method e.g. `.training_step(...)`
            out = method(*_args, **_kwargs)
            self.on_after_inner_forward(wrapper_module, original_module)
            return out

        # Patch the original_module's forward so we can redirect the arguments back to the real method
        original_module.forward = wrapped_forward  # type: ignore[method-assign]

        wrapper_output = wrapper_module(*args, **kwargs)
        self.on_after_outer_forward(wrapper_module, original_module)
        return wrapper_output

    def on_after_inner_forward(
        self, wrapper_module: nn.Module, original_module: nn.Module
    ) -> None:
        pass

    def on_after_outer_forward(
        self, wrapper_module: nn.Module, original_module: nn.Module
    ) -> None:
        pass


def is_liger_available() -> bool:
    return find_spec("liger_kernel") is not None


def model_loader(model_id: str, **model_kwargs) -> PreTrainedModel:
    """Load a model using its architecture directly from transformers."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    architecture = getattr(transformers, config.architectures[0])
    return architecture.from_pretrained(
        model_id,
        trust_remote_code=True,
        **model_kwargs
    )

def get_model(
    model_name: str,
    use_liger: bool = True,
    liger_patch_suffix: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> Any:
    if model_kwargs is None:
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
    if is_liger_available() and use_liger:
        print("Using Liger kernel")
        try:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM  # type: ignore

            model = AutoLigerKernelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )
            return model
        except ValueError:  # try monkey patch
            print(
                f"Model {model_name} is not supported with AutoLigerKernelForCausalLM. Attempting monkey patch..."
            )
            if liger_patch_suffix is None:  # try with model tpe
                liger_patch_suffix = AutoConfig.from_pretrained(
                    model_name, trust_remote_code=True
                ).model_type
                print(
                    f"No liger_patch_suffix provided, attempting with model_type: {liger_patch_suffix}"
                )
            patch_func_name = f"apply_liger_kernel_to_{liger_patch_suffix}"
            ligermod = importlib.import_module("liger_kernel.transformers")
            patch_func = getattr(ligermod, patch_func_name, None)
            if callable(patch_func):
                patch_func()
                model = model_loader(model_name, **model_kwargs)
                print(f"Applied Liger-Kernel patch to {model_name}")
                return model
            else:
                raise ValueError(
                    f"Model {model_name} may not be supported with Liger-Kernel in verifiers. Check the Liger-Kernel documentation."
                )
    else:
        return model_loader(model_name, **model_kwargs)


def get_processor(model_name: str, padding_side: str = "left") -> Any:
    processor = AutoProcessor.from_pretrained(model_name, padding_side=padding_side)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if not hasattr(tokenizer, "chat_template"):
        raise ValueError(
            f"Tokenizer for model {model_name} does not have chat_template attribute, \
                            and could not find a tokenizer with the same name as the model with suffix \
                            '-Instruct'. Please provide a tokenizer with the chat_template attribute."
        )
    return processor


def get_model_and_processor(
    model_name: str,
    use_liger: bool = True,
    liger_patch_suffix: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    model = get_model(model_name, use_liger, liger_patch_suffix, model_kwargs)
    processor = get_processor(model_name)
    return model, processor


# alias
get_model_and_tokenizer = get_model_and_processor
