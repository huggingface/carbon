"""Compatibility patches for GENERator remote-code models.

GENERator's trusted remote code expects a few tokenizer/model internals from
older Transformers releases. Keep these patches scoped here so evaluation
scripts can run against the repository-level Transformers version.
"""


def patch_legacy_tokenizer_base():
    """Restore tokenizer attributes expected by GENERator's tokenizer code."""
    from transformers import PreTrainedTokenizerBase

    PreTrainedTokenizerBase._special_tokens_map = {}
    PreTrainedTokenizerBase._added_tokens_decoder = {}
    PreTrainedTokenizerBase._added_tokens_encoder = {}
    PreTrainedTokenizerBase.verbose = False


def patch_generator_sample(model):
    """Adapt GENERator's custom `_sample` signature to current `generate()`."""
    cls = model.__class__
    if cls.__name__ != "GENERatorForCausalLM" or getattr(cls, "_carbon_sample_patched", False):
        return

    original_sample = cls._sample

    def _sample(
        self,
        input_ids,
        logits_processor,
        stopping_criteria,
        generation_config,
        synced_gpus=False,
        streamer=None,
        **model_kwargs,
    ):
        return original_sample(
            self,
            input_ids,
            logits_processor,
            stopping_criteria,
            generation_config,
            synced_gpus,
            streamer,
            **model_kwargs,
        )

    cls._sample = _sample
    cls._carbon_sample_patched = True
