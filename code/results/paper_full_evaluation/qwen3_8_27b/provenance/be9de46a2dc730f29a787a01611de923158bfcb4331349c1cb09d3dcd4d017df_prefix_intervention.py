"""Differentiate a causal suffix while caching its unchanged prefix with HF."""

import copy

import torch

from jlens.model import HiddenStateIntervention


def discard_recurrent_update(state, *args, **kwargs):
    # This cache is disposable after one forward. HF's in-place recurrent-state
    # update would overwrite a constant needed by autograd for the suffix.
    return state


class PrefixIntervention(HiddenStateIntervention):
    """Equivalent intervention using HF's ordinary prefix cache API.

    Tokens strictly before the subject position cannot depend on an intervention
    at that position. Their attention keys/values and recurrent states can thus
    be evaluated without autograd, retaining only the suffix derivative graph.
    """

    def __init__(self, model, final_block):
        super().__init__(model, final_block)
        self.building_prefix = False
        self.cached_ids = None
        self.cached_position = None
        self.prefix_cache = None

    def hook(self, module, inputs, output):
        if self.building_prefix:
            return output
        return super().hook(module, inputs, output)

    def forward(self, hidden, input_ids, position):
        if self.cached_ids is not input_ids or self.cached_position != position:
            self.building_prefix = True
            try:
                with torch.no_grad():
                    self.prefix_cache = (
                        self.model(
                            input_ids=input_ids[:, :position], use_cache=True
                        ).past_key_values
                        if position
                        else None
                    )
            finally:
                self.building_prefix = False
            self.cached_ids = input_ids
            self.cached_position = position
        self.hidden = hidden
        self.position = 0
        captured = []

        def capture(module, inputs, output):
            states = output[0] if isinstance(output, tuple) else output
            captured.append(states[0, -1])

        handle = self._final_block.register_forward_hook(capture)
        try:
            cache = copy.deepcopy(self.prefix_cache)
            if cache is not None:
                cache.update_recurrent_state = discard_recurrent_update
            self.model(
                input_ids=input_ids[:, position:],
                use_cache=True,
                past_key_values=cache,
            )
        finally:
            handle.remove()
        return captured[0]
