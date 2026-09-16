"""Shared OLMo hidden-state interventions for double-backward Jacobian products.

This module is imported only by GPU commands. Layer numbers refer to the
Transformers hidden-state tuple: state 17 is the output of transformer block 16.
"""

import torch


class HiddenStateIntervention:
    """Replace one subject state and differentiate the final prompt state.

    Register ``hook`` on a transformer block only after caching unmodified
    states. Remove the returned hook handle before any unmodified forward pass.
    """

    def __init__(self, model):
        self.model = model
        self.hidden = None
        self.position = None

    def hook(self, module, inputs, output):
        """Replace a single hidden state without modifying the original tensor."""
        if self.hidden is None or self.position is None:
            raise RuntimeError(
                "Set the intervention state before a hooked forward pass"
            )
        states = output[0] if isinstance(output, tuple) else output
        states = states.clone()
        states[0, self.position, :] = self.hidden
        return (states, *output[1:]) if isinstance(output, tuple) else states

    def forward(self, hidden, input_ids, position):
        """Map an intervened subject state to the final prompt state."""
        self.hidden = hidden
        self.position = position
        return self.model(input_ids=input_ids, output_hidden_states=True).hidden_states[
            -1
        ][0, -1, :]

    def jvp(self, reference, direction, input_ids, position):
        """Compute J(reference) @ direction with the original double backward."""
        hidden = reference.clone().requires_grad_(True)
        output = self.forward(hidden, input_ids, position)
        cotangent = torch.zeros_like(output, requires_grad=True)
        (gradient,) = torch.autograd.grad(
            output, hidden, grad_outputs=cotangent, create_graph=True
        )
        (product,) = torch.autograd.grad(gradient, cotangent, grad_outputs=direction)
        return product.detach()
