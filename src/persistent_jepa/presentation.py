"""Unified human-facing renderer for the Paper A environment suite.

Native simulators continue to produce their original observations. Pass a state
for Spring/Poke/D-Clean, native RGB for external simulators and RH20T, or a
16-by-time tactile array for Baxter. See sprii_visuals for exact contracts.
"""
from sprii_visuals import draw_scene, render_rgb, ENVIRONMENTS
__all__ = ["draw_scene", "render_rgb", "ENVIRONMENTS"]
