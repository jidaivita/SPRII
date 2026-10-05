"""Ingest-time A representations with the original frozen pixel encoders.

This is an evaluation bridge, not a trained eight-dimensional state head.
Split models retain their native 64d persistent code; B0 retains its native
128d monolithic history code. B0 needs a separately fitted joint-access head
for donor-conditioned prediction, so it cannot use the split latent predictor.
"""
import numpy as np
from .schema import Config
from .observations import image_history, public_query_images

VARIANTS = ('B0', 'B0_split', 'B2', 'Bx', 'B3')
QUERY_TARGETS = ('joint_state_delta_8d', 'cold_rest_joint_state_delta_8d', 'a_observation_latent_128d')
REPRESENTATION_TARGET = 'a_retained_history_representation_v1'


def _zero_history(length):
    """Legacy development Null policy; not an optimal independently fitted Null."""
    mask = np.ones(length, bool); mask[0] = False
    return dict(observations=np.zeros((length, 2, 64, 64), np.float32),
        past_actions=np.zeros((length, 2), np.float32), past_action_mask=mask,
        relative_times=np.arange(length, dtype=np.float64) * .05)


class CompressedAHistoryAgent:
    """Representation lifecycle shared by monolithic and split A backbones.

    Default single-history/last-code behavior matches the old donor bridge.
    Multiple histories and mean pooling require explicit configuration; pooling
    is a declared evaluation rule, not a claim that A was trained for composition.
    """
    def __init__(self, model, *, max_histories=1, aggregation='last', no_history_policy='encoded_zero_history'):
        if type(max_histories) is not int or not 1 <= max_histories <= 8:
            raise ValueError('explicit bounded A history count required')
        if aggregation not in ('last', 'mean') or no_history_policy not in ('encoded_zero_history', 'zero_code'):
            raise ValueError('unregistered A memory or no-history policy')
        self.model = model; self.max_histories = max_histories
        self.aggregation = aggregation; self.no_history_policy = no_history_policy
        self.config = Config(resolution=128)
        self._ready = False; self._code = None; self.count = 0; self.episode_token = None

    def _descriptor(self):
        cfg = self.model.cfg
        return (self.model.variant, cfg.history_length, cfg.observation_dim, cfg.persistent_dim,
                getattr(self.model, 'normalization_profile', 'legacy_eval_running_statistics'))

    def _stamp(self):
        return tuple((kind, name, id(value), value._version, str(value.dtype), str(value.device))
                     for kind, items in (('parameter', self.model.named_parameters()), ('buffer', self.model.named_buffers()))
                     for name, value in items)

    def _check_pending_statistics(self):
        observation = self.model.observation
        if getattr(observation, '_snapshot', None) is not None or getattr(observation, '_history_count', 0):
            raise ValueError('finish or discard the A training-statistics step before evaluation')

    def _guard(self):
        if not self._ready:
            raise RuntimeError('initialize the A evaluation lifecycle first')
        self._check_pending_statistics()
        if any(m.training for m in self.model.modules()) or self._descriptor() != self._architecture or self._stamp() != self._artifact_stamp:
            raise ValueError('A frozen evaluation artifact or mode changed; retained code is stale')

    def _encode(self, observations, actions):
        import torch
        device = next(self.model.parameters()).device
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            images = torch.tensor(np.asarray(observations)[None], dtype=torch.float32, device=device)
            past = torch.tensor(np.asarray(actions)[None], dtype=torch.float32, device=device)
            embeddings = self.model.observation(images)
            _, persistent, context = self.model.codes(embeddings, past)
            code = context if self.model.variant == 'B0' else persistent
            if code is None or tuple(code.shape) != (1, self.representation_dim) or not torch.isfinite(code).all():
                raise ValueError('A history encoder returned an invalid native code')
            return code.detach().clone()

    def initialize(self, context):
        import torch
        from persistbench.contracts import CapabilityDeclaration, OutputType
        self._ready = False
        self._check_pending_statistics()
        variant = self.model.variant; cfg = self.model.cfg
        if variant not in VARIANTS or cfg.observation_dim != 128 or cfg.persistent_dim != 64 or cfg.history_length not in (24, 48, 96):
            raise ValueError('unsupported A pixel architecture or registered history length')
        if any(p.is_floating_point() and p.dtype != torch.float32
               for p in (*self.model.parameters(), *self.model.buffers())):
            raise ValueError('A bridge requires the declared float32 inference profile')
        self.model.eval()
        self.representation_dim = 128 if variant == 'B0' else 64
        self.count = 0; self.episode_token = None; self._code = None
        self._architecture = self._descriptor(); self._artifact_stamp = self._stamp()
        if self.no_history_policy == 'encoded_zero_history':
            empty = _zero_history(cfg.history_length)
            self._code = self._encode(empty['observations'], empty['past_actions'][1:])
        else:
            self._code = torch.zeros((1, self.representation_dim), device=next(self.model.parameters()).device)
        self._ready = True; self._guard()
        return CapabilityDeclaration((OutputType.REPRESENTATION,), representation_dim=self.representation_dim,
                                     output_keys=('representation',))

    def reset(self, context):
        self._guard()
        # Fresh interaction boundary does not discard the permitted donor memory.
        self.episode_token = context.episode_token

    def ingest(self, experience):
        self._guard()
        if self.count >= self.max_histories:
            raise ValueError('registered A donor count exceeded; initialize a new independent condition')
        metadata = {'observation_schema': 'strict_image_difference_v1', 'action_schema': 'previous_action_with_missing_first'}
        if experience.metadata != metadata:
            raise ValueError('non-public or unregistered A history metadata')
        _, actions = image_history(experience, self.config)
        if len(experience.observations) != self.model.cfg.history_length:
            raise ValueError('A donor window differs from the frozen position-embedding length')
        code = self._encode(experience.observations, actions)
        self._guard()
        if self.aggregation == 'mean' and self.count:
            self._code = self._code + code
        else:
            self._code = code
        self.count += 1

    def history_code(self):
        self._guard()
        code = self._code / self.count if self.aggregation == 'mean' and self.count else self._code
        return code.detach().clone()

    def mutable_state_bytes(self):
        self._guard()
        return self._code.numel() * self._code.element_size() + 8

    def diagnostics(self):
        self._guard()
        return dict(raw_history_retained=False, processed_histories=self.count,
            persistent_numeric_payload_bytes=self.mutable_state_bytes(), representation_dim=self.representation_dim,
            representation_kind='monolithic_history' if self.model.variant == 'B0' else 'persistent_branch',
            aggregation=self.aggregation, max_histories=self.max_histories, no_history_policy=self.no_history_policy,
            history_frames=self.model.cfg.history_length, strict_A_compatible_window=self.model.cfg.history_length == 24,
            normalization_profile=self._architecture[-1], artifact_stamp_entries=len(self._artifact_stamp),
            memory_accounting='retained float32 code and int64 count; excludes frozen weights, artifact-audit metadata and framework/container overhead',
            formal_results=False)

    def respond(self, query):
        from persistbench.contracts import AgentOutput, OutputType
        self._guard(); query.validate()
        payload = query.observations
        if (not isinstance(payload, dict) or set(payload) != {'target_spec'} or
                not isinstance(payload['target_spec'], str) or payload['target_spec'] != REPRESENTATION_TARGET or
                query.actions is not None):
            raise ValueError('representation request cannot include targets, private labels or extra query fields')
        return AgentOutput(OutputType.REPRESENTATION, {'representation': self.history_code().cpu().numpy()[0].copy()}, self.diagnostics())

    def _query_tensors(self, query):
        import torch
        self._guard()
        images, actions = public_query_images(query, self.config, allowed_targets=QUERY_TARGETS)
        if len(images) not in (1, 2) or len(actions) not in (1, 4, 16):
            raise ValueError('A latent/readout bridge requires q=0/1 and h=1/4/16; longer profiles need a declared query architecture')
        device = next(self.model.parameters()).device
        padded = torch.zeros((1, 16, 2), dtype=torch.float32, device=device)
        mask = torch.zeros((1, 16), dtype=torch.float32, device=device)
        padded[0, :len(actions)] = torch.tensor(actions, device=device); mask[0, :len(actions)] = 1
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            query_embedding = self.model.observation(torch.tensor(images[None], dtype=torch.float32, device=device))[:, -1]
        self._guard()
        return query_embedding.detach(), padded, mask, torch.tensor([(1, 4, 16).index(len(actions))], device=device)

    def readout_inputs(self, query):
        """Frozen public features for a future independently fitted state head.

        Query CNN features are 128d for both architectures. Donor features retain
        their native 128d/64d widths; equal readout capacity must be audited when
        selecting heads. This function neither fits nor invents such a head.
        """
        q, actions, mask, horizon = self._query_tensors(query)
        return dict(query_observation=q.cpu().numpy()[0].copy(), retained_history=self.history_code().cpu().numpy()[0].copy(),
            history_present=bool(self.count), future_actions=actions.cpu().numpy()[0].copy(),
            action_mask=mask.cpu().numpy()[0].copy(), horizon_index=int(horizon.item()))


class PixelALatentAgent(CompressedAHistoryAgent):
    """Original split A latent predictor, now with ingest-time donor encoding."""
    def initialize(self, context):
        from persistbench.contracts import CapabilityDeclaration, OutputType
        if self.model.variant == 'B0':
            raise ValueError('B0 has no split donor latent predictor; use its native representation and a separately fitted joint-access head')
        super().initialize(context)
        return CapabilityDeclaration((OutputType.PREDICTION, OutputType.REPRESENTATION),
            representation_dim=self.representation_dim, output_keys=('latent', 'representation'))

    def respond(self, query):
        import torch
        from persistbench.contracts import AgentOutput, OutputType
        if isinstance(query.observations, dict) and query.observations.get('target_spec') == REPRESENTATION_TARGET:
            return super().respond(query)
        q, actions, mask, horizon = self._query_tensors(query)
        with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
            transient = self.model.transient(q)
            prediction = self.model.predictor(torch.cat((transient, self.history_code()), dim=-1), actions, mask, horizon)
        if tuple(prediction.shape) != (1, 128) or not torch.isfinite(prediction).all():
            raise ValueError('invalid A observation-latent prediction')
        self._guard()
        return AgentOutput(OutputType.PREDICTION, {'latent': prediction.detach().cpu().numpy()[0].copy()},
            dict(self.diagnostics(), latent_only=True, joint_state_head_fitted=False))
