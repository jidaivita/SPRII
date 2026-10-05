"""Evaluator-owned window support and training-pair legality.

All identifiers, relation labels and supports stay in the private sampler.
Only history payloads and legal future targets are projected to the model.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowSupport:
    episode_key: str
    first_token_frame: int
    token_count: int
    difference_policy: str = "strict"

    def raw_frames(self):
        if self.token_count < 1 or self.first_token_frame < 0:
            raise ValueError("invalid window bounds")
        if self.difference_policy not in ("strict", "legacy_previous"):
            raise ValueError("unknown difference policy")
        start = self.first_token_frame-(self.difference_policy == "legacy_previous")
        if start < 0:
            raise ValueError("legacy first difference needs an actual predecessor")
        return tuple(range(start, self.first_token_frame+self.token_count))


def validate_pair(donor, recipient, *, pairing_profile="Independent",
                  cross_paths=(("donor", "recipient"),)):
    """Validate actual observation support, including hidden predecessors.

    SameEp permits earlier donor -> later recipient only. The common A/Z
    training comparison also uses one directed cross for Independent pairs,
    preserving prediction counts, direction and objective weight across arms.
    """
    dframes, rframes = donor.raw_frames(), recipient.raw_frames()
    if pairing_profile not in ("Independent", "SameEp"):
        raise ValueError("unknown pairing profile")
    same = donor.episode_key == recipient.episode_key
    if pairing_profile == "Independent" and same:
        raise ValueError("Independent requires distinct episodes")
    if pairing_profile == "SameEp":
        if not same:
            raise ValueError("SameEp requires the same episode")
        if max(dframes) >= min(rframes):
            raise ValueError("SameEp donor support must be strictly before recipient support")
    if tuple(cross_paths) != (("donor", "recipient"),):
        raise ValueError("this comparison requires exactly one donor-to-recipient cross path")
    return dict(pairing_profile=pairing_profile, same_episode=same,
                donor_raw_support=dframes, recipient_raw_support=rframes,
                donor_raw_frames=len(dframes), recipient_raw_frames=len(rframes),
                cross_paths=tuple(cross_paths),
                donor_transitions=len(dframes)-1, recipient_transitions=len(rframes)-1)


def validate_relation(donor_metadata, recipient_metadata, relation):
    if donor_metadata["split"] != recipient_metadata["split"]:
        raise ValueError("training relation crosses data splits")
    shared = {"G1": ("m",), "G2": ("m", "gamma"), "G3": ("m", "gamma", "k")}
    varied = {"G1": ("gamma", "k"), "G2": ("k",), "G3": ()}
    if relation not in shared:
        raise ValueError("unknown physical relation")
    a, b = donor_metadata["theta"], recipient_metadata["theta"]
    if not all(a[k] == b[k] for k in shared[relation]):
        raise ValueError("declared shared factor does not match")
    if not all(a[k] != b[k] for k in varied[relation]):
        raise ValueError("relation control must actively vary its declared nonshared factors")
    return True
