# External implementation notices

## Neural Operator Discovery

The NOD-derived clean trainer, adapted loader and FHN port in `benchmarks/nod/` build on **Zituo Chen, Neural Operator Discovery Code, version 1**, [DOI 10.5281/zenodo.20406332](https://doi.org/10.5281/zenodo.20406332). The upstream release is CC BY 4.0. The Burgers dataset is [DOI 10.5281/zenodo.20372988](https://doi.org/10.5281/zenodo.20372988), also CC BY 4.0. See the [license text](https://creativecommons.org/licenses/by/4.0/legalcode). Changes include SPRII losses, equal-access sampling, training/selection separation, standalone evaluation, path normalization and Python FHN interfaces. No endorsement by the upstream creator is implied. Upstream-derived portions are not relicensed by the repository's MIT license.

## CoDA

CoDA is fixed at revision `17b73521394f2a5986e5418c32ad2965c97cd8c0` of [yuan-yin/CoDA](https://github.com/yuan-yin/CoDA). Its MIT copyright and permission text is preserved in `licenses/CoDA-MIT.txt`. The adapters load the numerical definitions from that dependency and document the singleton-axis Burgers adaptation explicitly.

## CaDM and GEPS

[CaDM](https://github.com/younggyoseo/CaDM) is pinned to `38c11a58d959bfd597f9323e58f28b17f6bf4fd9`. [GEPS](https://github.com/itsakk/geps) is pinned to `e9a865218ecffacb7007ac7d719f3741afcf8c02`. Neither inspected snapshot contains a root LICENSE file. The full upstream source trees are not redistributed; fetch them from the public upstream repositories. This release supplies its own environment adapters, interfaces, evaluation and experimental-control implementations. Their upstream origin is cited here without claiming new permission over upstream code.

## Gym reference

The Pendulum environment equivalence check reads [Gym 0.16.0 Pendulum](https://github.com/openai/gym/blob/0.16.0/gym/envs/classic_control/pendulum.py), fetched separately under its upstream MIT license.
