# Attribution and third-party dependencies

This is an independent reproduction of RL² (Duan et al., 2016), using PPO rather than the original paper's TRPO.

The bandit/tabular implementation invokes Lucas Lingle's `pytorch_rl2` at commit `51afa4b6c6d05b469c7f74076b9490ae0809b289`. Local recurrent/device adapters and visual recurrent modules follow details of that reference implementation. See https://github.com/lucaslingle/pytorch_rl2 . The checkout is obtained separately; its source and history are not vendored here. The root MIT license applies to original contributions; it does not override third-party rights. No blanket relicensing of upstream-derived code is asserted. Consult the relevant upstream authors/terms before redistribution or use requiring a license grant.

ViZDoom: https://github.com/Farama-Foundation/ViZDoom (1.3.2).
Freedoom textures are extracted locally from ViZDoom's bundled `freedoom2.wad`; the Freedoom notice is preserved under `licenses/`. No game assets are distributed here.
ACC (1.60): https://github.com/ZDoom/acc/releases/tag/1.60 . Obtain the compiler and include files separately; no compiler binaries are distributed here.

Paper: https://arxiv.org/abs/1611.02779 . Architecture details: Yan Duan's dissertation, https://escholarship.org/uc/item/20z5k8dq .
