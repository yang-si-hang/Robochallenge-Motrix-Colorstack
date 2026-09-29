# Vendor asset manifest

Source: https://github.com/google-deepmind/mujoco_menagerie
Commit: `c96a32d28fb5da84da38c1da4d749e7a13212855`

Both directories retain their original `LICENSE` files. Vendored source files are unchanged. `scripts/build_scene.py` creates the combined XML separately, prefixes Robotiq names and references, relocates mesh paths, mounts the gripper, removes the arm-only keyframe, and adds table, blocks, and cameras.

| File | SHA-256 |
|---|---|
| `universal_robots_ur5e/CHANGELOG.md` | `ef67f701df8698fb2e1a77a5de5d9a39aa22e7b4b101f68fc311d34e3ace87ed` |
| `universal_robots_ur5e/LICENSE` | `5ec71ccf66c8d03261448f2441a586765b97f2248c860a1ae19689fb1c45cee6` |
| `universal_robots_ur5e/README.md` | `435600d4f00cfdbd7edec4b60f0cd7fbb539f39abaefcd8bb3994beed3b1156a` |
| `universal_robots_ur5e/assets/base_0.obj` | `9a3b21f1524baef6af8a52d7af90b6e94b79f7e3c2b04a0f6581f74d22291b0b` |
| `universal_robots_ur5e/assets/base_1.obj` | `cf7d54691a2c1eeda280955ebd6bd83c2dbecfc46fb53c10198f05d107511b50` |
| `universal_robots_ur5e/assets/forearm_0.obj` | `64c049a55149c2ab355594d52302b53d3e34da3677ceaa1917c427cde7743146` |
| `universal_robots_ur5e/assets/forearm_1.obj` | `a96fe203015cf532b193f414cd01268c7c36795e51a837dd33afd122334f81f2` |
| `universal_robots_ur5e/assets/forearm_2.obj` | `05b436505fc595aeab26675590c2b56b88a65b2d52d6352096fed9f07cb4c4ca` |
| `universal_robots_ur5e/assets/forearm_3.obj` | `fa31af19cfad7a765b62e8a2d5931cb014b88de84195a7850b4e5ee902112e9a` |
| `universal_robots_ur5e/assets/shoulder_0.obj` | `477387a7e7312782a6040484a3c2c2fcee33f40109f419076e31b6f6e1692c68` |
| `universal_robots_ur5e/assets/shoulder_1.obj` | `8537bba02804640ddf6e260f79fe66e895b3229ed3d44298a95e8e027eb5d435` |
| `universal_robots_ur5e/assets/shoulder_2.obj` | `291714518416bb8f0bafbf123867db011ecbf5146a05245ca5893ee8c57d4e21` |
| `universal_robots_ur5e/assets/upperarm_0.obj` | `14143d4f1175a7dbacc8e2f2f98f1cac5a2b2e238cd46cd03654d31f2610eb53` |
| `universal_robots_ur5e/assets/upperarm_1.obj` | `272899e927f4a32b7c25e34560ec70654c0346cc8ed1c616c0555f3f53b32d14` |
| `universal_robots_ur5e/assets/upperarm_2.obj` | `86c05350d60eb0e7e28a81b544267478b7db997baba41cef19088ebbeffecfa1` |
| `universal_robots_ur5e/assets/upperarm_3.obj` | `b257dab441e6fab64ea03b4b87029d8a96554ad1a513cce944d56acba0ad948a` |
| `universal_robots_ur5e/assets/wrist1_0.obj` | `7ef0c59b5c5d56a9902435c9a058b27d2a38c85fe2bae02d00a9284dd1f4d519` |
| `universal_robots_ur5e/assets/wrist1_1.obj` | `ee6d5c562e6becef94109b0543329080d285d1aef521a01ed98e18a5bf1e8684` |
| `universal_robots_ur5e/assets/wrist1_2.obj` | `88c66dc64f15391287147d50add37729a3f41dadd80b1f859cf2cb6d45daf32a` |
| `universal_robots_ur5e/assets/wrist2_0.obj` | `2ac2e2e39d8f7005bc3b3ab26cd61634be2e14930074615da5318b9ee7c569a8` |
| `universal_robots_ur5e/assets/wrist2_1.obj` | `a721d06dc61d3b3a8ec58c035c9e3358ed30d6991a72e077d8abf3c3c203b633` |
| `universal_robots_ur5e/assets/wrist2_2.obj` | `6df4335c8bd5e881fda2c224f770115ffe9ddd0cd321ebe3c0a4a310796f0987` |
| `universal_robots_ur5e/assets/wrist3.obj` | `0e51748f9c93bd66604360a5cf7e2e8a6e754550be64076333ff4fc19ec90c61` |
| `universal_robots_ur5e/scene.xml` | `e877852d40095926ff7db9cfb85e595b550e03da398bd6f6f58885d39038d1b6` |
| `universal_robots_ur5e/ur5e.png` | `90310086f1187fcda94252ff68f8c4f8be97f65bec8f4eea449179b0979da795` |
| `universal_robots_ur5e/ur5e.xml` | `ffd0a7b336d3c2d27c35a15cc0d4aaa95196769aca190ac8d6522d3318c345a0` |
| `robotiq_2f85/2f85.png` | `cce07123183ffe562768aadcdbaa487336c0c2a485c2a26c47697750630bc912` |
| `robotiq_2f85/2f85.xml` | `d48aca5f9151798ffd38111ce4e8b2081f3ec2d4f525161b33643451580010de` |
| `robotiq_2f85/CHANGELOG.md` | `38082db6de93a95ef85a319c2a8cac8b1dbcff597a718d8cdf843c339e7b23a6` |
| `robotiq_2f85/LICENSE` | `3d5edd18d1e22a17666e9bb84270171ed5ac90b192d24e0177fd171708ea2316` |
| `robotiq_2f85/README.md` | `a10f34027d417f8f7b9102b74b42976031f0246f3090444728ab322bc4632aaf` |
| `robotiq_2f85/assets/base.stl` | `1c7b9f2bd92d705fc4e897c94e905973a3c05f406845e942229433deb7041453` |
| `robotiq_2f85/assets/base_mount.stl` | `87971dfdd6e5311b5a723e7b73c4701ebf5c2364df671500f30dfc1c53c8e61a` |
| `robotiq_2f85/assets/coupler.stl` | `d5ee95e62f8415bf5b6e503c831a958f5fc1990bf9b2865329ec38a28932727c` |
| `robotiq_2f85/assets/driver.stl` | `bbd6e868b1778bead60d2516c7ede581d7ad4e431744b1828757d6ea5129c112` |
| `robotiq_2f85/assets/follower.stl` | `633793332c081641ce200df27a40643cc293b29956e3cb5cb29cb811c33ef1c7` |
| `robotiq_2f85/assets/pad.stl` | `4af1fa8d9bb285abecbf5dfc88e74cb30f059a7a074c2030dc35e9ee4316019d` |
| `robotiq_2f85/assets/silicone_pad.stl` | `ca20c0fa61e6d3ce7b04bed25360e048e31dfd88ace04187aa6d9e8f4ca3fd0f` |
| `robotiq_2f85/assets/spring_link.stl` | `4fb74a8a0d76c0e471cf19fd48bc676fd5b19123798e7a39eb6aa56869354283` |
| `robotiq_2f85/scene.xml` | `dbc7255f3e1a5539e59cffac3000a49f19aa3cae384061aa74ec787e6a5760ff` |
