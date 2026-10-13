# Panda model attribution

The robot description and collision meshes are derived from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/0059d4335f8156206f63a35662313385f7ad6d74/franka_emika_panda),
commit `0059d4335f8156206f63a35662313385f7ad6d74`, under the included Apache-2.0
[license](LICENSE). Menagerie's description derives from the Franka Emika Panda
URDF in [franka_ros](https://github.com/frankaemika/franka_ros).

This variant uses `panda_nohand.xml`: seven arm joints, fixed base, no gripper.
Position servos are replaced with unit-gear torque motors, with limits of
87 N m for joints 1 to 4 and 12 N m for joints 5 to 7. Keyframe controls are
zero torque. Inertias, kinematics, passive joint damping, armature, joint
ranges and collision meshes are retained. Detailed visual meshes are omitted;
collision meshes are coloured for display. The floor is visual only.

`source.json` records the upstream XML hash and the hashes of the packaged
variant, collision meshes and license. Loading verifies these files.
