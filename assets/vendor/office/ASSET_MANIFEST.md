# Office background assets

Vendored files make the scene runnable without network access.

| Object | Source | License | Scene mesh |
|---|---|---|---|
| Desk chair | [Kenney Furniture Kit](https://kenney.nl/assets/furniture-kit), `Models/OBJ format/chairDesk.obj` | CC0; see `KENNEY_LICENSE.txt` | `chair_frame.obj`, `chair_cushion.obj` |
| Office desk | [Poly Haven Metal Office Desk](https://polyhaven.com/a/metal_office_desk), 1K glTF | [CC0](https://polyhaven.com/license) | `desk/metal_office_desk_1k.gltf` |
| Potted plant | [Poly Haven Potted Plant 02](https://polyhaven.com/a/potted_plant_02), 1K glTF | [CC0](https://polyhaven.com/license) | `plant/potted_plant_02_{pot,leaves}.gltf` |

The original Kenney ZIP and GLB are retained. `scripts/prepare_office_chair.py` derives the two chair OBJ meshes from the ZIP by selecting faces by material. This preserves the OBJ vertex coordinates and avoids the node offset observed when its GLB was loaded as a single MJCF mesh. The original Poly Haven glTF files, buffers, and JPEG textures are retained; local pot and leaves derivatives select their respective meshes so they can be positioned and colored separately. The leaves node translation is applied in the scene config. All background geoms have collisions disabled.

| File | SHA-256 |
|---|---|
| `KENNEY_LICENSE.txt` | `bc0de1a0742cb490f9ed4bae0bd284286ebb27e23149b9917c7d8c0d590b8b29` |
| `chair_cushion.obj` | `078c3b63abebf0783f12df0c2b8445fef997e16f2db5b3616c724026eb5de624` |
| `chair_desk.glb` | `46406619186034cbe92b19b79a5f1a8e3f442a17a70a7524ef2c0ad0f35095c6` |
| `chair_frame.obj` | `25017286b89b404de5e94da297d3b223920851278d6b6653108671e74013c40f` |
| `desk/metal_office_desk.bin` | `65b558f41a75ad4822e31cc2a97eb34ed359199279a6bfbd650deb27cef78767` |
| `desk/metal_office_desk_1k.gltf` | `f07032a5ee89aacceebf10fd9fa192c22775ae81861cfb0bdb1594497179373c` |
| `desk/textures/metal_office_desk_arm_1k.jpg` | `752916dd9bf8a163dc7e0d3673900791ee6d558bbe23894263ac92b42387f51b` |
| `desk/textures/metal_office_desk_diff_1k.jpg` | `1f02e3e77e8f82a3600664c991bd45c1c187452d61e9ede4336c6f90f7fa3139` |
| `desk/textures/metal_office_desk_nor_gl_1k.jpg` | `f57eb004937e46392151b803e0cb1d88983958d01ad11d93df78a82cfcc6c407` |
| `kenney_furniture-kit.zip` | `e67652d0932cee41683f74711c03d3e192a2af9979ef8e6b237711f5482d46b0` |
| `plant/potted_plant_02.bin` | `9c9812482c78a59789ed32fa842b4cbd9365e09c900c31ebb998e5efe627fe97` |
| `plant/potted_plant_02_1k.gltf` | `c533dcce0d1aa4379eb7db04601a04983f8df2422b28578de922655a416e64a9` |
| `plant/potted_plant_02_leaves.gltf` | `fa0e4b8ec6c6ece6c7762956212bfa58a570a0fc1d2eed5110fa21d8fc7e12b6` |
| `plant/potted_plant_02_pot.gltf` | `9ee432dcbf2891ecc00a76acb8e0eb98c30960197b8244a536af92fd217bede6` |
| `plant/textures/potted_plant_02_leaves_diff_1k.jpg` | `3643dce6e6c0a9f40129bd837ec1a11802e8f43ea43cb29e58219d82e0ac632b` |
| `plant/textures/potted_plant_02_leaves_nor_gl_1k.jpg` | `e0fe090ff7a511729c9bbe59df7288b400430fdab178529443dec75e0f06007d` |
| `plant/textures/potted_plant_02_leaves_rough_1k.jpg` | `ac8959f70c0e91e77f3cd6fd3f73578b8aed64ac6c33fc34ac027a4e88276960` |
| `plant/textures/potted_plant_02_pot_diff_1k.jpg` | `684949bf24821d7e951d46915366b5a388d2e4ccaaa9d5284e718c83604705e9` |
| `plant/textures/potted_plant_02_pot_nor_gl_1k.jpg` | `981a68721c8b7b9ac26b96d0210334599d5e8f529362f505082c071a0a58e86c` |
| `plant/textures/potted_plant_02_pot_rough_1k.jpg` | `4e92ad8ee0f098050785d9a025e2c5a7d55bc7a7f7e1916cc7627fc7824a938f` |
