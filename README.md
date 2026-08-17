**Note:** This projection uses NVDiffrast for rasterization, baking, etc., which will be replaced by DRTK (MIT) in the future.This project also uses third-party components with non-commercial license: Tencent Hunyuan3D-2.1 rendering code. and texkit code from [lumitex](https://github.com/LumiTexPBR/LumiTex).

### Troubleshooting

* **Missing / Incorrect Wheels:** If you have an issue where the custom index is not able to get the correct wheel, you can download them directly from [this GitHub release](https://github.com/PozzettiAndrea/cuda-wheels/releases/tag/nvdiffrast-latest).
* **NumPy Compatibility Error:** If you get the error `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.6 as it may crash`, you can solve it by downgrading NumPy to a `1.x` version. Run the following command:
  ```bash
  pip install "numpy<2"
