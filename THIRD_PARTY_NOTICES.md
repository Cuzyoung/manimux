# Third-party notices

## MolmoAct2

`src/manimux/integrations/molmoact_yam/` is derived from the YAM evaluation
example in `allenai/molmoact2` at commit
`3d15be665dec918d6b3fe0cf606284adc4677bba`. ManiMux reorganizes it as an
internal package and adds asynchronous rollout observation, viewer events,
runtime configuration, and shutdown behavior. The upstream project is under
Apache License 2.0; see `licenses/MolmoAct2-APACHE-2.0.txt`.

## i2rt YAM model assets

`src/manimux/assets/i2rt/robot_models/` contains the YAM and linear_4310 model
geometry from `i2rt-robotics/i2rt` at commit
`5d47b358bafb30c65e397f2ece506550a0db4594`. These assets are used only for
viewer rendering and forward kinematics. The upstream project is under the MIT
License; see `licenses/i2rt-MIT.txt`.
