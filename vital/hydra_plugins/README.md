# Hydra Plugins

Hydra allows the use of external plugins. This directory contains useful plugings for projects using vital.

[Hydra plugin development documentation](https://hydra.cc/docs/advanced/plugins/develop)

## Sweeper

[Example sweeper plugin](https://github.com/facebookresearch/hydra/tree/main/examples/plugins/example_sweeper_plugin)

### Config Sweeper

As of Hydra 1.1, Hydra's basic sweeper does not allow defining sweeps in configs. This slightly modified Sweeper allows defining sweeps
in the config by specifying them in the hydra/sweeper/overrides.

This problem is brought up in this [issue](https://github.com/facebookresearch/hydra/issues/1376) and a fix is proposed
in this [PR](https://github.com/facebookresearch/hydra/pull/1801). The PR was refused but the code is used in this repo.


For example:
```yaml
hydra:
  sweeper:
    overrides:
      system/module: enet,unet
      system.module.dropout: 0, 0.1, 0.5
```
Running this config with the `-m` argument will run 6 experiments.
# SearchPath

## Vital
This file allows other projects to access vital configs without specifying the search path in each primary config.

This replaces these lines in the primary configs:
 ```yaml
hydra:
  searchpath:
    - pkg://vital.config
```
[Hydra SearchPath documentation](https://hydra.cc/docs/advanced/search_path/)

[Hydra SearchPath example](https://github.com/facebookresearch/hydra/tree/main/examples/plugins/example_searchpath_plugin)
