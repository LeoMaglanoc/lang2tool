# SimToolReal task environment for Isaac Lab.
#
# The env_lab / env_lab_cfg modules depend on the installed Isaac Lab runtime
# (isaaclab.sim, isaaclab.envs, etc.).  Guard the import so that unit-test
# collection succeeds even when only the pure-torch submodules are needed.
# Callers in the full Isaac Lab context can import directly:
#   from simtoolreal_lab.tasks.simtoolreal.env_lab import SimToolRealEnv
try:
    from simtoolreal_lab.tasks.simtoolreal.env_lab import SimToolRealEnv
    from simtoolreal_lab.tasks.simtoolreal.env_lab_cfg import SimToolRealEnvCfg

    __all__ = ["SimToolRealEnv", "SimToolRealEnvCfg"]
except Exception:
    # Isaac Lab runtime not available or app context not initialized yet.
    __all__ = []
