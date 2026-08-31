import jax
import jax.numpy as jnp
import numpy as np


def apply_random_terrain(
    env,
    seed: int = 0,
    min_top: float = 0.0,
    max_top: float = 0.20,
    size_z: float = 0.15,
):
    mj_model = env.mj_model
    tile_ids = [
        i for i in range(mj_model.ngeom)
        if (mj_model.geom(i).name or "").startswith("tile_")
    ]
    if not tile_ids:
        return 0

    rng = np.random.default_rng(seed)
    pz = rng.uniform(min_top, max_top, size=len(tile_ids)) - size_z

    for tid, z in zip(tile_ids, pz):
        mj_model.geom_pos[tid, 2] = z

    ids = jnp.array(tile_ids)
    geom_pos = env._mjx_model.geom_pos.at[ids, 2].set(jnp.asarray(pz))
    env._mjx_model = env._mjx_model.tree_replace({"geom_pos": geom_pos})
    return len(tile_ids)


def _tile_rows(mj_model, tile_ids):
    xs = np.array([mj_model.geom_pos[tid][0] for tid in tile_ids])
    unique_x = np.unique(np.round(xs, 4))
    return np.array(
        [int(np.searchsorted(unique_x, round(float(x), 4))) for x in xs]
    )

def make_stair_randomizer(
    mj_model,
    num_tiles: int,
    min_step: float = 0.0,
    max_step: float = 0.15,
    size_z: float = 0.60,
):
    
    tile_ids_list = [mj_model.geom(f"tile_{i}").id for i in range(num_tiles)]
    rows = jnp.asarray(_tile_rows(mj_model, tile_ids_list), dtype=jnp.float32)
    tile_ids = jnp.array(tile_ids_list)

    def domain_randomize(model, rng):
        @jax.vmap
        def rand(rng):
            step = jax.random.uniform(rng, minval=min_step, maxval=max_step)
            pz = rows * step - size_z
            geom_pos = model.geom_pos.at[tile_ids, 2].set(pz)
            return geom_pos

        geom_pos = rand(rng)
        in_axes = jax.tree_util.tree_map(lambda _: None, model)
        in_axes = in_axes.tree_replace({"geom_pos": 0})
        model = model.tree_replace({"geom_pos": geom_pos})
        return model, in_axes

    return domain_randomize


def apply_random_stairs(env, seed=0, min_step=0.0, max_step=0.15, size_z=0.60):
    mj_model = env.mj_model
    tile_ids = [
        i for i in range(mj_model.ngeom)
        if (mj_model.geom(i).name or "").startswith("tile_")
    ]
    if not tile_ids:
        return 0
    rows = _tile_rows(mj_model, tile_ids)
    step = np.random.default_rng(seed).uniform(min_step, max_step)
    pz = rows * step - size_z
    for tid, z in zip(tile_ids, pz):
        mj_model.geom_pos[tid, 2] = z
    ids = jnp.array(tile_ids)
    geom_pos = env._mjx_model.geom_pos.at[ids, 2].set(jnp.asarray(pz))
    env._mjx_model = env._mjx_model.tree_replace({"geom_pos": geom_pos})
    return len(tile_ids)


def make_terrain_randomizer(
    mj_model,
    num_tiles: int,
    min_top: float = 0.0,
    max_top: float = 0.20,
    size_z: float = 0.15,
):
    tile_ids = jnp.array(
        [mj_model.geom(f"tile_{i}").id for i in range(num_tiles)]
    )

    def domain_randomize(model, rng):
        @jax.vmap
        def rand(rng):
            top = jax.random.uniform(
                rng, shape=(num_tiles,), minval=min_top, maxval=max_top
            )
            pz = top - size_z
            geom_pos = model.geom_pos.at[tile_ids, 2].set(pz)
            return geom_pos

        geom_pos = rand(rng)
        in_axes = jax.tree_util.tree_map(lambda _: None, model)
        in_axes = in_axes.tree_replace({"geom_pos": 0})
        model = model.tree_replace({"geom_pos": geom_pos})
        return model, in_axes

    return domain_randomize
