import jax
import jax.numpy as jnp


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
