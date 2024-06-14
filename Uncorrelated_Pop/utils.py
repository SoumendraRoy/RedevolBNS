import jax.numpy as jnp

def jnp_cumtrapz(ys, xs):
    """Cumulative trapezoidal integral of the function defined by values `ys` at points `xs`."""
    xs = jnp.array(xs)
    ys = jnp.array(ys)

    return jnp.concatenate((jnp.zeros(1), jnp.cumsum(0.5*jnp.diff(xs)*(ys[:-1] + ys[1:]))))