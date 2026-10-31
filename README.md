# Static balance for climbing robots

Code for the [Rock Climbing Robots Workshop](https://rock-climbing-robots.github.io/), 29 October – 1 November 2026, Bordeaux, France.

## Usage

You can run the scripts with [uv](https://docs.astral.sh/uv/). From the root of this repository:

```console
uv run microban_mink.py          # in MuJoCo's viewer
uv run microban_pinker.py        # in a Web browser
```

Each example prints its usage to the terminal. The robot and the climbing wall come from the hackathon's [simulator](https://github.com/rock-climbing-robots/simulator).

## Support polygon

As you move the hand/pelvis/foot targets around, the examples draw the **static-equilibrium polygon** of these contacts at the height of the center of mass (CoM).[^1] It is the region, seen from above, where the CoM must be in order for the contacts to hold the robot still, without any contact exceeding its maximum force. The polygon is green when the CoM is inside it, and turns red when the CoM leaves it, denoting a configuration that the robot cannot sustain.

![](https://github.com/user-attachments/assets/e89c348c-7440-463e-8a28-50ee36f190e4)

The Pinker example also shows contact forces. In general, there are many contact forces that can sustain the robot when the CoM is inside the static-equilibrium polygon: here we just pick those of minimum norm. This tells you which limb carries the most load, and which one would give up first as the CoM gets closer to the edge of the polygon.

**Extra insight:** the closer the CoM is to an edge of the support polygon, the larger the maximum force is on one of the contacts. When the CoM is on the edge, there is always at least one force hitting the maximum.

[^1]: The full zone is actually a vertical prism in which the CoM can be anywhere. We only draw the base polygon for convenience.

## To go further

Ideas:

- [Testing static equilibrium for legged robots](https://lall.stanford.edu/papers/bretl_eqmcut_ieee_tro_projection_2008_08_01_01/pubdata/entry.pdf) by Bretl and Lall (2008) introduced the iterative polytope projection algorithm we use here.
- [Contact stability](https://scaron.info/robotics/contact-stability.html) is the more general condition to check that a robot motion is *feasible* while maintaining contacts. It gives rise to various conditions in statics and dynamics.

Software:

- [mink](https://github.com/kevinzakka/mink): differential inverse kinematics in Python, based on the MuJoCo physics engine.
- [pinker](https://github.com/pink-kinematics/pinker): differential inverse kinematics in Python, lightweight for embedded robots.
- [pypoman](https://github.com/stephane-caron/pypoman): polyhedral projection functions we use to compute the CoM support polygon.
- [StabiliPy](https://github.com/haudren/stabilipy): a Python package implementing a more general recursive method for polytope projection.
