import matplotlib.pyplot as plt
from lozoya.plot import plot_streamlines, plot_cylinders, format_axes

from lozoya.physics import cylinder_stream_function, velocity_field, get_lims


def run(U, radii, centroids, density=14.7, atm=2116.2):
    xlim, ylim = get_lims(radii, centroids)
    fig, ax = plt.subplots(figsize=(4, 4))
    n = 0
    for radius, centroid in zip(radii, centroids):
        if n == 0:
            psi = cylinder_stream_function(U, radius, centroid)
        else:
            psi += cylinder_stream_function(U, radius, centroid)
        n += 1
    u, v = velocity_field(psi / len(radii))
    plot_streamlines(fig, ax, U, u, v, density, atm, xlim, ylim)
    plot_cylinders(ax, radii, centroids)
    format_axes(ax)
    plt.show()


run(
    U=50,
    radii=(
        30,
        # 13,
        # 13,
    ),
    centroids=(
        (0, 0),
        # (-75, 75),
        # (75, -75),
    )
)
