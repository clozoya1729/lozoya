import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = '#e8e8e8'
plt.rcParams['axes.edgecolor'] = '#e8e8e8'
plt.rcParams['figure.facecolor'] = '#e8e8e8'


def plot_2nd_order_contribution(helmholtzCoils):
    # 2D plot of 2nd Order contribution vs winding
    axesRange = [
        0,
        helmholtzCoils.Na0.max(),
        0,
        helmholtzCoils.Nb0.max(),
    ]
    fig = plt.figure(figsize=(8, 6))
    plt.title('Field Error Due to Winding Geomertry')
    plt.xlabel('# horizontal windings, $N_{a}$')
    plt.ylabel('# vertical windings, $N_{b}$')
    plt.xticks(helmholtzCoils.Na0 - 0.5, helmholtzCoils.Na0)  # align tick marks with data
    plt.yticks(helmholtzCoils.Nb0 - 0.5, helmholtzCoils.Nb0)
    plt.imshow(
        helmholtzCoils.percentO2,
        cmap='magma',
        vmin=helmholtzCoils.percentO2.min(),
        vmax=helmholtzCoils.percentO2.max(),
        interpolation='none',
        origin='lower',
        extent=axesRange,
        aspect='auto',
        norm=LogNorm(),
    )
    plt.axvline(
        12.5,
        color='black',
        linewidth=1,
        linestyle='--',
        alpha=0.3,
    )
    plt.axhline(
        13.5,
        color='black',
        linewidth=1,
        linestyle='--',
        alpha=0.3,
    )
    cbar = plt.colorbar()
    cbar.set_label('$\mathcal{O}(2)$ Weight (%)')
    plt.grid(False)
    plt.show()


def plot_field_deviation_central_patch(x,
                                       y,
                                       deviation,
                                       ptsPerAxis):
    # plot central patch of Helmholtz field deviation
    fig = plt.figure(figsize=(10, 8))
    plt.title('Helmholtz Field Inhomogeneities (ideal)')
    plt.xlabel('Axial Displacement, $x$ (mm)')
    plt.ylabel('Radial Displacement, $y$ (mm)')
    axesRange = [
        x.min(),
        x.max(),
        y.min(),
        y.max(),
    ]
    plt.imshow(
        deviation,
        cmap='magma',
        vmin=deviation.min(),
        vmax=deviation.max(),
        interpolation='none',
        origin='lower',
        extent=axesRange,
        aspect='auto',
    )
    cbar = plt.colorbar()
    cbar.set_label('Field Deviation, $\delta$B$_{0}$ ($\mu T$)')
    # add quiver arrows for direction of field divergence
    subsampleBy = int(ptsPerAxis / 10)
    plt.quiver(
        x[::subsampleBy],
        y[::subsampleBy],
        deviation[::subsampleBy, ::subsampleBy],
        deviation[::subsampleBy, ::subsampleBy],
        angles='xy',
        units='width',
        scale_units='xy',
        pivot='mid',
        alpha=0.25,
        width=0.005,
    )
    # add contour value overlay
    contourVals = [-1, -0.1, 0.1, 1]
    CP = plt.contour(
        x,
        y,
        deviation,
        contourVals,
        linewidths=3,
        cmap='gnuplot2',
    )
    plt.clabel(
        CP,
        inline=1,
        fontsize=10,
        fmt='%1.2f',
    )
    plt.grid(False)
    plt.show()
