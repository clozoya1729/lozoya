import config
from app.helmholtz_coil.hc_geometry import HelmholtzCoils
from app.helmholtz_coil.plot_util import plot_field_deviation_central_patch, plot_2nd_order_contribution


def run(wireProperties):
    helmholtzCoils = HelmholtzCoils(
        wireProperties=wireProperties,
        resolution=100
    )
    plot_2nd_order_contribution(
        helmholtzCoils=helmholtzCoils,
    )
    plot_field_deviation_central_patch(
        x=helmholtzCoils.x,
        y=helmholtzCoils.y,
        deviation=helmholtzCoils.emf.deviation,
        ptsPerAxis=config.ptsPerAxis,
    )
    helmholtzCoils.wire.print_properties()
    helmholtzCoils.emf.print_properties()


wireProperties = dict(
    innerPoleRadius=94,
    formWallWidth=5,
    windingsHorizontal=13,
    windingsVertical=14,
    coreThickness=0.8128,
    insulationThickness=0.0150,
    ohmPerMeter=0.028494094488075,
    current=5,
    tau=0,
    rho=0,
    Nstart=1,
    Nstop=21,
)
run(wireProperties)
