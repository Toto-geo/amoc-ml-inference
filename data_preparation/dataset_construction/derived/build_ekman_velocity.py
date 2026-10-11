#!/usr/bin/env python3
"""Build the 1958-2025 ORAS5 Ekman-equivalent meridional velocity at 34.5 S.

Reads the ORAS5 V-section, the zonal wind-stress band and the mesh mask, and
writes the Ekman transport, the equivalent V-face velocity distributed over
the 0-50 m layer, and the geometry needed to reconstruct it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


FILENAME = "oras5_ekman_velocity_34p5S_1958_2025.nc"
RHO0 = 1025.0               # Boussinesq reference density (kg m-3)
OMEGA = 7.292115e-5         # Earth angular velocity (rad s-1)
LAYER_DEPTH = 50.0          # depth over which the Ekman transport is distributed (m)


def build(velocity: xr.Dataset, wind: xr.Dataset, mesh: xr.Dataset) -> xr.Dataset:
    """Compute the Ekman-equivalent V-section velocity and its diagnostics."""
    # Check time axis and section shapes.
    expected = np.arange(np.datetime64("1958-01"), np.datetime64("2026-01"), dtype="datetime64[M]")
    if not (np.array_equal(velocity.time_counter.values, wind.time_counter.values)
            and np.array_equal(velocity.time_counter.values.astype("datetime64[M]"), expected)):
        raise ValueError("Velocity and wind must share all 816 months from 1958 through 2025")
    if (velocity.sizes.get("depthv"), velocity.sizes.get("y"), velocity.sizes.get("x")) != (75, 1, 295):
        raise ValueError("Unexpected velocity section")
    if (wind.sizes.get("y"), wind.sizes.get("x")) != (15, 295):
        raise ValueError("Unexpected wind-stress band")

    # Section geometry: pick the U/V row at 34.5 S and verify its coordinates against the mesh.
    row = int(velocity.attrs["selected_j_global"])
    x0, x1 = int(velocity.attrs["x_start_global"]), int(velocity.attrs["x_end_global"])
    wind_row = row - int(wind.attrs["y_start_global"])            # local U-row inside the wind band
    if (row, x0, x1, wind_row) != (350, 933, 1227, 6):
        raise ValueError("Unexpected section indices")
    section = dict(t=0, y=row, x=slice(x0, x1 + 1))
    lat_u = mesh.gphiu.isel(**section).values                     # U-point latitudes along the section
    lon_u = mesh.glamu.isel(**section).values
    lat_v = mesh.gphiv.isel(**section).values                     # V-point latitudes along the section
    lon_v = mesh.glamv.isel(**section).values
    e1u = mesh.e1u.isel(**section).values                         # zonal cell width at U points (m)
    e1v = mesh.e1v.isel(**section).values                         # zonal cell width at V points (m)
    if not (np.allclose(wind.nav_lat.isel(y=wind_row).values, lat_u, rtol=0, atol=1e-5)
            and np.allclose(wind.nav_lon.isel(y=wind_row).values, lon_u, rtol=0, atol=1e-5)
            and np.allclose(velocity.nav_lat.isel(time_counter=0, y=0).values, lat_v,
                            rtol=0, atol=1e-5)
            and np.allclose(velocity.nav_lon.isel(time_counter=0, y=0).values, lon_v,
                            rtol=0, atol=1e-5)):
        raise ValueError("Dynamic U/V coordinates do not match the ORAS5 mesh")

    # Ekman transport across the Atlantic U section.
    mask_u = mesh.umask.isel(t=0, z=0, y=row, x=slice(x0, x1 + 1)).values.astype(np.int8)
    mask_v = mesh.vmask.isel(t=0, z=0, y=row, x=slice(x0, x1 + 1)).values.astype(np.int8)
    tau = np.where(mask_u[None, :] == 1, wind.sozotaux.isel(y=wind_row).values, 0.0)
    f_u = 2.0 * OMEGA * np.sin(np.deg2rad(lat_u))                                      # Coriolis parameter at U points
    m_ek = -(tau / (RHO0 * f_u[None, :])) * mask_u[None, :]                            # Ekman transport per unit width (m^2 s-1)
    q_ek = m_ek * e1u[None, :]                                                         # per-cell Ekman transport (m^3 s-1)
    t_ek = np.sum(q_ek, axis=1)                                                        # total transport per month (m^3 s-1)

    # Reconstruct the effective V-face vertical thickness, including partial bottom cells.
    x_slice = slice(x0, x1 + 1)
    bottom_s = mesh.mbathy.isel(t=0, y=row, x=x_slice).values                          # bottom level on the T row south of V face
    bottom_n = mesh.mbathy.isel(t=0, y=row + 1, x=x_slice).values                      # bottom level on the T row north of V face
    partial_s = mesh.e3t_ps.isel(t=0, y=row, x=x_slice).values                         # partial-cell thickness, south T row
    partial_n = mesh.e3t_ps.isel(t=0, y=row + 1, x=x_slice).values                     # partial-cell thickness, north T row
    nominal = mesh.e3t_0.isel(t=0).values[:, None]                                     # nominal T-cell thickness per level
    level = np.arange(75)[:, None]
    dz_s = np.where(level < bottom_s[None, :] - 1, nominal,
                    np.where(level == bottom_s[None, :] - 1, partial_s[None, :], 0.0))
    dz_n = np.where(level < bottom_n[None, :] - 1, nominal,
                    np.where(level == bottom_n[None, :] - 1, partial_n[None, :], 0.0))
    dz_v = np.where((dz_s > 0) & (dz_n > 0), 0.5 * (dz_s + dz_n), 0.0)                 # effective V-face thickness (m)
    vmask = mesh.vmask.isel(t=0, y=row, x=x_slice).values
    if not np.array_equal(dz_v > 0, vmask == 1):
        raise ValueError("Reconstructed V-face thickness disagrees with the mesh mask")

    # Distribute the Ekman transport over the 0-50 m layer by face fraction.
    z_top = np.vstack([np.zeros((1, 295)), np.cumsum(dz_v[:-1], axis=0)])              # top depth of each V face (m)
    z_bottom = np.cumsum(dz_v, axis=0)                                                 # bottom depth of each V face (m)
    dz_50 = np.maximum(0.0, np.minimum(z_bottom, LAYER_DEPTH) - z_top)                 # face thickness inside 0-50 m
    alpha = np.divide(dz_50, dz_v, out=np.zeros_like(dz_50), where=dz_v > 0)           # fraction of each V face inside 0-50 m
    area = mask_v[None, :] * e1v[None, :] * dz_50                                      # effective 0-50 m area per V face (m^2)
    area_total = area.sum()
    v50 = t_ek / area_total                                                            # depth-uniform 50 m-layer velocity (m s-1)
    v_eq = v50[:, None, None] * alpha[None, :, :]                                      # equivalent V-face velocity (t, depthv, x)
    reconstructed = np.sum(v_eq * (e1v[None, None, :] * dz_v[None, :, :]), axis=(1, 2))  # reconstructed transport (closure check)

    # Output: dynamic Ekman variables, geometry, and provenance attributes.
    specifications = {
        "v_ek_eq": (("time_counter", "depthv", "x"), v_eq.astype("f4"), "Ekman-equivalent meridional velocity correction on V section", "m s-1"),
        "v50_star": (("time_counter",), v50.astype("f4"), "Equivalent vertically uniform Ekman velocity over effective 0-50 m area", "m s-1"),
        "T_Ek": (("time_counter",), t_ek, "Total meridional Ekman transport across Atlantic U section", "m3 s-1"),
        "T_Ek_reconstructed": (("time_counter",), reconstructed, "Transport reconstructed from v_ek_eq on V section", "m3 s-1"),
        "transport_error": (("time_counter",), reconstructed - t_ek, "Transport closure error: reconstructed minus diagnosed Ekman transport", "m3 s-1"),
        "tau_x_sec": (("time_counter", "x"), tau.astype("f4"), "Zonal wind stress on chosen U row, cleaned outside Atlantic domain", "N m-2"),
        "M_Ek": (("time_counter", "x"), m_ek.astype("f4"), "Meridional Ekman transport per unit width on U section", "m2 s-1"),
        "q_Ek": (("time_counter", "x"), q_ek, "Ekman transport contribution per U point along section", "m3 s-1"),
        "f_u": (("x",), f_u.astype("f8"), "Coriolis parameter on U row", "s-1"),
        "dz_V": (("depthv", "x"), dz_v.astype("f4"), "Reconstructed real vertical thickness of V faces", "m"),
        "dz_50": (("depthv", "x"), dz_50.astype("f4"), "Thickness of each V face contained in the 0-50 m layer", "m"),
        "alpha_50": (("depthv", "x"), alpha.astype("f4"), "Fraction of each V face contained in the 0-50 m layer", "1"),
        "A_50": (("depthv", "x"), area, "Effective 0-50 m area contribution of each V face", "m2"),
        "A_50_tot": ((), np.float64(area_total), "Total effective 0-50 m area on Atlantic V section", "m2"),
        "m_Atl_U": (("x",), mask_u, "Atlantic mask on extracted U section", "1"),
        "m_Atl_V": (("x",), mask_v, "Atlantic mask on extracted V section", "1"),
        "lon_u": (("x",), lon_u.astype("f4"), "Longitude of U section points", "degrees_east"),
        "lat_u": (("x",), lat_u.astype("f4"), "Latitude of U section points", "degrees_north"),
        "lon_v": (("x",), lon_v.astype("f4"), "Longitude of V section points", "degrees_east"),
        "lat_v": (("x",), lat_v.astype("f4"), "Latitude of V section points", "degrees_north"),
        "e1u": (("x",), e1u, "Zonal scale factor at U points", "m"),
        "e1v": (("x",), e1v, "Zonal scale factor at V points", "m"),
    }
    data_vars = {name: (dims, values, {"long_name": label, "units": units})
                 for name, (dims, values, label, units) in specifications.items()}
    return xr.Dataset(
        data_vars=data_vars,
        coords={"time_counter": ("time_counter", velocity.time_counter.values),
                "depthv": ("depthv", velocity.depthv.values.astype("f4")),
                "x": ("x", np.arange(295, dtype=np.int32))},
        attrs={
            "description": "Ekman-equivalent meridional velocity correction dataset on ORAS5 34.5S Atlantic section",
            "selected_j_global_v": row, "selected_j_global_u": row,
            "selected_u_local_index_in_sozotaux_band": wind_row,
            "x_start_global": x0, "x_end_global": x1,
            "rho0_kg_m3": RHO0, "omega_s_minus_1": OMEGA,
            "ekman_layer_depth_m": LAYER_DEPTH,
            "vface_thickness_rule": "mean_of_adjacent_T_thicknesses_on_intersection_support",
            "transport_definition": "T_Ek = sum_i [-(tau_x/(rho0*f))*e1u] over Atlantic U section",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velocity", type=Path, required=True)
    parser.add_argument("--wind-stress", type=Path, required=True)
    parser.add_argument("--mesh-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir / FILENAME
    if output.exists():
        raise FileExistsError(output)
    with (xr.open_dataset(args.velocity) as velocity,
          xr.open_dataset(args.wind_stress) as wind,
          xr.open_dataset(args.mesh_mask) as mesh):
        result = build(velocity, wind, mesh)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")   # staging file
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        result.to_netcdf(temporary, engine="netcdf4")
        temporary.replace(output)                            # rename when fully written
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {output}: {dict(result.sizes)}", flush=True)


if __name__ == "__main__":
    main()
