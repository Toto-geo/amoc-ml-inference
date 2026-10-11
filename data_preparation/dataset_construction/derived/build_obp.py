#!/usr/bin/env python3
"""Build the 1958-2025 ORAS5 ocean bottom pressure source.

Inputs are the regional 3-D temperature and salinity sections, the regional
SSH section and the ORAS5 mesh mask. The 300-km-FWHM spatial filter belongs
to the later model-input construction step, not to this source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import time
import warnings
from pathlib import Path
from urllib.request import Request, urlopen

import gsw
import numpy as np
import xarray as xr


FILENAME = "oras5_obp_gracelike_34p5S_pm2deg_1958_2025_ref2004_2009_fixedints.nc"
MESH_URL = (
    "https://icdc.cen.uni-hamburg.de/thredds/fileServer/ftpthredds/"
    "EASYInit/oras5/ORCA025/mesh/mesh_mask.nc"
)
MESH_SHA256 = "ba88418bc8055972b675ceedbd4345b3827f0ede8cef04d53c97b66a5c7a46a7"
MESH_SIZE = 650753384
TEMP_ROWS = slice(5, 25)         # 20-row window inside the 30-row T band matching the SSH grid
MESH_ROWS = slice(341, 361)      # ORCA025 rows for the SSH 2-deg band
MESH_COLUMNS = slice(933, 1228)  # ORCA025 columns for the Atlantic longitudes


def prepare_mesh(path: Path, download: bool) -> None:
    """Ensure the ORAS5 mesh mask is at path and verify its SHA-256."""
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"Missing ORAS5 mesh: {path}; use --download-mesh")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")   # staging file
        if temporary.exists():
            raise FileExistsError(temporary)
        try:
            digest = hashlib.sha256()
            # Download in 8 MB chunks over HTTP Range requests.
            with temporary.open("wb") as output:
                for start in range(0, MESH_SIZE, 8 * 1024 * 1024):
                    end = min(start + 8 * 1024 * 1024, MESH_SIZE) - 1
                    request = Request(MESH_URL, headers={"Range": f"bytes={start}-{end}"})
                    for attempt in range(3):
                        try:
                            with urlopen(request, timeout=120) as source:
                                expected_range = f"bytes {start}-{end}/{MESH_SIZE}"
                                if source.status != 206 or source.headers.get("Content-Range") != expected_range:
                                    raise ValueError(f"Unexpected mesh download range at byte {start}")
                                block = source.read()
                            if len(block) != end - start + 1:
                                raise OSError(f"Incomplete mesh download range at byte {start}")
                            break
                        except OSError:
                            if attempt == 2:
                                raise
                            time.sleep(attempt + 1)
                    output.write(block)
                    digest.update(block)
            if digest.hexdigest() != MESH_SHA256:
                raise ValueError("Downloaded ORAS5 mesh differs from the paper source")
            temporary.replace(path)                          # rename when fully written
            return
        finally:
            temporary.unlink(missing_ok=True)
    # Already on disk: verify its hash.
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != MESH_SHA256:
            raise ValueError(f"ORAS5 mesh SHA-256 mismatch: {path}")


def geometry(temp: xr.Dataset, sal: xr.Dataset, ssh: xr.Dataset, mesh: xr.Dataset) -> dict:
    """Validate inputs and build the shared T-grid geometry used by column integration."""
    # Time axis and grid consistency between the three dynamic sources.
    expected = np.arange(np.datetime64("1958-01"), np.datetime64("2026-01"), dtype="datetime64[M]")
    for name, source in (("temperature", temp), ("salinity", sal), ("SSH", ssh)):
        if not np.array_equal(source.time_counter.values.astype("datetime64[M]"), expected):
            raise ValueError(f"{name} must contain all 816 months from 1958 through 2025")
    if not (np.array_equal(temp.time_counter.values, sal.time_counter.values)
            and np.array_equal(temp.time_counter.values, ssh.time_counter.values)):
        raise ValueError("Temperature, salinity and SSH timestamps differ")
    if dict(temp.sizes) != dict(sal.sizes) or temp.sizes.get("y") != 30 or ssh.sizes.get("y") != 20:
        raise ValueError("Unexpected 3-D or SSH grid dimensions")
    if temp.sizes.get("x") != 295 or ssh.sizes.get("x") != 295 or temp.sizes.get("deptht") != 75:
        raise ValueError("Unexpected section width or number of depth levels")
    if not np.array_equal(temp.deptht.values, sal.deptht.values):
        raise ValueError("Temperature and salinity depth levels differ")

    lat, lon = ssh.nav_lat.values, ssh.nav_lon.values          # SSH grid is the shared T-grid
    for name, source in (("temperature", temp), ("salinity", sal)):
        if not (np.array_equal(source.nav_lat.values[TEMP_ROWS], lat)
                and np.array_equal(source.nav_lon.values[TEMP_ROWS], lon)):
            raise ValueError(f"{name} section does not match SSH")
    for name, grid in (("nav_lat", lat), ("nav_lon", lon)):
        static = mesh[name].values[MESH_ROWS, MESH_COLUMNS]
        if not np.allclose(static, grid, rtol=0, atol=1e-5):
            raise ValueError(f"Mesh {name} does not match SSH")

    # Effective T-cell thickness including partial bottom cells.
    selection = dict(t=0, y=MESH_ROWS, x=MESH_COLUMNS)
    mask = mesh.tmask.isel(**selection).values.astype(bool)             # (deptht, y, x) wet-cell mask
    mbathy = mesh.mbathy.isel(**selection).values.astype(np.int16)      # number of wet levels per column
    surface = mask[0]                                                   # ocean mask at the surface
    e3t = mesh.e3t_0.isel(t=0).values.astype(np.float64)                # nominal T-cell thickness per level
    e3t_ps = mesh.e3t_ps.isel(**selection).values.astype(np.float64)    # partial-cell thickness per column
    levels = np.arange(len(e3t))[:, None, None]
    dz = np.where(levels < mbathy - 1, e3t[:, None, None],
                  np.where(levels == mbathy - 1, e3t_ps[None], 0.0))
    if not np.array_equal(dz > 0, mask) or not np.array_equal(mbathy > 0, surface):
        raise ValueError("Effective T-cell thickness disagrees with the mesh mask")
    depth = temp.deptht.values.astype(np.float64)
    pressure = gsw.p_from_z(-depth[:, None, None], lat.astype(np.float64)[None])   # reference pressure field (dbar)
    return {"lat": lat, "lon": lon, "depth": depth, "mask": mask,
            "surface": surface, "mbathy": mbathy, "dz": dz, "pressure": pressure}


def column_mass_at(index: int, temp: xr.Dataset, sal: xr.Dataset,
                   ssh: xr.Dataset, grid: dict) -> np.ndarray:
    """Compute ocean column mass per unit area at one month index."""
    theta = temp.votemper.isel(time_counter=index, y=TEMP_ROWS).values.astype(np.float64)
    practical_salinity = sal.vosaline.isel(time_counter=index, y=TEMP_ROWS).values.astype(np.float64)
    height = ssh.sossheig.isel(time_counter=index).values.astype(np.float64)
    # TEOS-10: practical salinity -> absolute salinity -> conservative temperature -> in-situ density.
    absolute_salinity = gsw.SA_from_SP(
        practical_salinity, grid["pressure"],
        grid["lon"].astype(np.float64)[None], grid["lat"].astype(np.float64)[None],
    )
    conservative_temperature = gsw.CT_from_pt(absolute_salinity, theta)
    density = gsw.rho(absolute_salinity, conservative_temperature, grid["pressure"])
    interior = np.sum(np.where(grid["mask"], density, 0.0) * grid["dz"], axis=0)   # column integral of rho*dz
    interior = np.where(grid["surface"], interior, np.nan)
    free_surface = np.where(grid["surface"], density[0] * height, np.nan)          # SSH mass term using surface density
    return np.where(grid["surface"], interior + free_surface, np.nan)


def build(temp: xr.Dataset, sal: xr.Dataset, ssh: xr.Dataset, mesh: xr.Dataset) -> xr.Dataset:
    """Loop over 816 months to build column mass and derive OBP anomaly products."""
    grid = geometry(temp, sal, ssh, mesh)
    mass = np.empty((816, 20, 295), dtype=np.float64)
    for index in range(816):
        mass[index] = column_mass_at(index, temp, sal, ssh, grid)
        if (index + 1) % 120 == 0 or index in (0, 815):
            print(f"Column mass: {index + 1}/816", flush=True)
    # Land columns are all NaN by construction.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        baseline = np.nanmean(mass[552:624], axis=0)   # 2004-01 through 2009-12
    anomaly = mass - baseline
    f32 = np.float32
    result = xr.Dataset(
        data_vars={
            "column_mass_abs": (("time_counter", "y", "x"), mass.astype(f32), {
                "long_name": "Absolute ocean column mass per unit area", "units": "kg m^-2",
                "note": "Includes interior column integral plus free-surface contribution"}),
            "column_mass_anom_ref_2004_2009": (("time_counter", "y", "x"), anomaly.astype(f32), {
                "long_name": "Ocean column mass anomaly relative to 2004-01 to 2009-12 mean", "units": "kg m^-2"}),
            "obp_anom_ref_2004_2009": (("time_counter", "y", "x"), (9.81 * anomaly).astype(f32), {
                "long_name": "Ocean bottom pressure anomaly relative to 2004-01 to 2009-12 mean", "units": "Pa"}),
            "obp_anom_eqw_ref_2004_2009": (("time_counter", "y", "x"), (anomaly / 1000.0).astype(f32), {
                "long_name": "Ocean bottom pressure anomaly expressed as equivalent water thickness relative to 2004-01 to 2009-12 mean",
                "units": "m", "reference_density": "1000 kg m^-3"}),
            "column_mass_ref_mean_2004_2009": (("y", "x"), baseline.astype(f32), {
                "long_name": "Reference mean ocean column mass per unit area for 2004-01 to 2009-12", "units": "kg m^-2"}),
            "tmask_surface": (("y", "x"), grid["surface"].astype(np.int8), {
                "long_name": "Surface T-grid ocean mask on common domain", "units": "1"}),
            "mbathy": (("y", "x"), grid["mbathy"].astype(np.int16), {
                "long_name": "Number of wet T levels in water column", "units": "1"}),
        },
        coords={
            "time_counter": ("time_counter", temp.time_counter.values, {"long_name": "time"}),
            "y": ("y", np.arange(20), {"long_name": "meridional index on common domain"}),
            "x": ("x", np.arange(295), {"long_name": "zonal index on common domain"}),
            "deptht": ("deptht", grid["depth"].astype(f32), {"long_name": "Vertical T levels", "units": "m"}),
            "nav_lat": (("y", "x"), grid["lat"].astype(f32), {"long_name": "Latitude", "units": "degrees_north"}),
            "nav_lon": (("y", "x"), grid["lon"].astype(f32), {"long_name": "Longitude", "units": "degrees_east"}),
        },
        attrs={
            "title": "ORAS5 GRACE-like ocean bottom pressure anomaly dataset on common T-grid domain",
            "source_model": "ORAS5",
            "domain": "Common T-grid domain consistent with temperature, salinity and SSH datasets",
            "reference_period": "2004-01 to 2009-12",
            "method_summary": (
                "Column mass built from TEOS-10 in-situ density computed from practical salinity and potential temperature, "
                "integrated over effective T-cell thickness including partial bottom cells, with added free-surface mass term."),
            "gravity": 9.81,
            "rho_fw": 1000.0,
            "notes": ("Absolute atmospheric pressure is not included. "
                      "Product is oceanic and GRACE-like, not a full GRACE processing replica."),
        },
    )
    return result


def write_new(dataset: xr.Dataset, path: Path) -> None:
    """Write dataset to path atomically with per-variable encoding."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")   # staging file
    if temporary.exists():
        raise FileExistsError(temporary)
    encoding = {name: {"dtype": "float32", "zlib": True, "complevel": 4,
                       "_FillValue": np.float32(np.nan)}
                for name in ("column_mass_abs", "column_mass_anom_ref_2004_2009",
                             "obp_anom_ref_2004_2009", "obp_anom_eqw_ref_2004_2009",
                             "column_mass_ref_mean_2004_2009", "nav_lat", "nav_lon", "deptht")}
    encoding.update({"time_counter": {"dtype": "float64", "units": "days since 1958-01-01 00:00:00",
                                      "calendar": "standard"},
                     "tmask_surface": {"dtype": "int8", "zlib": True, "complevel": 4},
                     "mbathy": {"dtype": "int16", "zlib": True, "complevel": 4}})
    try:
        dataset.to_netcdf(temporary, engine="netcdf4", format="NETCDF4", encoding=encoding)
        temporary.replace(path)                          # rename when fully written
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=Path, required=True)
    parser.add_argument("--salinity", type=Path, required=True)
    parser.add_argument("--ssh", type=Path, required=True)
    parser.add_argument("--mesh-mask", type=Path, required=True)
    parser.add_argument("--download-mesh", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir / FILENAME
    if output.exists():
        raise FileExistsError(output)
    prepare_mesh(args.mesh_mask, args.download_mesh)
    with (xr.open_dataset(args.temperature, engine="netcdf4") as temp,
          xr.open_dataset(args.salinity, engine="netcdf4") as sal,
          xr.open_dataset(args.ssh, engine="netcdf4") as ssh,
          xr.open_dataset(args.mesh_mask, engine="netcdf4") as mesh):
        result = build(temp, sal, ssh, mesh)
    write_new(result, output)
    print(f"Wrote {output}: {dict(result.sizes)}", flush=True)


if __name__ == "__main__":
    main()
