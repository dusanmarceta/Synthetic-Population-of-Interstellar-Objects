import pyvo
from astropy.table import Table
import time

p_min = 100
T_min = 0
T_max = 1000000
m_1 = 0.2 # metaličnost
m_2 = -0.2

query = f"""
SELECT
    g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.parallax, 
    g.radial_velocity, g.phot_g_mean_mag, g.teff_gspphot, 
    ap.radius_gspphot, g.mh_gspphot
FROM 
    gaiadr3.gaia_source AS g
JOIN 
    gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id
WHERE 
    g.parallax > {p_min}
    AND g.radial_velocity IS NOT NULL
    AND ap.radius_gspphot IS NOT NULL
    AND g.teff_gspphot IS NOT NULL
	AND g.teff_gspphot BETWEEN {T_min} AND {T_max}
    AND (g.mh_gspphot > {m_1} OR g.mh_gspphot < {m_2})
    AND g.parallax_over_error > 10
    AND g.visibility_periods_used > 8
    AND g.astrometric_n_obs_al > 100
"""

tap_service = pyvo.dal.TAPService("https://gaia.aip.de/tap")

print("Submitting Async Job to AIP Mirror...")
job = tap_service.submit_job(query)
job.run()

print("Job running...")
while job.phase not in ("COMPLETED", "ERROR", "ABORTED"):
    time.sleep(2)

if job.phase == "COMPLETED":
    print("Job Finished!")
    results = job.fetch_result().to_table()
    print(f"Retrieved {len(results)} stars.")
else:
    print(f"Job Failed. Server Message: {job.messages}")


print(results)