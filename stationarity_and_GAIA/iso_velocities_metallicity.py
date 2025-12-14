import numpy as np
import auxiliary_functions as aux
import pyvo
from astropy.table import Table
import time


    
p_min = 8
T_min = 0
T_max = 1000000


#    m = [-100000, -1]

m = [0, 10000]


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
    AND g.parallax_over_error > 10
    AND g.visibility_periods_used > 8
    AND g.astrometric_n_obs_al > 100
"""



#query = f"""
#SELECT
#    g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.parallax, 
#    g.radial_velocity, g.phot_g_mean_mag, g.teff_gspphot, 
#    ap.radius_gspphot, g.mh_gspphot
#FROM 
#    gaiadr3.gaia_source AS g
#JOIN 
#    gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id
#WHERE 
#    g.radial_velocity IS NOT NULL
#    AND ap.radius_gspphot IS NOT NULL
#    AND g.teff_gspphot IS NOT NULL
#	AND g.teff_gspphot BETWEEN {T_min} AND {T_max}
#    AND g.parallax_over_error > 10
#    AND g.visibility_periods_used > 8
#    AND g.astrometric_n_obs_al > 100
#"""



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

alpha = np.deg2rad(results["ra"])
delta = np.deg2rad(results["dec"])

v_d = 4740 * results["pmdec"] / results["parallax"]
v_a = 4740 * results["pmra"] / results["parallax"] 
v_r = results["radial_velocity"] * 1000

v = np.sqrt(v_d**2 + v_a**2 + v_r**2)

metallicity  = results["mh_gspphot"]

selection = np.logical_and(v<150e3, metallicity > -2)

v = v[selection]
metallicity = metallicity[selection]

np.savetxt('metalicnost.txt', np.column_stack((v, metallicity)))