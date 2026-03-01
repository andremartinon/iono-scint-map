# Ionospheric Scintillation Map Generation Tool (IONO_SCINT_MAP)

The IONO_SCINT_MAP library provides tools for the generation of three types of 
scintillation maps, employed in scintillation monitoring, which are related to
the scintillation indexes S4, $\sigma_\phi$ and ROTI. The making of these 
scintillation maps requires the interpolation of IPP samples, given by the 
ionospheric scintillation index values for each IPP of each satellite-station 
link considering the set of GNSS stations of the given area and time interval. 
Interpolation is performed with the aim of filling in IPP data gaps, resulting 
in a smoother map for the considered grid in longitude and latitude, and for 
the integration interval.

These maps were implemented using a new approach (MARTINON et al., 2023). This 
new approach consists of a set of preprocessing options and an interpolation
method. It allows to generate more accurate scintillation maps with a low 
computational cost that is compatible with real-time demands. Former approaches
to investigate ionospheric scintillation over the Brazilian territory include
the generation of regional S4 index maps (REZENDE et al., 2007), and Vani (2018)
employed different preprocessing options and interpolation methods.

## Approach for generating scintillation maps

The proposed approach (MARTINON et al., 2023) is named GPR(VQI), since it uses
the Gaussian Process Regression for interpolation of the samples of a 15-minute
interval, and the specific set of preprocessing options VQI, which corresponds
to the use of the vertical projection of values of the considered scintillation
index reduced by the average value above the 3rd quartile for the group of 
15-minute samples of the considered aggregation cell. The interpolation grid has
the same spatial resolution of the map 0.25° $\times$ 0.25°, spanning latitudes
-39° to 9° and longitudes -78° to -30°, and forming a regular grid of 193 
$\times$ 193 grid points. A coarser aggregation grid with 1.0° $\times$ 1.0°
resolution with square cells centered at each grid point of the map delimitates
the grouping of the samples for the considered grid point. The interpolated grid
points are generated using the following assumptions:

1. Only samples corresponding to satellite elevations higher than 30° are 
   considered, in order to ﬁlter out data of L-band links affected by ground 
   interference and the related multipath reﬂections;
2. Scintillation index samples considered outliers are filtered out, according 
   to the following threshold:
   1. S4 values above 1.4;
   2. $\sigma_\phi$ above 1.4 radian and;
   3. ROTI above 0.4 TECU/s.
3. The ionosphere is modeled as a thin spherical shell over the Earth at the 
   mean altitude of 350 km (typically), where the maximum value of the 
   ionospheric electron density is observed;
4. The IPP for each satellite-receiver link is defined as being the intersection
   of the line-of-sight receiver-satellite (the slant path) with the ionosphere,
   using the receiver coordinates and the azimuthal and elevation angles of the
   satellite;
5. The scintillation index values are projected to the vertical direction in 
   order to take into account the geometrical effects on the measurements made
   at different elevation angles;
6. An aggregation grid with 1.0° $\times$ 1.0° resolution with square cells 
   centered at each grid point of the map is defined to group the 15-minute 
   samples of each cell, and to reduce them to an "interpolation sample" 
   associated with the grid point. The value of each "interpolation sample" is 
   given by the average of the scintillation index values above the 3rd quartile
   (top 25% values of the samples contained in the cell of the corresponding 
   grid point). Square cells without samples are discarded;
7. The centroid of each "interpolation sample" is defined by the reduction 
   function, in this case, the centroid of the group of samples with 
   scintillation index values above the 3rd quartile;
8. The interpolation process employs the "interpolation samples", each one with
   a reduced value and a centroid, to obtain the values for the interpolation 
   regular grid 0.25° $\times$ 0.25°, which are then employed to render the 
   scintillation map.

This approach and its correlated assumptions are employed to generate amplitude
scintillation maps, using the S4 index, phase scintillation maps, using the 
$\sigma_\phi$ index, and also to generate the ROTI scintillation maps.