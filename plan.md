# Data extraction

- How to find cold air outbreaks where it is hard visually (Polar night)?
- Function to extract only between given longitudes and latitude
- Unsupervised learning?
  - Classifying different cloud fraction areas and closed-open-cell transition between them
  - Problems - transition being orthogonal to wind direction. Classification may not understand this.
    - Solutions -
- Supervised learning?
  - Classifying a "line" based on certain properties by definition of closed-open-cell transition
  - Already defined positions of the transitions?
  - Problems are limited available classifications

# Analysis

- Measurements with cloud top above > 500 hPa needs to be eliminated (from Rebecca et al.)
- Cloud_Multi_Layer_Flag to filter data only containing single layer clouds. (from Rebecca et al.)
- filtering to remove bias “Cloud_Mask_SPI” >30; (Zhang and Platnick, 2011) and high solar-zenith angles (> 65∘) or high viewing angles (> 50∘) (Grosvenor and Wood, 2014).
- Why is Rebecca et al calculating LWP and Nd by themselves and not using the already calculated values from MODIS?
- Rebecca et al. removing ice clouds by filtering by minimum cloud top temperature
