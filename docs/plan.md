# Data extraction

- How to find cold air outbreaks where it is hard visually (Polar night)?
- Unsupervised learning?
  - Classifying different cloud fraction areas and closed-open-cell transition between them
  - Problems - transition being orthogonal to wind direction. Classification may not understand this.
    - Solutions - line up with era 5 wind direction and extract only boundaries orthogonal to the wind
  - boundary between two clusters may not be of interest. Follow wind to find area of interest, or even how variables change following the wind backwards. 
  
- Supervised learning?
  - Classifying a "line" based on certain properties by definition of closed-open-cell transition
  - Already defined positions of the transitions?
  - Problems are limited available classifications

# Analysis

- Measurements with cloud top above > 500 hPa needs to be eliminated (from Rebecca et al.)
- Cloud_Multi_Layer_Flag to filter data only containing single layer clouds. (from Rebecca et al.)
- filtering to remove bias “Cloud_Mask_SPI” >30; (Zhang and Platnick, 2011) and high solar-zenith angles (> 65∘) or high viewing angles (> 50∘) (Grosvenor and Wood, 2014).
- Remove pixels with sea ice or land 

