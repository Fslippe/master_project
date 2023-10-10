# Data extraction

- How to find cold air outbreaks where it is hard visually (Polar night)?
  - Use thermal infrared (band 29)

- Remove pixels if covering more than a percentage of land. 
  - Removed patches containing less than 90% ocean.
    - Problems:
      - loosing a lot of data on the land/sea boundaries if patches fall in the wrong places
      - Still patches containing land which may lead to problems


- Unsupervised learning?
  - Classifying different cloud fraction areas and closed-open-cell transition between them
    - Problems - transition being orthogonal to wind direction. Classification may not understand this.
      - Solutions - line up with era 5 wind direction and extract only boundaries orthogonal to the wind

  - boundary between two clusters may not be of interest. Follow wind to find area of interest, or even how variables change following the wind backwards. 

  - Secondary unsupervised network:
    - Extract patches or region of interest (find a way to save longitude latitude data)
    - Train model on outbreak data. Use already trained model for further optimization
    - Train clustering on encoded data from secondary model
    - Do predictions on pathces from region of interest with a stride (1km) to gain a transition of the same resolution. 

- Supervised learning?
  - Classifying a "line" based on certain properties by definition of closed-open-cell transition
  - Already defined positions of the transitions?
  - Problems are limited available classifications

# Analysis

- Measurements with cloud top above > 500 hPa needs to be eliminated (from Rebecca et al.)
- Cloud_Multi_Layer_Flag to filter data only containing single layer clouds. (from Rebecca et al.)
- filtering to remove bias “Cloud_Mask_SPI” >30; (Zhang and Platnick, 2011) and high solar-zenith angles (> 65∘) or high viewing angles (> 50∘) (Grosvenor and Wood, 2014).
- Remove pixels with sea ice or land 


# Problems met underway which needed fixing
- Using just band 29 gives same clusters for cloud streets in non-CAO cases over central europe 
  - Solutions? Extract only certain parts of pictures 
- negative Radiances. min value being -1.5W
  - solutions: Setting all values less than 0 to 0
- Some values being out of valid range
  - Solutions: setting value to 0
- Land topography being labeled as CAO
  - Solutions: masking patches with more than half including land areas.
  - Masking using modis data - have to find lon lat corresponding to label of mask 
- Stretching because of pixel not being of same resolution for every angle "Spatial resolution for pixels at nadir is 1 km, degrading to 4.8 km in the along-scan direction at the scan extremes." (https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD021KM)
  - Solutions: - Use only nadir for 1km resolution in both directions.  
               - Give angle as input to the encoder while only performing clustering on layers containing bands. - Is it possible given that encoded patches has increased depth?
- Slow convergence in training of autoencoder
  - Solutions: increase learning rate from 1e-4 to 1e-3. OR implement learning rate scheduler 
  

# TO DOO
- #perform cluster training on whole season 
- #get a historgram of monthly frequency of CAOs 
- #Train on nighttime data, and perform clustering on whole set.  
- #Use newly trained model and try clustering
- #Clustering results on july - not able to understand the higher values when normalized using training max and only trained on winter/spring
- Extract longitude latitude from clusters and gain histogram map


