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
- #Clustering results on july 
  - not able to understand the higher values when normalized using training max and only trained on winter/spring
  - Problem fixed by removing land areas which confused the model 
- #Extract longitude latitude from clusters and gain histogram map
- #Learning rate scheduler - 1e-3 towards 1e-4
- #Remove overlapping patches with high viewing angles 
  - Solution use less than 50deg zenith
  - Problem - still double distance in outer regions than in inner ones
- #Train model where high viewing angles are removed
- Extract patches of interest and use for further analysis and training. 
  - Approaches:
    - Train Secondary Autoencoder on the Encoded Representations of the Primary Autoencoder
    - Train Secondary Autoencoder Directly on the Areas of Interest
  - Use whole pictures or just parts of them for training. - Understand difference or just region of interest. 
  - Optimize to see the difference between open/closed cell - may need to change the patch size  
  - Try HAC
- Perform clustering and change stride for more accurate pinpointing of tranition position
  - Make manual function to get indexes of those regions. Extract pixels in region and perform secondary patch extraction 
- Run model with different number of filters. Perfect representation may be overfitting.
  - Starting out with 8-16-32-64 for test instead of 16-32-64-128


- Combine pictures of same swath path to make it easier to undestand transitions  

- Perform sensitivity tests on encoder setup 
  - SETUP OF TEST MATRIX 
  - Parameters to test
    - Patch size - Capture larger areas closer to sizes of CAO - problem in transition as these will not see differences. 
    - filter sizes - Interesting to make it easier for clustering
      - Last filter of 32 is performing worse
      - Using 64 or 128???
    - kernel sizes - may be too much
  - Test metric
    - High resolution mask ie 64
    - Larger sizes assign one value 
      - Problem - non overlapping patches from patch extraction 
    - Accuracy metric on binary mask. Right or wrong
  - Get test set
    - CAO cases - 100 pics
    - non-CAO cases - 20 pics 
    - Use more seasons 
      - Clear cases with boundaries 
      - randomized cases. 


#### TEST MATRIX
-- TESTING accuracy metric 

- Problem is different number of clusters 
  - Solution: Cluster metric algorithm 
  - Use ex 12 Clusters for all tests, to see performance - Of cource very limited - Could use 11, 12, 13 

Patch size | filters |  
    64     |   32    |     
    64     |   64    |     
    64     |   128   |     
    128    |   32    |     
    128    |   64    |     
    128    |   128   |     


nohup python3 read_tf.py > output_l90_z50.log 2>&1 &


idx_list = 
[139, 238, 453, 461, 466, 490, 625, 631, 668, 825, 831, 953, 994, 1128, 1146, 1153, 1157, 1158, 1357, 1366, 1367, 1372]

nohup wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 "https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/502046887/" --header "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJlbWFpbF9hZGRyZXNzIjoiZnZvbmRlcmxpcHBlQGdtYWlsLmNvbSIsImlzcyI6IkFQUyBPQXV0aDIgQXV0aGVudGljYXRvciIsImlhdCI6MTY5OTg5MTk0MSwibmJmIjoxNjk5ODkxOTQxLCJleHAiOjE4NTc1NzE5NDEsInVpZCI6ImZpbGlwc2V2ZXJpbiIsInRva2VuQ3JlYXRvciI6ImZpbGlwc2V2ZXJpbiJ9.HpXspqyv0ldi3i7bgqqcUXc-cx2ZQVy1Rp5J2asb8RY" -P . > 2019dec.out 2>&1 &
nohup wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 "https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/502046876/" --header "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJlbWFpbF9hZGRyZXNzIjoiZnZvbmRlcmxpcHBlQGdtYWlsLmNvbSIsImlzcyI6IkFQUyBPQXV0aDIgQXV0aGVudGljYXRvciIsImlhdCI6MTY5OTg5MTM0MSwibmJmIjoxNjk5ODkxMzQxLCJleHAiOjE4NTc1NzEzNDEsInVpZCI6ImZpbGlwc2V2ZXJpbiIsInRva2VuQ3JlYXRvciI6ImZpbGlwc2V2ZXJpbiJ9.Sd0DqwTDVTSA-wAyqirQXWP8SFiXF7OEH4cxZQPx3kE" -P . > 2021okt.out 2>&1 &
