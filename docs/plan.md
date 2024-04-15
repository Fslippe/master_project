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

#### Wind direction and transition extraction

- Using minimum threshold for sizes of label areas to mask unsure predictions.
- Using Douglas Peucker algorithm to get lines that can be used as angles.
  - Problem: data not ordered on a line, but in terms of position in 2d-Space
- Using linear regression on closest datapoints
  - REMBEMBER TO USE max distance for closest points to count towards linreg
  - Problem: Linreg can be combination of different orientations that are orthogonal, Using what is wrong to fix it gives just more problems
- Using Wind direction to check if there are boundaries more upwind.
  - Use average wind direction of CAO / open cells
    - Inaccurate
  - Follow contour lines
    - Problems:
      - super computationally intensive
      - Hard to define what the contour lines should "hit" - a regression line or a point? - Either more computation or innacuracy because of holes in dataset. One also have to interpolate to closest grid cell of satellite swath.
  - Step with wind direction and check if any other points closer than original point.
    - if so check angle between new point and that point and see if it is in the direction of the wind +/- some threshold
    - Problems:
  - # Step with wind 100km
    - Check angle with every points.
    - Only append for those who dont have points more upwind inside a threshold +/-5 degrees
  - Make threshold check between every angle of all points and compare to wind_dir
  - Step with wind direction and check if any other points closer than before.
  - Step one step with wind direction and check if any angle of other points matches up wind wind_direction of new point +/- some threshold.
  - Problems for all
    - Computationally intensive

#### Reanalysis boundary extraction

- get all reanalysis points closest to boundary and step with wind from this.

#### TEST MATRIX

-- TESTING accuracy metric

- Problem is different number of clusters
  - Solution: Cluster metric algorithm
  - Use ex 12 Clusters for all tests, to see performance - Of cource very limited - Could use 11, 12, 13

---

|**n_K**|**\_\_\_**Filters**\_\_\_**|
|**\_\_\_**|**32**|**64**|**128**|  
| 10 | | | |
| 11 | | | |
| 12 | | | |
| 13 | | | |
| 14 | | | |
| 15 | | | |
|**16\_**|**\_\_**|**\_\_**|**\_\_\_**|

#### Calculate scores

- Find way of drawing area
  - make brush
  - Make lower resolution grid and average over every grid box
  - Make same brush for model output and append to grid
    - Important to give better scores to models being close than those that are completely off.
- Scores
  - calculate accuary scores
  - Weight the accuracies by multiplying by highest truth probability (More people predicting one area, More likely to be right)
    - Multiply over whole score, or just the areas
    - Multiply when everyone agrees on nothing.
  - Border score
    - Change calculation to be percentage of border inside drawn border
  - should post processing be used?

idx_list =
[139, 238, 453, 461, 466, 490, 625, 631, 668, 825, 831, 953, 994, 1128, 1146, 1153, 1157, 1158, 1357, 1366, 1367, 1372]

nohup wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 "https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/502046887/" --header "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJlbWFpbF9hZGRyZXNzIjoiZnZvbmRlcmxpcHBlQGdtYWlsLmNvbSIsImlzcyI6IkFQUyBPQXV0aDIgQXV0aGVudGljYXRvciIsImlhdCI6MTY5OTg5MTk0MSwibmJmIjoxNjk5ODkxOTQxLCJleHAiOjE4NTc1NzE5NDEsInVpZCI6ImZpbGlwc2V2ZXJpbiIsInRva2VuQ3JlYXRvciI6ImZpbGlwc2V2ZXJpbiJ9.HpXspqyv0ldi3i7bgqqcUXc-cx2ZQVy1Rp5J2asb8RY" -P . > 2019dec.out 2>&1 &

### LABELING SESSION

- Generate dataset
  - 25% predicted by model, 25% not predicted by model, 25% random sample, 25% block randomly sampled from the three others (10, 10, 10, 10)\*5 years
  - Get visible band sets of these
  - compress to jpg - conserve pixel ratio
  - Download data

#### DOWNLOADED MERRA PARAMETERS

CLDTMP = cloud top temperature
TQI = total precipitable ice water
TQL = total precipitable liquid water
TQV = total precipitable water vapor
TS = surface skin temperature
so4 = sulfate

#### FIND CAO cases to use

- run clustering on all years, and save times for those with more than threshold cao labels
- run on these again with a stride of 16 or 32
- get closest locations to open and closed cell labels on MERRA grid and save
- get closest locations to boundary on MERRA grid and save
- use indexes to extract

#### ANALYSIS OF EXTRACTED DATA

- Plot histograms
- Perform significance tests - H0: data from same distribtutions?
- Perform random forest with prediction labels of 0/1/2 closed/border/open
- Accurate model can be used to find most important predictors for border and if of interest open/closed
- Using EIS
- modis LWP closed - open cell mean / std
  - Comment: this is really cool and great that you checked Filip. Quick question on the LWP retrieval, is this based on emissivity or primarily on reflected solar? I ask as in my mind, at higher latitudes, even though you would effectively be seeing a longer cloud path, you would also expect more sunlight to be forward scattered and thus modis would see less solar radiation and a lower albedo. If this is the case, then you would register a lower LWP for closed cells/ more north clouds, as you seem to be seeing

### GENERAL PLANS

- Plots of labeled dataset to gether with model
- t

### Possible research questions and analysis

-


### TODAYS PLAN
- Use dict list to generate new histogram climatologies 
- Use dict list to generate timeline of CAO in FRAM strait 
- Use dict list to generate monthly climatology of example year
- Extract dict_list for filter 128 
- Writing....
- need to further explain the 6 images used for finding open/closed cell labels


### THESIS STRUCTURE
- background and theory 
  - including machine learning techniques 
  - 
- Method
  - Only to include how I used machine learning and how I calculated scores +++ 

  - Explanation of chossing NN model 
  - Citizen science 
    - Generating dataset
    - Lavterskel session
    - Expert panel 
  - Scores 
    - including wind stepping algorithm
  - Removal of smaller areas 
  - Meteorology 
    - Open and closed cell following established research 
      - Climatology histograms (+ ex Fram strait / seasonal histograms)
      - Reanalysis histograms (open - closed cell distibution differences) 
    - Wind stepping histograms
      - Using LWP/IWP histograms to show the need
      - How wind stepping is preformed
- Results and discussion
    





### COMMANDS

wget --http-user=filip --http-password=yBPxPYhWcUeZZgKzkVyMthWAT5+3sUImcsK+dMJWb0I https://filip-master.vercel.app/results

scp subset*M2T1NXSLV_5.12.4_20240201_124036*.txt nird:/nird/projects/NS9600K/data/MERRA/subset*M2T1NXSLV_5.12.4_20240201_124036*.txt

wget --load-cookies ~/.urs*cookies --save-cookies ~/.urs_cookies --keep-session-cookies --content-disposition -i subset_M2T1NXSLV_5.12.4_20240201_124036*.txt

rsync -av --progress /source/directory user@remote:/destination/directory

rsync -av --progress slippe@login.nird.sigma2.no:/nird/projects/NS9600K/data/MERRA/202301_new/  /uio/hume/student-u37/fslippe/MERRA/202301_new/        

  #fslippe@mimi.uio.no:/scratch/fslippe/MERRA/

nohup python3 read_tf.py 128 256 > log_outs/output_ps128_f256.log 2>&1 &

nohup python3 read_tf_old.py > log_outs/output_ps128_f128_old.log 2>&1 &

rsync -av --progress /scratch/fslippe/modis/MOD02/training_data/tf_data/dnb_l95_z50_ps128_band29 fslippe@login.nird.sigma2.no:/nird/projects/NS9600K/fslippe/mimi_backup/




nohup wget wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 "https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/502158532/" --header "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJlbWFpbF9hZGRyZXNzIjoiZnZvbmRlcmxpcHBlQGdtYWlsLmNvbSIsImlzcyI6IkFQUyBPQXV0aDIgQXV0aGVudGljYXRvciIsImlhdCI6MTcxMTcyMDk1NSwibmJmIjoxNzExNzIwOTU1LCJleHAiOjE4Njk0MDA5NTUsInVpZCI6ImZpbGlwc2V2ZXJpbiIsInRva2VuQ3JlYXRvciI6ImZpbGlwc2V2ZXJpbiJ9.TCfvDNyNPiqdNzWIViiXmyhseF1ZsiOMwoHYreVQQk8" -P . &
