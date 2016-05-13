# Image_Processing
Projects that utilize the OpenCV library. Coded in Python.

#### Project 1 - Histograms
Takes in an image and calculates the histogram. 
######Input: 
  -Image file

######Output:
  -Histogram calculation summed
  
#### Project 3 - Kernel Filter, Median & Gaussian Filter Noise Reduction, Damaged Image Masking
#####kernel_filter.py
Takes in an input image, applies kernel with filter and displays output.
######Input: 
  -Image file 

######Output:
  -Display with output image
#####filter_noise_reduction.py
Takes in an input image, performs median and gaussian filter to remove noise. 
######Input: 
  -Image file 

######Output:
  -Display with output images
  
#####masking.py
Takes in an input and mask image. Determines location of damaged pixels and uses Gaussian Blur to replace the pixels.
######Input: 
  -Image file 
  
  -Mask file

######Output:
  -Display with damaged pixels removed from input image
  
#### Project 5 - Filters,Canny Edge Detection
#####filter2d.py
Takes in an input image, applies filter on x,y then merges and outputs the image.
######Input: 
  -Image file 

######Output:
  -Display with output image
#####canny_edge_detection.py
Takes in an input image, provides trackbars to adjust minimum and maximum for Canny Edge Detection.
######Input: 
  -Image file 

######Output:
  -Display with trackbars and output image

#### Project 6 - Key Points
Takes in two images and uses SIFT detection and Lowe's ratio test to stitch the result image. 
######Input: 
  -Two image files 

######Output:
  -Display with input and output stitched image

#### Project 7 - Channels
Takes in a grayscale image, grabs the RGB channels and reconstructs the image.
######Input: 
  -Image file 

######Output:
  -Display with reconstructed output image

#### Project 8 - Detection, Thresholding, K-Means Clustering
#####kmeans_clustering Function
Takes in an image and performs K-Means Clustering.
######Input: 
  -Image file

######Output:
  -Display with K-Means Clustering performed
#####thresholding Function
Takes in an image and performs Image Thresholding using OTSU.
######Input: 
  -Image file

######Output:
  -Display with image thresholding performed
#####circle_detection Function
Takes in an image and performs Hough Circle Transform to detect circles.
######Input: 
  -Image file

######Output:
  -Display with detected circles on input image

  
#### Project 9 - Wavelet Fusion
Takes in two images and performs Wavelet CDF 9/7 and fuses them into one resultant image.
######Input: 
  -Two image files 

######Output:
  -Display with input and output images
  
  -Merged image saved to project location

#### Project 10 - Image Tracking
Takes in a sequence of frames as well as interest points and tracks it.
Also utilizes Lucas-Kanade optical flow method.
######Input: 
  -Ground truth file
  
  -Sequence of images frame by frame
######Output:
  -Display with bounding box of currently tracked area
  
  -Mean FPS
  
  -Mean Error
  
  -Total Frames
