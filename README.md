# Image_Processing
Projects that utilize the OpenCV library. Coded in Python.

#### Project 8 - Detection, Thresholding
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
