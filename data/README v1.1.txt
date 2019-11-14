README v1.0 
by Zhangyang (Atlas) Wang, as of 07/12/2015.




AdobeVFR dataset covers 2,383 classes of popular fonts in graphics design. It is made up of the following four parts:

1. Synthetic Data

1.1 VFR_syn_train
1.2 VFR_syn_val

To create a sufficiently large set of synthetic training data, we render long English words sampled from a large corpus, and generate tightly cropped, gray-scale, and size-normalized text images. For each of the 2383 classes, we assign 1,000 images for training, and 100 for validation, which are denoted as VFR_syn_train and  VFR_syn_val, respectively. 


2. Real-world Data

2.1 VFR_real_test
2.2 VFR_real_u 

We collected 201,780 text images from various typography forums, where people post these images seeking help from experts to identify the fonts. Most of them come with hand-annotated font labels which may be inaccurate. Unfortunately, only a very small portion of them fall into our list of 2,383 fonts. All images are first converted into gray scale. Those images with our target class labels are then selected and inspected by independent experts if their labels are correct. Images with verified labels are then manually cropped with tight bounding boxes and normalized proportionally in size, to be with the identical height of 105 pixels. Finally, we obtain 4,384 real-world test images with reliable labels, covering 617 classes (out of 2,383), as the VFR_real_test set. Compared to the synthetic data, these images typically have much larger appearance variations caused by scaling, background clutter, lighting, noise, perspective distortions, and compression artifacts. 

Removing the 4,384 labeled images from the full set, we are left with 197,396 unlabeled real-world images which we denote as VFR_real_u. In our paper, those unlabeled data were utilized to pre-train a ``shared-feature'' extraction subnetwork to reduce the domain gap.


[1.0]
In current release, we ONLY INCLUDE VFR_real_test and VFR_real_u, due to certain IP concerns on generating and releasing the synthetic data from copyrighted fonts. 

We instead include a fontlist that speicifies the 2,383 font classes that we used. With an Adobe Creative Cloud subscription, one could download those font files (.otf) and render (unlimited) synthetic images.

We are still working hard to get our original VFR_syn_train and VFR_syn_val sucessfully released. Before then, if you're interested in following our expeirments, please don't hesistate to send an email to: masterwant@gmail.com and I'll try my best to help.

[1.1]
Synthetic data added for training and validation, in bcf format. To see how to process bcf data, sample codes are provided to be used with cuda-convenet. Note those are only my own experiment codes and are not in any way optimized for product-level usage.







