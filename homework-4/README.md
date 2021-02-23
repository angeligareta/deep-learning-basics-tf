# Detection Assignment

In this final assignment the task is not only to classify traffic signs but to locate them beforehand in the whole image, so a detection task is added. The followed approach was based on Faster R-CNN, in which we have to follow two stages. First we had to predict class-agnostic bounding box proposals using a Region Proposal Network (RPN) and then classify and fine-regress location of boxes. In order to classify we could use the information for the previous assignments as the object to categorize is the same: traffic signs.

The RPN was already provided so our focus was in discarding bounding boxes that were not traffic
signs. However, apart from that we hypertuned the parameters for the RPN in order to generate shapes that were adequate and in line with our ground truth. As previously explained in the description of the detection dataset GTSDB (1.2.1) the traffic signs could appear in sizes from 16x16 to 128x128 and the shape was almost squared, so this data was useful in order to choose the best parameters for the RPN.

## Implementation

The full implementation details can be found in the [final report](final_report.pdf), for a better understanding we encourage you to read it.

### RPN Tuning

Fist of all, we needed a way of seeing how accurate was the RPN generating bounding boxes, so after
their generation, we developed a script that counted the number of ground truth from only the training boxes where there was at least one bounding box intersecting with an IoU (Intersection over Union) higher than 0.7. We decided to use that threshold to be more severe in the hyperparameter tuning and generate boxes that then the CNN could classify better. After having that number we divided it by the total number of bounding boxes for training, resulting in the precision accuracy of the RPN.

### Bounding Boxes Reduction

In the second part of the assignment we focused on how to reduce the number of bounding boxes that
were generated. The goal was to improve the mAP (mean Average Precision), which is the average of
the individual average precision per class, which takes into account the IoU, precision and recall. Hence, by reducing the number of false positives we would obtain higher precision and higher mAP, so that was our first approach by using a position and shape filter.

### [Background CNN](traffic_sign_detection_tf_cnn.ipynb)

The next step in our detector was to implement a CNN that was able to classify bounding boxes that
contain background and traffic signs. In order to obtain the data for the CNN, we would use the train data from GTSDB and GTSRB. However, to obtain background data, we decided to use the generated bounding boxes. The approach was the following:

1. Loop over the generated bounding boxes for train_images.
2. To differentiate between background and traffic sign, we calculated the IoU of the bounding boxes with the ground truth, and if it was higher than a threshold T, we considered them traffic sign or otherwise background data.
3. For each image, take N boxes that refer to background. We did that by throwing a dice with a probability (n / number of bounding boxes) to have a representative distribution of the background data for all the images.
4. For each image, take all the possible images for traffic signs as the ratio is uneven and add them to the GTSDB and GTSRB train data. This will be useful because its a real example of the images that our detector is going to see.
5. Repeat until having 50.000 data for each class.

### [Traffic Sign CNN](traffic_sign_detection_tf_cnn.ipynb)

The second step was to improve the given traffic sign classifier, because, as previously explained in this same section, a higher precision contributed to a higher mAP. In order to do so, we decided to further improve the CNN of the previous assignment by training it with more traffic signs labeled data, through using the GTSRB (seen in section 1.3) database.

## Results

The complete hypertuning can be found at [traffic_sign_detection_hypertuning.csv](traffic_sign_detection_hypertuning.csv). The best results in respect of ratio precision-speed were obtained with the parameters found below. With this configuration we obtained a final mAP of 66.28% in 2183 seconds, achieving one of the best scores for the assignment.

### RPN Details

- Anchor ratios: (0.8, 1.0, 1.2)
- Anchor sizes: (32, 62, 112, 212, 252)
- Number of bboxes before NMS: 10000
- Number of bboxes after NMS: 1500
- NMS threshold: 0.

### Filters and models hyperparameters:

- Min size allowed: 16
- Max size allowed: 130
- Probability threshold Traffic Sign CNN: 0.
- Probability threshold Background CNN: 0.
- IoU threshold: 0.

## Conclusion

Through this assignment we have comprehended the importance of the Region Proposal Generator in
the two stages approach. By tuning the parameters of the RPN we achieved better and more realistic
bounding boxes, despite probably changing the generator weights or model could have resulted in a
faster detector. On top of that, the metrics of the train data such as size, shape and position heatmap were essential to achieve our final score and that can be interpolated to any Machine Learning problem, analyze the data is crucial. Finally, the CNN able to distinguish between background images and traffic sign allowed us to have a faster and more accurate model.

## Authors

- Student Name 1: Stefano Baggetto
- Student Name 2: Giorgio Segalla
- Student Name 3: Angel Igareta ([angel@igareta.com](angel@igareta.com))
