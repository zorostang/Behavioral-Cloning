My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode which has been slightly modified so pleasae use this one and not your own
* model.h5 containing a trained convolutional neural network
* this readme.md

model.py contain the code for preprocess and training the nearual network

### Model Archetecture and Strategy

#### Final Archetecture
This model consists of a CNN composed of 4 convolutional layers with 5 x 5 filters, exponential linear units between each layer (ELU). Then I flatten the inputs before feeding them to a fully connected layer. The output of the fully connected layer is 512. I feed those output to one final fully connected layer which gives me the predicted steering angle. I used an adam optimizer so that manually training the learning rate wasn't necessary. I trained the model for 9 epochs. The number of samples per epoch was set equal to the total number training points (~27,000). 

#### Overfitting
I've included a total of three dropout layer to help prevent overfitting. I don't think my model overfit the data because both training and validation loss (Mean Squared Error) continued to decrease through every epoch. 

#### Training Data
Initial I was using the data given by Udacity. However, I also came across a much larger dataset through the slack channel, and ended up training on that one. Training off the larger dataset helped the model finally train well enough to drive autonomously around the track.

#### Model Performance
I've spent a great deal of time trying to implement different models like comma.ai, VGG16, Lenet-5, and custom. For this submission my car is able to make it around the track on some runs, but not every one. I believe next steps in this model would be finetuning, but I'm not sure what would be the best start. I don't think I need more data. That means I could try tuning hyperparamters that feed into the model. The first thing I might try next if I have time would be to increaase the starting image size (gain more parameters) and increase the number of training epoch by 2x.

#### Other Design Approach Notes
I started with the comma.ai model because it trained much faster than the VGG16 and I wanted a faster feedback loop. The greatest challenge in the beginning was keeping all the moving parts organized. This type of project requires very clean, organized code. Something that I know I could still improve on. Eventually, I got the car driving past the first hard curve, but I was stuck for a very long time on the soft curve right after that. So I gathered additional training data for that partiular curve, and also gathered additional recovery data.

Eventually I ended up throwing out that data and sticking with the large dataset shared on slack. That dataset had around 27,000 datapoints

#### Preprocessing
This project preprocess the images in the following ways:
* crop the top and bottom of the imgage to focus on the road
* flip the image on the vertical axis 50% of the time
* change the brightness randomly
* resize the image to 32 X 64

I orginally started this project with many more preprocessing steps than I use now, but found that they were not helping as much as the model archetecture and training data.

#### Things to try if I have more time
* add max or average pooling after ever convolutional layer
* figure out best starting approach for "fine tuning" model at this point, because I don't think more data collection is necessaryS.

#### Response to Feedback
10 Example images from the data set have been include in the output_images directory. I've also saved the jupyter notebook as html so that you may view the sample images along with their respective angles.
![sample](/output_images/sample_image_1.jpg "Sampe Image")

A histogram of the collected data has now been included in the jupyter notebook. Note this dataset was large and 0 angles were not filtered out. Upon initial inspection I noticed that I could not even see the non-zero angle in the original histogram, so then I plotted it on a log scale. You can see that there are 400x more 0 angle images than non-zero angle images! This mean the training data was not well balanced. This also might explain why my car did not perform well on track two. 
![Histograms](/output_images/histogram.png "Histograms")

The network  is clearly biased towards zero. If I had more time, I would balance the dataset better and then retrain the model without changing anything else to see if the car drives better on track 1 and track 2 with balanced training data.

Furthermore, from the histogram, we can also see that there are more negative counts than positive counts. This was easily balanced in the training pipeline by randomly flipping the images. 



