#########################
### Home page content ###
#########################

home_general = '''
## General information and links

- The dataset:
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database which includes already developed models, which can be used, downloaded, copied into
a new kaggle jupyter notebook and edited.
- A public GitHub repository:
https://github.com/MarkusFuchs68/Covid-19_detection where all our notebooks can be found.
- An example, of how to approach the problem:
https://drive.google.com/file/d/1OotfOLNfP3wJgutKR2W2oXXDxRiHd-wX/view
'''

home_context = '''
## Context and project description

The aim of this project is to develop an AI solution that will make it possible to identify patients who
have been infected with Covid-19. This is to be achieved by recognising the changes in the lungs
caused by the disease by analysing X-ray images. We have a Kaggle database with X-ray images and a
database with matching masks (links above). More detailed information on the databases, their structure and
potential problems with their analysis can be found in **Analysis**.

After the first modelling iterations using basic models, we tried to improve and further develop these
in the following using various aspects (e.g. model architecture or data augmentation).
For this purpose we created a set of different advanced neural network models, document the
results and tried to improve their performance based on our observations and the advice of our
project mentor. The results can be found in **Modelisation**.

The resulting models can be found in **Model selection**. The models themselves are stored in a Google Drive, from which they can also be downloaded. Just click the 'download models' button there.

After having a model selected, you can go to the **Prediction** page and let the model make predictions on any jung x-ray images. You can either upload an image or specify a URL, where the picture can be retrieved from the internet.
'''

home_samples = '''
## Sample images

Here are some examples of the images and their according classification. They are taken from the Kaggle database. 
The first image is a healthy lung (class: ***normal***), the second one is a lung with pneumonia (class: ***viral_pneumonia***), the third one is a lung with lung opacity (class: ***lung_opacity***) and the fourth one is a lung with Covid-19 (class: ***COVID***).
'''

home_normal = '''#### Class: normal'''
home_viral_pneumonia = '''#### Class: viral_pneumonia'''
home_lung_opacity = '''#### Class: lung_opacity'''
home_covid = '''#### Class: COVID'''


#############################
### Analysis page content ###
#############################

analysis_structure_1 = '''
## Dataset Structure

A first look at the data reveals the following structure:
The current version 5 (as of April 2025) of the dataset COVID-19_Radiography_Dataset contains 4 sets of
chest X-ray images in 4 categorical folders:
- Normal: 10192 images of non-infected lungs
- Viral Pneumonia: 1345 images of other viral infections, which led to a pneumonia
- Lung_Opacity: 6012 images of non-Covid (but other bacteria or virus) infections
- COVID: 3616 images of Covid-19 positive infections
'''

analysis_class_distribution = '''
##### Class Distribution
'''

analysis_structure_2 = '''
Each of the folders holds 2 subfolders:
- images: the X-ray image itself
- masks: a binary mask to filter out only the lungs from the X-ray image

For each X-rax image an according mask image is available. Thankfully the dataset is very
clean regarding the mask images. By checking on sample images, it seems, that all
X-ray-image/mask pairs are well aligned.

For our early modeles we wanted to know: Does masking the images helps to to improve the performance of the to be developed model? 

- Pro arguments: By masking out the non-lung parts, the model will hopefully learn only from the lung image details. 
- Contra arguments: If our model expects masked images, then it will make it harder to apply the model to an x-ray image, because the user must first mask the image.
'''

analysis_masking = '''
Here is an example of an original and masked image:
'''

analysis_structure_3 = '''
Indeed it turned out, that masking the image had a negative effect on the model performance, because the model identified especially the sharp edges of the mask. It is clear, that the edges of the lung does not tell anything about a potential disease. Hence we decided early to train further models solely on unmasked images.
'''

analysis_challenges = '''
## Dataset Challenges
- The images have size 299x299 pixels. The masks have size 256x256 pixels. Henceit is necessary to resize, before the masks can be applied to the x-ray images. Inorder not to loose important details from the x-ray images due to resize-losses, we resize the masksinstead to match the shape of the x-ray images.
- As already mentioned, the masks are rather decreasing model performance than enhancing it. So no challenges regarding the masks.
- The X-ray images show a usually well placed patient, however the lung images are randomly smaller, bigger, larger, rotated a bit etc. Nevertheless we also tried, if data augmentation helps our model to better predict (see **Modelisation**).
- As shown above, the classes are not balanced. We considered this by calculating the class weights and gave them to the models during training in order to compensate the imbalance.
'''

analysis_biases = '''
## Potential Biases
- We don’t know, how trustfully and exact the diagnosis of the X-ray images is. It could be, that we have images labeled as COVID, but actually the diagnosis was wrong or
a data mistake was made and hence the label is just wrong.
We could try to apply an unsupervised approach on all X-ray images and maybe the model identifies COVID and non-Covid (including healthy, bacteria or other virus
infections) themselves. Such a result we could compare to the given labels. On the other hand, for the moment, we trust the data. We can pick up this idea again, if we
see, that the model doesn’t perform for some not-understood reason maybe caused by bias/mistaken labels.
- Some of the pictures show cables (probably from some electrocardiogram device). It might be, that the model gets confused by this. Those cables will be identified by
quite straight lines, so edge detection could identify these. In order to mitigate this, we could consider applying a blur effect in the image preprocessing. We can pick up
this idea again, if we don’t achieve good model performance otherwise. On the other hand, the same principle as with the masks apply: When the model is trained with cables etc. in the images, it might be more robust on such images. For the user there is the advantage again, that NO preprocessing at all on the x-ray image is necessary.
- Other pictures show text in the x-ray image (e.g. to mark what is ‘right’, or dates, etc.). This could confuse the model. Same principle as previously.
- Data origins: The dataset includes metadata, from where the images were collected. After some sample inspections, we see, that there is not much difference in the quality. Again on the other hand, the more diversified the data is, the better our models can generalize and not overfit.
'''


#################################
### Modelisation page content ###
#################################

modelisation = '''
## Modelisation
'''

'''
2_dense121_2-classes_dense512_masked
2_vgg16_2-classes_dense512_masked
2a_vgg16_4-classes_dense512_masked
3a_vgg16_dense512
3b_vgg16_augmented_dense512
3c_vgg16_dense128
3d_efficientnetb1_dense128
4_25ep_medparam_4xconv2d_dense128
4_50ep_medparam_4xconv2d_dense128
4_preproc_effnetb1retrained_dense128
4_preproc_vgg16retrained_dense128
'''


###############################
### Prediction page content ###
###############################

prediction_note = '''
*Note: please remove any uploaded image before entering an image URL*
'''


prediction_or = '''
<div style='text-align: center; padding-top: 2em;'>OR</div>
'''

##########################
### About page content ###
##########################

about = '''
## About
#### Contributors
- Alexander Ückert 
- Markus Fuchs
- Robert Kaspar
- Thomas Klemp

#### Thankyou
- to our project mentor **Souhail Hadgi** for his support and help.
- to **[DataScientest](https://www.datascientest.com/)** for the great training and the opportunity to work on this project.
- to **[Kaggle](https://www.kaggle.com/)** for the great dataset.

#### Image References
- The images used in the app are taken from the Kaggle dataset and are used for educational purposes only.
- Other images were created during our project and are also used for educational purposes only.
'''