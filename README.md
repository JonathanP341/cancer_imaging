# cancer_imaging
A neural network that is used to determine the presence of a brain tumor on the BraTS 2021 dataset.

The model uses the Unet framework and currently I it for 7 epochs with a smaller subset of the database on an Nvidia 1660Ti GPU. It runs for around 10 minutes. While not perfect, the model could easily be made better with a longer training time and more examples from the dataset. However, I don't own a Data Center so this is good enough for now. I will try to continue making improvements in the meantime besides adding more epochs.

I have run it several times with different models and my most recent model generates these images on the sample scans:

<img width="1464" height="983" alt="image" src="https://github.com/user-attachments/assets/40417313-d2c2-4402-9433-6082ecc2e25b" />


<img width="1103" height="971" alt="image" src="https://github.com/user-attachments/assets/4f1d4bd1-04a5-4e37-82ad-80a3783e9a42" />

Clearly, it's not perfect. But I will continue to update the model and post the most updated screenshots here. 
