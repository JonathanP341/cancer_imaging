# cancer_imaging
A neural network that is used to determine the presence of a brain tumor on the BraTS 2021 dataset.

The model uses the Unet framework and currently I it for 17 epochs with a smaller subset of the database on an Nvidia 1660Ti GPU. It runs for around 15 minutes. While not perfect, the model learns at a very constant rate and could be made better with more resources and training time. However, I don't own a Data Center so this is good enough for now. I will try to continue making improvements in the meantime besides adding more epochs.

- Loss: Dice Loss(90%) + BCE Loss(10%)
- Final Valiation Loss: 0.8203
- Batch Size: 4
- Epochs: 17
- Optimizer: AdamW
- Learning Rate: 1e-4
- Weight Decay: 1e-5


I have run it several times with different models and my most recent model generates these images on the sample scans:

<img width="1426" height="979" alt="image" src="https://github.com/user-attachments/assets/a5b67cdb-67e5-454d-9874-265f9cc6d46a" />
Loss: 0.91

<img width="1428" height="983" alt="image" src="https://github.com/user-attachments/assets/ed04ec55-5614-4eff-b5a5-30fd023245d7" />
Loss: 0.97

Clearly, it's not perfect. But I will continue to update the model and post the most updated screenshots here. 
