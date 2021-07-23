# 3D Left Atrium Segmentation and Triangulation using 3D Unet and Marching Cubes.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Acknowledgements
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

I would like to express my heartfelt gratitude to my supervisor, Dr. Chris Cantwell, for providing the
opportunity to work with him on this Cardiac Electrophysiology research project. Thank you for introducing
me to Machine Learning, for your continual support and advice over the last three years, and for helping me realise my
passion for Deep Learning and Artificial Intelligence.

My sincere thanks also goes to Dr. Francesco Montomoli, whose feedback during the interim was critical in
refining and reshaping this project.

I owe gratitude to my friends and family, who were always supportive of me. Thank you
for encouraging me to study in London and for your unwavering support during my term at Imperial. I would
especially like to thank my cousin for his guidance and continual support during the COVID-19 pandemic.

I would also like to thank Dr Sreenivas Bhattiprolu. Please visit his GitHub page: https://github.com/bnsreenu/python_for_microscopists
and youtube channel: https://www.youtube.com/channel/UC34rW-HtPJulxr5wp2Xa04w. I learnt absolutely incredible and wholistic concepts from Dr Sreeni via his youtube tutorials that cover Deep Learning for microscopists. 

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Installation Guide
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0. Create a virtual env with python 3.6.8
1. Clone this repository on your local machine.
2. Install the packages in requirements.txt via pip or conda.
3. Place the data in the appropriate folders described in the next section.
4. Run the script train3d.py in the main directory to output a model and a prediction on test data. Open train3d.py and edit hyperparameters to change the default values before running. Note that the train3d.py comes with preloaded weighs from the training I did on my dataset. This will save you an incredible amount of training time. 
6. Run the script Meshgen.py in the main directory to output a triangulation based on the predictions output in the previous step. Note this script contains a loop hence once needs to be run once at the end and will automatically generate meshes from all the subfolders in ./test3d/
7. The output will appear in the .tests3d/ folder with a unique folder name that contains the date, time and hyperparameters such as no of epochs etc. It's fairly intuitive. This is done to organise the different testing runs and avoid having to move files, every time the network needs to be trained.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Files
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

simple_Unet_3d - this file contains a 3D unet architecture. This is already configured so should work out of the box but can edit optimizer/Loss function/CNN layers here. 
Meshgen - this file contains the marching cube algorithm and will generate meshes from all the predictions made in the subfolders ./tests3d/XXXXXX/prediction.nii to output a .stl surface mesh in the same location: ./tests3d/XXXXXX/mesh.stl.
train3d - this file contains the main script that calls on the model, trains it, and outputs the result. Edit this script to change all the hyperparameters. This file includes a data augmentation func that will translate each data sample a particular number of pixels in the x and y direction and add this augmented dataset to the original, hence doubling the effective amount of data available to the network. Note this will take much longer to train and will use a much greater memory. 
dataloader - this file contains a script to convert a .mhd/raw MRI scan to a 3D numpy array to be fed into the network. Note edit this file if your data is in a different format. I will update this file to include other datatypes like .nii.gz , .nrrd , .dcm etc in the future. 

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Folder Structure
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The output of the train.py script is going to be saved in a subfolder in ./tests3d/. This output will contain a sample prediction on an external image saved as nifty file: prediction.nii , validation and training loss curves in png format, and a 2D slice from the midpoint of the dataset to check that the masks and images are correctly aligned.
......**** Going to update and complete the documentation by 01/09/2021**** in the meantime, if you have any questions, feel free to contact me @ rutvik239@gmail.com / +447432692807
