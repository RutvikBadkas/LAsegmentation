# Atriumsegmentation
UPDATE 25/04/2021 - I am currently training the model on HPC servers (96GB ram, 16 CPU cores, 4xRTX6000). Will add a commit once training is complete. Also going to build a class/function to augment a Numpy dataset to add additional functionality to the package as a whole.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Hey there! I am building a pipeline to go from an MRI scan to a Finite Element Mesh using Tensorflow, Unet, Numpy, Scikit... bla bla bla
I am still experimenting and refining so the repo will be a bit messy for now.

I have compiled and tested my code locally but am refining the project so that all of you can easily utilize my framework for your experiments so stay tuned, I will also write a comprehensive documentation so you can easily learn to import your data and train the model. I Will be done with everything by 15th May 2021.

Publishing My Research Paper on 8st June 2021

Stages
1. Segmentation using Unet Based CNN ..................................done
2. Convert output to stl file or set of coordinates of choice..........done
3. Create Surface or Volume Mesh using Gmsh or Nekmesh.................done
