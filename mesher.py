import vtk
from vtk.util import numpy_support
import os
import numpy
import plotly
from plotly.graph_objs import *
import chart_studio.plotly
chart_studio.plotly.sign_in("somada141", "1t2qb5b9y1")

# https://pyscience.wordpress.com/2014/09/11/surface-extraction-creating-a-mesh-from-pixel-data-using-python-and-vtk/
# https://nbviewer.jupyter.org/urls/bitbucket.org/somada141/pyscience/raw/master/20140908_SurfaceExtraction/Material/SurfaceExtractionVTK.ipynb


def vtkImageToNumPy(image, pixelDims):
    pointData = image.GetPointData()
    arrayData = pointData.GetArray(0)
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    ArrayDicom = ArrayDicom.reshape(pixelDims, order='F')

    return ArrayDicom


def plotHeatmap(array, name="plot"):
    data = Data([
        Heatmap(
            z=array,
            showscale=True
        )
    ])
    layout = Layout(
        autosize=False,
        title=name
    )
    fig = Figure(data=data, layout=layout)

    return chart_studio.plotly.plot(fig, filename=name)


from IPython.display import Image
def vtk_show(renderer, width=400, height=300):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.
    """
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(width, height)
    renderWindow.Render()

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    data = str(memoryview(writer.GetResult()))

    return Image(data)

#DICOM INPUT
PathDicom = "gt_binary.mhd"
reader = vtk.vtkMetaImageReader()
reader.SetFileName(PathDicom)
reader.Update()

 # Load dimensions using `GetDataExtent`
_extent = reader.GetDataExtent()
ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

# Load spacing values
ConstPixelSpacing = reader.GetPixelSpacing()

#shiftScale = vtk.vtkImageShiftScale()
#shiftScale.SetScale(reader.GetRescaleSlope())
#shiftScale.SetShift(reader.GetRescaleOffset())
#shiftScale.SetInputConnection(reader.GetOutputPort())
#shiftScale.Update()

# In the next cell you would simply get the output with 'GetOutput' from 'shiftScale' instead of 'reader'

ArrayDicom = vtkImageToNumPy(reader.GetOutput(), ConstPixelDims)

#=========================================
#import skimage.io as io
#ArrayDicom = io.imread('gt_binary.mhd', plugin='simpleitk')
#=========================================
#import SimpleITK as sitk
#import matplotlib.pylab as plt
#ArrayDicom = sitk.GetArrayFromImage(sitk.ReadImage("gt_binary.mhd", sitk.sitkFloat32))
#=========================================checking if the file is not empty
#numpy.set_printoptions(threshold=numpy.inf)
#print((ArrayDicom))
#=========================================

plotHeatmap(numpy.rot90(ArrayDicom[:,256, :]), name="Original Scan")

threshold = vtk.vtkImageThreshold ()
threshold.SetInputConnection(reader.GetOutputPort())
threshold.ThresholdByLower(0.5)  # remove all soft tissue
threshold.ReplaceInOn()
threshold.SetInValue(0)  # set all values below 400 to 0
threshold.ReplaceOutOn()
threshold.SetOutValue(1)  # set all values above 400 to 1
threshold.Update()

ArrayDicom = vtkImageToNumPy(threshold.GetOutput(), ConstPixelDims)
plotHeatmap(numpy.rot90(ArrayDicom[:, 256, :]), name="CT_Thresholded")

#%%time
dmc = vtk.vtkDiscreteMarchingCubes()
dmc.SetInputConnection(threshold.GetOutputPort())
dmc.GenerateValues(1, 1, 1)
dmc.Update()

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(dmc.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(1.0, 1.0, 1.0)

camera = renderer.MakeCamera()
camera.SetPosition(-500.0, 245.5, 122.0)
camera.SetFocalPoint(301.0, 245.5, 122.0)
camera.SetViewAngle(30.0)
camera.SetRoll(-90.0)
renderer.SetActiveCamera(camera)
vtk_show(renderer, 600, 600)

camera = renderer.GetActiveCamera()
camera.SetPosition(301.0, 1045.0, 122.0)
camera.SetFocalPoint(301.0, 245.5, 122.0)
camera.SetViewAngle(30.0)
camera.SetRoll(0.0)
renderer.SetActiveCamera(camera)
vtk_show(renderer, 600, 600)

writer = vtk.vtkSTLWriter()
writer.SetInputConnection(dmc.GetOutputPort())
writer.SetFileTypeToBinary()
writer.SetFileName("bones.stl")
writer.Write()

renderer_window = vtk.vtkRenderWindow()
renderer_window.AddRenderer(renderer)
renderer_interactor = vtk.vtkRenderWindowInteractor()
renderer_interactor.SetRenderWindow(renderer_window)
renderer.SetBackground(1,1,1)
renderer_window.SetSize(1024, 768)

renderer_interactor.Initialize()
renderer_window.Render()
renderer_interactor.Start()
