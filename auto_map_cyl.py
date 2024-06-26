#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Legacy VTK Reader'
coilsvtk = LegacyVTKReader(FileNames=['/home/ssharpe/REALTA/REALTA 21.05.24/coils.vtk'])

# create a new 'Legacy VTK Reader'
radvtk = LegacyVTKReader(FileNames=['/home/ssharpe/REALTA/REALTA 21.05.24/rad_b_new.vtk'])

# set active source
SetActiveSource(coilsvtk)

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [2265, 1037]

# show data in view
coilsvtkDisplay = Show(coilsvtk, renderView1)
# trace defaults for the display properties.
coilsvtkDisplay.Representation = 'Surface'
coilsvtkDisplay.ColorArrayName = [None, '']
coilsvtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
coilsvtkDisplay.SelectOrientationVectors = 'None'
coilsvtkDisplay.ScaleFactor = 1.968
coilsvtkDisplay.SelectScaleArray = 'None'
coilsvtkDisplay.GlyphType = 'Arrow'
coilsvtkDisplay.GlyphTableIndexArray = 'None'
coilsvtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
coilsvtkDisplay.PolarAxes = 'PolarAxesRepresentation'
coilsvtkDisplay.ScalarOpacityUnitDistance = 0.6817433060777606

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
coilsvtkDisplay.OSPRayScaleFunction.Points = [-2.777055759186478e-23, 0.0, 0.5, 0.0, 1.0000001192092896, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera()

# set active source
SetActiveSource(radvtk)

# show data in view
radvtkDisplay = Show(radvtk, renderView1)
# trace defaults for the display properties.
radvtkDisplay.Representation = 'Surface'
radvtkDisplay.ColorArrayName = [None, '']
radvtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
radvtkDisplay.SelectOrientationVectors = 'None'
radvtkDisplay.ScaleFactor = 1.4400000000000002
radvtkDisplay.SelectScaleArray = 'None'
radvtkDisplay.GlyphType = 'Arrow'
radvtkDisplay.GlyphTableIndexArray = 'None'
radvtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
radvtkDisplay.PolarAxes = 'PolarAxesRepresentation'
radvtkDisplay.ScalarOpacityUnitDistance = 0.32523651279827276

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
radvtkDisplay.OSPRayScaleFunction.Points = [-2.777055759186478e-23, 0.0, 0.5, 0.0, 1.0000001192092896, 1.0, 0.5, 0.0]

# show data in view
coilsvtkDisplay = Show(coilsvtk, renderView1)

# show data in view
radvtkDisplay = Show(radvtk, renderView1)

# create a new 'CSV Reader'
b_totcsv = CSVReader(FileName=['/home/ssharpe/GITHUB/SBRI-Year-2/mag_field_calc/B_tot.csv'])

# Create a new 'SpreadSheet View'
spreadSheetView1 = CreateView('SpreadSheetView')
spreadSheetView1.ColumnToSort = ''
spreadSheetView1.BlockSize = 1024L
# uncomment following to set a specific view size
# spreadSheetView1.ViewSize = [400, 400]

# get layout
layout1 = GetLayout()

# place view in the layout
layout1.AssignView(2, spreadSheetView1)

# show data in view
b_totcsvDisplay = Show(b_totcsv, spreadSheetView1)
# trace defaults for the display properties.
b_totcsvDisplay.FieldAssociation = 'Row Data'

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
spreadSheetView1.Update()

# create a new 'Table To Points'
tableToPoints1 = TableToPoints(Input=b_totcsv)
tableToPoints1.XColumn = 'Bmag'
tableToPoints1.YColumn = 'Bmag'
tableToPoints1.ZColumn = 'Bmag'

# Properties modified on tableToPoints1
tableToPoints1.XColumn = 'x'
tableToPoints1.YColumn = 'y'
tableToPoints1.ZColumn = 'z'

# show data in view
tableToPoints1Display = Show(tableToPoints1, spreadSheetView1)
# trace defaults for the display properties.
tableToPoints1Display.FieldAssociation = 6

# hide data in view
Hide(b_totcsv, spreadSheetView1)

# update the view to ensure updated data information
spreadSheetView1.Update()

# set active view
SetActiveView(renderView1)

# set active source
SetActiveSource(tableToPoints1)

# show data in view
tableToPoints1Display_1 = Show(tableToPoints1, renderView1)
# trace defaults for the display properties.
tableToPoints1Display_1.Representation = 'Surface'
tableToPoints1Display_1.ColorArrayName = [None, '']
tableToPoints1Display_1.OSPRayScaleArray = 'Bmag'
tableToPoints1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
tableToPoints1Display_1.SelectOrientationVectors = 'Bmag'
tableToPoints1Display_1.ScaleFactor = 2.1
tableToPoints1Display_1.SelectScaleArray = 'Bmag'
tableToPoints1Display_1.GlyphType = 'Arrow'
tableToPoints1Display_1.GlyphTableIndexArray = 'Bmag'
tableToPoints1Display_1.DataAxesGrid = 'GridAxesRepresentation'
tableToPoints1Display_1.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
tableToPoints1Display_1.OSPRayScaleFunction.Points = [-2.777055759186478e-23, 0.0, 0.5, 0.0, 1.0000001192092896, 1.0, 0.5, 0.0]

# set active view
SetActiveView(spreadSheetView1)

# destroy spreadSheetView1
Delete(spreadSheetView1)
del spreadSheetView1

# close an empty frame
layout1.Collapse(2)

# set active view
SetActiveView(renderView1)

# create a new 'Delaunay 3D'
delaunay3D1 = Delaunay3D(Input=tableToPoints1)

# show data in view
delaunay3D1Display = Show(delaunay3D1, renderView1)
# trace defaults for the display properties.
delaunay3D1Display.Representation = 'Surface'
delaunay3D1Display.ColorArrayName = [None, '']
delaunay3D1Display.OSPRayScaleArray = 'Bmag'
delaunay3D1Display.OSPRayScaleFunction = 'PiecewiseFunction'
delaunay3D1Display.SelectOrientationVectors = 'Bmag'
delaunay3D1Display.ScaleFactor = 2.1
delaunay3D1Display.SelectScaleArray = 'Bmag'
delaunay3D1Display.GlyphType = 'Arrow'
delaunay3D1Display.GlyphTableIndexArray = 'Bmag'
delaunay3D1Display.DataAxesGrid = 'GridAxesRepresentation'
delaunay3D1Display.PolarAxes = 'PolarAxesRepresentation'
delaunay3D1Display.ScalarOpacityUnitDistance = 0.4876970348587557

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
delaunay3D1Display.OSPRayScaleFunction.Points = [-2.777055759186478e-23, 0.0, 0.5, 0.0, 1.0000001192092896, 1.0, 0.5, 0.0]

# hide data in view
Hide(tableToPoints1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Resample With Dataset'
resampleWithDataset1 = ResampleWithDataset(Input=delaunay3D1,
    Source=radvtk)

# Properties modified on resampleWithDataset1
resampleWithDataset1.Tolerance = 2.22044604925031e-16

# show data in view
resampleWithDataset1Display = Show(resampleWithDataset1, renderView1)
# trace defaults for the display properties.
resampleWithDataset1Display.Representation = 'Surface'
resampleWithDataset1Display.ColorArrayName = [None, '']
resampleWithDataset1Display.OSPRayScaleArray = 'Bmag'
resampleWithDataset1Display.OSPRayScaleFunction = 'PiecewiseFunction'
resampleWithDataset1Display.SelectOrientationVectors = 'Bmag'
resampleWithDataset1Display.ScaleFactor = 1.4400000000000002
resampleWithDataset1Display.SelectScaleArray = 'Bmag'
resampleWithDataset1Display.GlyphType = 'Arrow'
resampleWithDataset1Display.GlyphTableIndexArray = 'Bmag'
resampleWithDataset1Display.DataAxesGrid = 'GridAxesRepresentation'
resampleWithDataset1Display.PolarAxes = 'PolarAxesRepresentation'
resampleWithDataset1Display.ScalarOpacityUnitDistance = 0.32523651279827276

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
resampleWithDataset1Display.OSPRayScaleFunction.Points = [-2.777055759186478e-23, 0.0, 0.5, 0.0, 1.0000001192092896, 1.0, 0.5, 0.0]

# hide data in view
Hide(delaunay3D1, renderView1)

# hide data in view
Hide(radvtk, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(resampleWithDataset1Display, ('POINTS', 'Bmag'))

# rescale color and/or opacity maps used to include current data range
resampleWithDataset1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
resampleWithDataset1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Bmag'
bmagLUT = GetColorTransferFunction('Bmag')

# set active source
SetActiveSource(b_totcsv)

# set active source
SetActiveSource(radvtk)

# set active source
SetActiveSource(coilsvtk)

# change solid color
coilsvtkDisplay.DiffuseColor = [0.6862745098039216, 0.5568627450980392, 0.396078431372549]

# split cell
layout1.SplitHorizontal(0, 0.5)

# set active view
SetActiveView(None)

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [1128, 1037]
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.StereoType = 0
renderView2.Background = [0.32, 0.34, 0.43]

# place view in the layout
layout1.AssignView(2, renderView2)

# set active view
SetActiveView(renderView1)

# set active view
SetActiveView(renderView2)

# set active source
SetActiveSource(radvtk)

# show data in view
radvtkDisplay_1 = Show(radvtk, renderView2)
# trace defaults for the display properties.
radvtkDisplay_1.Representation = 'Surface'
radvtkDisplay_1.ColorArrayName = [None, '']
radvtkDisplay_1.OSPRayScaleFunction = 'PiecewiseFunction'
radvtkDisplay_1.SelectOrientationVectors = 'None'
radvtkDisplay_1.ScaleFactor = 1.4400000000000002
radvtkDisplay_1.SelectScaleArray = 'None'
radvtkDisplay_1.GlyphType = 'Arrow'
radvtkDisplay_1.GlyphTableIndexArray = 'None'
radvtkDisplay_1.DataAxesGrid = 'GridAxesRepresentation'
radvtkDisplay_1.PolarAxes = 'PolarAxesRepresentation'
radvtkDisplay_1.ScalarOpacityUnitDistance = 0.32523651279827276

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
radvtkDisplay_1.OSPRayScaleFunction.Points = [-2.777055759186478e-23, 0.0, 0.5, 0.0, 1.0000001192092896, 1.0, 0.5, 0.0]

# reset view to fit data
renderView2.ResetCamera()

# hide data in view
Hide(radvtk, renderView2)

# set active source
SetActiveSource(coilsvtk)

# show data in view
coilsvtkDisplay_1 = Show(coilsvtk, renderView2)
# trace defaults for the display properties.
coilsvtkDisplay_1.Representation = 'Surface'
coilsvtkDisplay_1.ColorArrayName = [None, '']
coilsvtkDisplay_1.OSPRayScaleFunction = 'PiecewiseFunction'
coilsvtkDisplay_1.SelectOrientationVectors = 'None'
coilsvtkDisplay_1.ScaleFactor = 1.968
coilsvtkDisplay_1.SelectScaleArray = 'None'
coilsvtkDisplay_1.GlyphType = 'Arrow'
coilsvtkDisplay_1.GlyphTableIndexArray = 'None'
coilsvtkDisplay_1.DataAxesGrid = 'GridAxesRepresentation'
coilsvtkDisplay_1.PolarAxes = 'PolarAxesRepresentation'
coilsvtkDisplay_1.ScalarOpacityUnitDistance = 0.6817433060777606

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
coilsvtkDisplay_1.OSPRayScaleFunction.Points = [-2.777055759186478e-23, 0.0, 0.5, 0.0, 1.0000001192092896, 1.0, 0.5, 0.0]

# reset view to fit data
renderView2.ResetCamera()

# set active source
SetActiveSource(resampleWithDataset1)

# show data in view
resampleWithDataset1Display_1 = Show(resampleWithDataset1, renderView2)
# trace defaults for the display properties.
resampleWithDataset1Display_1.Representation = 'Surface'
resampleWithDataset1Display_1.ColorArrayName = [None, '']
resampleWithDataset1Display_1.OSPRayScaleArray = 'Bmag'
resampleWithDataset1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
resampleWithDataset1Display_1.SelectOrientationVectors = 'Bmag'
resampleWithDataset1Display_1.ScaleFactor = 1.4400000000000002
resampleWithDataset1Display_1.SelectScaleArray = 'Bmag'
resampleWithDataset1Display_1.GlyphType = 'Arrow'
resampleWithDataset1Display_1.GlyphTableIndexArray = 'Bmag'
resampleWithDataset1Display_1.DataAxesGrid = 'GridAxesRepresentation'
resampleWithDataset1Display_1.PolarAxes = 'PolarAxesRepresentation'
resampleWithDataset1Display_1.ScalarOpacityUnitDistance = 0.32523651279827276

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
resampleWithDataset1Display_1.OSPRayScaleFunction.Points = [-2.777055759186478e-23, 0.0, 0.5, 0.0, 1.0000001192092896, 1.0, 0.5, 0.0]

# set scalar coloring
ColorBy(resampleWithDataset1Display_1, ('POINTS', 'Bmag'))

# rescale color and/or opacity maps used to include current data range
resampleWithDataset1Display_1.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
resampleWithDataset1Display_1.SetScalarBarVisibility(renderView2, True)

# create a new 'Clip'
clip1 = Clip(Input=resampleWithDataset1)
clip1.ClipType = 'Plane'
clip1.Scalars = ['POINTS', 'Bmag']
clip1.Value = 0.903721653272177

# get opacity transfer function/opacity map for 'Bmag'
bmagPWF = GetOpacityTransferFunction('Bmag')

# show data in view
clip1Display = Show(clip1, renderView2)
# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['POINTS', 'Bmag']
clip1Display.LookupTable = bmagLUT
clip1Display.OSPRayScaleArray = 'Bmag'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'Bmag'
clip1Display.ScaleFactor = 1.439317036387247
clip1Display.SelectScaleArray = 'Bmag'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'Bmag'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = bmagPWF
clip1Display.ScalarOpacityUnitDistance = 0.33472894808697323

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
clip1Display.OSPRayScaleFunction.Points = [-2.777055759186478e-23, 0.0, 0.5, 0.0, 1.0000001192092896, 1.0, 0.5, 0.0]

# hide data in view
Hide(resampleWithDataset1, renderView2)

# show color bar/color legend
clip1Display.SetScalarBarVisibility(renderView2, True)

# update the view to ensure updated data information
renderView2.Update()

# set active source
SetActiveSource(coilsvtk)

# change solid color
coilsvtkDisplay_1.DiffuseColor = [0.6862745098039216, 0.5568627450980392, 0.396078431372549]

# create a new 'Clip'
clip2 = Clip(Input=coilsvtk)
clip2.ClipType = 'Plane'
clip2.Scalars = [None, '']

# show data in view
clip2Display = Show(clip2, renderView2)
# trace defaults for the display properties.
clip2Display.Representation = 'Surface'
clip2Display.ColorArrayName = [None, '']
clip2Display.DiffuseColor = [0.6862745098039216, 0.5568627450980392, 0.396078431372549]
clip2Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip2Display.SelectOrientationVectors = 'None'
clip2Display.ScaleFactor = 1.968
clip2Display.SelectScaleArray = 'None'
clip2Display.GlyphType = 'Arrow'
clip2Display.GlyphTableIndexArray = 'None'
clip2Display.DataAxesGrid = 'GridAxesRepresentation'
clip2Display.PolarAxes = 'PolarAxesRepresentation'
clip2Display.ScalarOpacityUnitDistance = 0.7082737922494425

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
clip2Display.OSPRayScaleFunction.Points = [-2.777055759186478e-23, 0.0, 0.5, 0.0, 1.0000001192092896, 1.0, 0.5, 0.0]

# hide data in view
Hide(coilsvtk, renderView2)

# update the view to ensure updated data information
renderView2.Update()

# reset view to fit data
renderView2.ResetCamera()

# set active source
SetActiveSource(radvtk)

# current camera placement for renderView2
renderView2.CameraPosition = [-50.0693413950585, 0.0, 0.0]
renderView2.CameraFocalPoint = [1e-20, 0.0, 0.0]
renderView2.CameraViewUp = [0.0, 1.0, 2.220446049250313e-16]
renderView2.CameraParallelScale = 18.973124214448493

# current camera placement for renderView1
renderView1.CameraPosition = [-50.0693413950585, 0.0, 0.0]
renderView1.CameraFocalPoint = [1e-20, 0.0, 0.0]
renderView1.CameraViewUp = [0.0, 1.0, 2.220446049250313e-16]
renderView1.CameraParallelScale = 18.973124214448493


# current camera placement for renderView2
renderView2.CameraPosition = [-50.0693413950585, 0.0, 0.0]
renderView2.CameraFocalPoint = [1e-20, 0.0, 0.0]
renderView2.CameraViewUp = [0.0, 1.0, 2.220446049250313e-16]
renderView2.CameraParallelScale = 18.973124214448493

# save screenshot
SaveScreenshot('/home/ssharpe/FIN_MAG_CODE/MAG_FIELD_CALC/Tokamak_half.png', renderView2, ImageResolution=[1135, 1065])

# set active view
SetActiveView(renderView1)

# current camera placement for renderView1
renderView1.CameraPosition = [-50.0693413950585, 0.0, 0.0]
renderView1.CameraFocalPoint = [1e-20, 0.0, 0.0]
renderView1.CameraViewUp = [0.0, 1.0, 2.220446049250313e-16]
renderView1.CameraParallelScale = 18.973124214448493

# save screenshot
SaveScreenshot('/home/ssharpe/FIN_MAG_CODE/MAG_FIELD_CALC/Tokamak_full.png', renderView1, ImageResolution=[1135, 1065])

#### saving camera placements for all active views

# current camera placement for renderView2
renderView2.CameraPosition = [-50.0693413950585, 0.0, 0.0]
renderView2.CameraFocalPoint = [1e-20, 0.0, 0.0]
renderView2.CameraViewUp = [0.0, 1.0, 2.220446049250313e-16]
renderView2.CameraParallelScale = 18.973124214448493

# current camera placement for renderView1
renderView1.CameraPosition = [-50.0693413950585, 0.0, 0.0]
renderView1.CameraFocalPoint = [1e-20, 0.0, 0.0]
renderView1.CameraViewUp = [0.0, 1.0, 2.220446049250313e-16]
renderView1.CameraParallelScale = 18.973124214448493

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
