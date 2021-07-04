"""
Module name: LabelFusion
Author:      Robert Finnegan
Date:        December 2018
Description:
---------------------------------
- UWV, GWV, LWV, BWV, STAPLE
- Label processing
---------------------------------
"""

from __future__ import print_function
import os, sys

import SimpleITK as sitk
import numpy as np
from functools import reduce
import re

def computeWeightMap(targetImage, movingImage, voteType='local', voteParams={'sigma':2.0, 'epsilon':1E-5, 'factor':1e12, 'gain':6, 'blockSize':5}):
	"""
	Computes the weight map
	"""
	targetImage = sitk.Cast(targetImage, sitk.sitkFloat32)
	movingImage = sitk.Cast(movingImage, sitk.sitkFloat32)

	squareDifferenceImage = sitk.SquaredDifference(targetImage, movingImage)
	squareDifferenceImage = sitk.Cast(squareDifferenceImage, sitk.sitkFloat32)

	if voteType.lower()=='majority':
		weightMap = targetImage * 0.0 + 1.0

	elif voteType.lower()=='global':
		factor = voteParams['factor']
		sumSquaredDifference  = sitk.GetArrayFromImage(squareDifferenceImage).sum(dtype=np.float)
		globalWeight = factor / sumSquaredDifference

		weightMap = targetImage * 0.0 + globalWeight

	elif voteType.lower()=='local':
		sigma = voteParams['sigma']
		epsilon = voteParams['epsilon']

		rawMap = sitk.DiscreteGaussian(squareDifferenceImage, sigma*sigma)
		weightMap = sitk.Pow(rawMap + epsilon , -1.0)

	elif voteType.lower()=='block':
		factor = voteParams['factor']
		gain = voteParams['gain']
		blockSize = voteParams['blockSize']
		if type(blockSize)==int:
			blockSize = (blockSize,)*targetImage.GetDimension()

		#rawMap = sitk.Mean(squareDifferenceImage, blockSize)
		rawMap  = sitk.BoxMean(squareDifferenceImage, blockSize)
		weightMap = factor * sitk.Pow(rawMap, -1.0) ** abs(gain/2.0)
		# Note: we divide gain by 2 to account to using the squared difference image
		#       which raises the power by 2 already.

	else:
		raise ValueError('Weighting scheme not valid.')

	return sitk.Cast(weightMap, sitk.sitkFloat32)

def combineLabelsSTAPLE(labelListDict, threshold=1e-4):
	"""
	Combine labels using STAPLE
	"""

	combinedLabelDict = {}

	caseIdList = list(labelListDict.keys())
	structureNameList = [list(i.keys()) for i in labelListDict.values()]
	structureNameList = np.unique([item for sublist in structureNameList for item in sublist] )

	for structureName in structureNameList:
		# Ensure all labels are binarised
		binaryLabels = [sitk.BinaryThreshold(labelListDict[i][structureName], lowerThreshold=0.5) for i in labelListDict]

		# Perform STAPLE
		combinedLabel = sitk.STAPLE(binaryLabels)

		# Normalise
		combinedLabel = sitk.RescaleIntensity(combinedLabel, 0, 1)

		# Threshold - grants vastly improved compression performance
		if threshold:
			combinedLabel = sitk.Threshold(combinedLabel, lower=threshold, upper=1, outsideValue=0.0)

		combinedLabelDict[structureName] = combinedLabel

	return combinedLabelDict


def combineLabels(weightMapDict, labelListDict, threshold=1e-4):
	"""
	Combine labels using weight maps
	"""

	combinedLabelDict = {}

	caseIdList = list(weightMapDict.keys())
	structureNameList = [list(i.keys()) for i in labelListDict.values()]
	structureNameList = np.unique([item for sublist in structureNameList for item in sublist] )

	for structureName in structureNameList:
		# Find the cases which have the strucure (in case some cases do not)
		validCaseIdList = [i for (i,j) in list(labelListDict.items()) if structureName in j.keys()]

		# Get valid weight images
		weightImageList = [weightMapDict[caseId] for caseId in validCaseIdList]

		# Sum the weight images
		weightSumImage = reduce(lambda x,y:x+y, weightImageList)
		weightSumImage = sitk.Mask(weightSumImage, weightSumImage==0, maskingValue=1, outsideValue=1)

		# Combine weight map with each label
		weightedLabels = [weightMapDict[caseId]*sitk.Cast(labelListDict[caseId][structureName], sitk.sitkFloat32) for caseId in validCaseIdList]

		# Combine all the weighted labels
		combinedLabel = reduce(lambda x,y:x+y, weightedLabels) / weightSumImage

		# Normalise
		combinedLabel = sitk.RescaleIntensity(combinedLabel, 0, 1)

		# Threshold - grants vastly improved compression performance
		if threshold:
			combinedLabel = sitk.Threshold(combinedLabel, lower=threshold, upper=1, outsideValue=0.0)

		combinedLabelDict[structureName] = combinedLabel

	return combinedLabelDict

def processProbabilityImage(probabilityImage, threshold=0.1):

	# Get the starting binary image
	binaryImage = sitk.BinaryThreshold(probabilityImage, lowerThreshold=threshold)

	# Fill holes
	binaryImage = sitk.BinaryFillhole(binaryImage)

	# Apply the connected component filter
	labelledImage = sitk.ConnectedComponent(binaryImage)

	# Measure the size of each connected component
	labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
	labelShapeFilter.Execute(labelledImage)
	labelIndices = labelShapeFilter.GetLabels()
	voxelCounts  = [labelShapeFilter.GetNumberOfPixels(i) for i in labelIndices]
	if voxelCounts==[]:
		return binaryImage

	# Select the largest region
	largestComponentLabel = labelIndices[np.argmax(voxelCounts)]
	largestComponentImage = (labelledImage==largestComponentLabel)

	return largestComponentImage

if __name__ == '__main__':

	if len(sys.argv)!=7:

		print("Combine labels from registered images.")
		print("Arguments:")
		print("   1: Target image")
		print("      Input/Case_01_CT.nii.gz")
		print("   2: Directory with (registered) moving images")
		print("      Input/RegisteredImages")
		print("   3: Base name for propagated structure images")
		print("      Input/RegisteredImages/Structures/Case_{0}_to_Case_{1}_{2}_DEMONS.nii.gz")
		print("   4: Vote type")
		print("      majority (or global, local, block, staple)")
		print("   5: Text file with structure names")
		print("      Inputs/structures.txt")
		print("   6: Output base name")
		print("      Output/Case_{0}_{1}_LOCALVOTE_{2}.nii.gz")
		sys.exit()
	else:
		targetImageName  = sys.argv[1]
		movingImageDir   = sys.argv[2]
		movingStructBase = sys.argv[3]
		voteType         = sys.argv[4]
		structureFile    = sys.argv[5]
		outputBase       = sys.argv[6]

		for i, x in enumerate(sys.argv):
			print('{0}: {1}'.format(i,x))

		targetImage = sitk.ReadImage(targetImageName)
		targetId    = re.split('\.|_', targetImageName)[-3]

		movingImageNameList = [i for i in os.listdir(movingImageDir) if ( ('.nii.gz' in i) and ('FIELD' not in i) )]

		with open(structureFile,'r') as f:
			structureNameList = [i.strip() for i in f.readlines()]

		weightMapDict = {}
		labelListDict = {}
		#for loop
		for movingImageName in movingImageNameList:
			movingId    = movingImageName.split('_')[1]
			print(movingImageName)
			if voteType.lower()!='staple':#如果不是staple的
				movingImage = sitk.ReadImage(movingImageDir+'/'+movingImageName)
				# moving image
				weightMap   = computeWeightMap(targetImage, movingImage, voteType=voteType)

			labelListDict[movingId] = {}
			#all segmentation file
			for structureName in structureNameList:
				structImageName = movingStructBase.format(movingId, targetId, structureName)
				labelListDict[movingId][structureName] = sitk.ReadImage(structImageName)

			if voteType.lower()!='staple':
				weightMapDict[movingId] = weightMap

		#staple
		if voteType.lower()!='staple':
			combinedLabelDict = combineLabels(weightMapDict, labelListDict)

		else:
			combinedLabelDict = combineLabelsSTAPLE(labelListDict)

		for structureName in structureNameList:
			probabilityImage = combinedLabelDict[structureName]
			binaryImage      = processProbabilityImage(probabilityImage)

			outputName = outputBase.format(structureName, 'PROBABILITY')
			sitk.WriteImage(probabilityImage, outputName)

			outputName = outputBase.format(structureName, 'BINARY')
			sitk.WriteImage(binaryImage, outputName)