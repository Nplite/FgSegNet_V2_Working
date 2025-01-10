import os
import shutil
import subprocess
import sys
import os
import numpy as np
from PIL import Image
from Stats import Stats
call = subprocess.call

def main():    
    # datasetPath = sys.argv[1]
    # binaryRootPath = sys.argv[2]

    datasetPath = '/home/ai/Namdeo/FGBG/datasets/CDnet2014_dataset'
    binaryRootPath = 'FgSegNet_v2/results25_th0.7'
    
    if not isValidRootFolder(datasetPath):
        print('The folder ' + datasetPath + ' is not a valid root folder.');
        return
    
    processFolder(datasetPath, binaryRootPath)

def processFolder(datasetPath, binaryRootPath):
    """Call your executable for all sequences in all categories."""
    stats = Stats(datasetPath)  #STATS
    f = open(datasetPath + '\\' +  'fscore.txt', 'w')
    for category in getDirectories(datasetPath):
        stats.addCategories(category)  #STATS
        
        categoryPath = os.path.join(datasetPath, category)
        for video in getDirectories(categoryPath):
            videoPath = os.path.join(categoryPath, video)
            binaryPath = os.path.join(binaryRootPath, category, video)
            if isValidVideoFolder(videoPath):
                confusionMatrix = compareWithGroungtruth(videoPath, binaryPath)
                stats.update(category, video, confusionMatrix)
                alpha = 0.000001
                fscore = (2.0 * confusionMatrix[0])/ (((2.0 * confusionMatrix[0]) + confusionMatrix[1] + confusionMatrix[2]) + alpha)
                f.write(video + ' : ' + str(fscore) + '\n')
            else:
                print ('Invalid folder : ' + videoPath)
        stats.writeCategoryResult(category)
    stats.writeOverallResults()
    f.close()


def compare_images(gt_path, result_path):
    """
    Compare ground truth and result images to generate confusion matrix
    Returns: [TP, FP, FN, TN, total_pixels]
    """
    try:
        # Read ground truth image
        gt_img = Image.open(gt_path).convert('L')
        gt_arr = np.array(gt_img) > 0  # Convert to binary

        # Read result image
        result_img = Image.open(result_path).convert('L')
        result_arr = np.array(result_img) > 0  # Convert to binary

        # Ensure same size
        if gt_arr.shape != result_arr.shape:
            raise ValueError(f"Image sizes don't match: {gt_arr.shape} vs {result_arr.shape}")

        # Calculate confusion matrix
        TP = np.sum((gt_arr == True) & (result_arr == True))
        FP = np.sum((gt_arr == False) & (result_arr == True))
        FN = np.sum((gt_arr == True) & (result_arr == False))
        TN = np.sum((gt_arr == False) & (result_arr == False))
        total = gt_arr.size

        return [TP, FP, FN, TN, total]

    except Exception as e:
        print(f"Error comparing images: {str(e)}")
        return [0, 0, 0, 0, 0]
def compareWithGroungtruth(videoPath, binaryPath):
    """Compare your binaries with the groundtruth and return the confusion matrix"""
    statFilePath = os.path.join(videoPath, 'stats.txt')
    deleteIfExists(statFilePath)

    groundtruth_path = os.path.join(videoPath, 'groundtruth')
    binary_files = sorted(os.listdir(binaryPath))
    gt_files = sorted(os.listdir(groundtruth_path))
    
    total_cm = [0, 0, 0, 0, 0]
    
    # Compare each pair of images
    for gt_file, binary_file in zip(gt_files, binary_files):
        if gt_file.endswith(('.png', '.jpg', '.bmp')):
            gt_path = os.path.join(groundtruth_path, gt_file)
            binary_path = os.path.join(binaryPath, binary_file)
            
            if os.path.exists(binary_path):
                cm = compare_images(gt_path, binary_path)
                total_cm = [x + y for x, y in zip(total_cm, cm)]
    
    # Write stats file
    with open(statFilePath, 'w') as f:
        f.write(f"cm: {' '.join(map(str, total_cm))}\n")
    
    return total_cm

def readCMFile(filePath):
    """Read the file, so we can compute stats for video, category and overall."""
    if not os.path.exists(filePath):
        print("The file " + filePath + " doesn't exist.\nIt means there was an error calling the comparator.")
        raise Exception('error')
    
    with open(filePath) as f:
        for line in f.readlines():
            if line.startswith('cm:'):
                numbers = line.split()[1:]
                return [int(nb) for nb in numbers[:5]]





def isValidRootFolder(path):
    """A valid root folder must have the six categories"""
    #categories = set(['Board_a', 'Candela_m1.10_a', 'CAVIAR1_a', 'CAVIAR2_a', 'CaVignal_a','Foliage_a', 'HallAndMonitor_a', 'HighwayI_a', 'HighwayII_a', 'HumanBody2_a', 'IBMtest2_a', 'PeopleAndFoliage_a', 'Toscana_a','Snellen_a'])
    categories = set([ 'baseline',  
                      'cameraJitter', 
                      'badWeather', 
                      ])
    
    #categories = set(['baseline_2', 'baseline_4', 'thermal_2', 'thermal_4', 'cameraJitter_2', 'cameraJitter_4'])

    folders = set(getDirectories(path))
    return len(categories.intersection(folders)) == 3

def isValidVideoFolder(path):
    """A valid video folder must have \\groundtruth, \\input, ROI.bmp, temporalROI.txt"""
    return os.path.exists(os.path.join(path, 'groundtruth'))  and os.path.exists(os.path.join(path, 'ROI.bmp')) and os.path.exists(os.path.join(path, 'temporalROI.txt'))
    # and os.path.exists(os.path.join(path, 'input'))
def getDirectories(path):
    """Return a list of directories name on the specifed path"""
    return [file for file in os.listdir(path)
            if os.path.isdir(os.path.join(path, file))]

def deleteIfExists(path):
    if os.path.exists(path):
        os.remove(path)


if __name__ == "__main__":
    main()
