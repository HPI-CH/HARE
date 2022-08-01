"""
test with new config

"""

from fileinput import filename
import os
import random

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from models.RainbowModel import RainbowModel
import pandas as pd
import utils.settings as settings
from evaluation.metrics import accuracy
from evaluation.conf_matrix import create_conf_matrix
from loader.Preprocessor import Preprocessor
from models.JensModel import JensModel
from models.RainbowModel import RainbowModel
from models.ResNetModel import ResNetModel
from utils.filter_activities import filter_activities
from utils.folder_operations import new_saved_experiment_folder
from utils.DataConfig import Sonar22CategoriesConfig, OpportunityConfig
from tensorflow.keras.layers import (Dense)
from utils.Recording import Recording
import matplotlib.pyplot as plt

# Init
# TODO: refactor, find out why confusion matrix sometimes shows different results than accuracy
# TODO: - make train and test datasets evenly distributed
#       - make 
""" 
Number of recordings per person
{'connie.csv': 6, 'alex.csv': 38, 'trapp.csv': 9, 'anja.csv': 13, 'aileen.csv': 52, 'florian.csv': 16, 'brueggemann.csv': 36, 'oli.csv': 20, 'rauche.csv': 9, 'b2.csv': 6, 'yvan.csv': 8, 'christine.csv': 7, 'mathias.csv': 2, 'kathi.csv': 17}
"""

data_config = Sonar22CategoriesConfig(dataset_path='/dhc/groups/bp2021ba1/data/filtered_dataset_without_null')#OpportunityConfig(dataset_path='/dhc/groups/bp2021ba1/data/opportunity-dataset')
settings.init(data_config)
random.seed(1678978086101)

k_fold_splits = 3
numEpochsBeforeTL = 10
numEpochsForTL = 3
minimumRecordingsPerLeftOutPerson = 5
# Load dataset
recordings = settings.DATA_CONFIG.load_dataset()#limit=75

# Preprocess
recordings = Preprocessor().our_preprocess(recordings)

def split_list_by_people(recordings: "list[Recording]", peopleForListA: "list[str]") -> tuple[np.ndarray, np.ndarray]:
    """ Splits the recordings into a tuple of a sublist of recordings of people in peopleForListA and the recordings of other people"""
    return np.array(list(filter(lambda recording: recording.subject in peopleForListA,recordings))), np.array(list(filter(lambda recording: recording.subject not in peopleForListA, recordings)))
window_size = 100
n_features = recordings[0].sensor_frame.shape[1]
print(n_features)
n_outputs = settings.DATA_CONFIG.n_activities()

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(
    "transferLearningTobi"
)

def count_activities_per_person(recordings: "list[Recording]"): 
    values = pd.DataFrame({recordings[0].subject: recordings[0].activities.value_counts()}) 
    for rec in recordings[1:]: 
        values = values.add(pd.DataFrame({rec.subject: rec.activities.value_counts()}), fill_value=0) 
 
    return values

def plot_activities_per_person(recordings: "list[Recording]", fileName: str, title: str = ""):
    values = count_activities_per_person(recordings) 
    
    values.plot.bar(figsize=(22,16)) 
    plt.title(title) 
    plt.xlabel("x") 
    plt.ylabel("y") 
    plt.savefig(os.path.join(experiment_folder_path,fileName))


def save_pie_chart_from_dict(labelsAndFrequencyDict: dict, fileName: str, title: str = None, subtitle: str = None)-> None:
    plt.cla()
    plt.clf()
    print(labelsAndFrequencyDict)
    print(settings.DATA_CONFIG.raw_label_to_activity_idx_map)
    data = [labelsAndFrequencyDict[label] for label in labelsAndFrequencyDict.keys()]
    labels = [f"{settings.DATA_CONFIG.raw_label_to_activity_idx_map[label]} {label} {int(labelsAndFrequencyDict[label]/60)} secs" for label in labelsAndFrequencyDict.keys()]
    plt.pie(data, labels=labels)
    if title:
        plt.suptitle(title, y=1.05, fontsize=18)
    if subtitle:
        plt.title(subtitle, fontsize=10)

    plt.savefig(os.path.join(experiment_folder_path, fileName))

def getActivityCountsFromRecordings(recordings: "list[Recording]")-> dict:
    resultDict = {}
    for recording in recordings:
        counts = recording.activities.value_counts()
        for activity_id, count in counts.items():
            if activity_id in resultDict:
                resultDict[activity_id] += count
            else:
                resultDict[activity_id] = count 
    for activity in settings.DATA_CONFIG.raw_label_to_activity_idx_map:
        if not activity in resultDict:
            resultDict[activity] = 0
    return resultDict

def getActivityCounts(yTrue):
    unique, counts = np.unique(np.argmax(yTrue, axis=1), return_counts=True)
    countsDict = dict(zip([settings.DATA_CONFIG.activity_idx_to_activity_name_map[item] for item in unique], counts))
    return countsDict

def save_activity_distribution_pie_chart(yTrue, fileName: str, title: str=None, subtitle: str = None) -> None:
    unique, counts = np.unique(np.argmax(yTrue, axis=1), return_counts=True)
    countsDict = dict(zip([settings.DATA_CONFIG.activity_idx_to_activity_name_map[item] for item in unique], counts))
    subtitleSuffix = f"Mean activity duration difference from mean activity duration: {int(getMeanCountDifferenceFromMeanActivityCount(yTrue)/60)}s"
    if subtitle != None:
        subtitle += "\n"+subtitleSuffix
    else:
        subtitle = subtitleSuffix
    save_pie_chart_from_dict(countsDict, fileName, title, subtitle)

def count_recordings_of_people(recordings: "list[Recording]")-> dict:
    peopleCount = {}
    for recording in recordings:
        if recording.subject in peopleCount:
            peopleCount[recording.subject] += 1
        else:
            peopleCount[recording.subject] = 1
    return peopleCount

def getMeanCountDifferenceFromMeanActivityCount(yTrue)-> float:
    activityCounts = getActivityCounts(yTrue)

    activitySum = 0
    for activity in activityCounts:
       activitySum += activityCounts[activity]
    meanActivityCount = activitySum / len(activityCounts)

    diffSum = 0
    for activity in activityCounts:
       diffSum += abs(activityCounts[activity]-meanActivityCount)
    
    return diffSum / len(activityCounts)

def get_people_in_recordings(recordings: "list[Recording]") -> list[str]:
    people = set()
    for recording in recordings:
        people.add(recording.subject)
    return list(people)


def evaluate(model: "RainbowModel",X_test: np.ndarray, y_test_true: np.ndarray, confusionMatrixFileName=None, confusionMatrixTitle="") -> tuple[float, float,float, np.ndarray]:
    y_test_pred = model.predict(X_test)
    acc = accuracy(y_test_pred, y_test_true)
    if confusionMatrixFileName:
        create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true, file_name = confusionMatrixFileName, title=confusionMatrixTitle+", acc:"+str(int(acc*10000)/100)+"%") 
    f1_macro = f1_score(np.argmax(y_test_true, axis=1), np.argmax(y_test_pred, axis=1), average="macro")   
    f1_weighted = f1_score(np.argmax(y_test_true, axis=1), np.argmax(y_test_pred, axis=1), average="weighted")    
    return acc, f1_macro, f1_weighted, y_test_true

def instanciateModel():
    return ResNetModel(
        n_epochs=numEpochsBeforeTL,
        window_size=100,
        n_features=n_features,
        n_outputs=n_outputs,
        batch_size=64,
    )
def freezeDenseLayers(model: RainbowModel):
    # Set non dense layers to not trainable (freezing them)
    for layer in model.model.layers:
        layer.trainable = type(layer) == Dense

def getAveragesOfAttributesInDicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    result = {}
    for d in dicts:
        for key in d:
            if key in result:
                result[key] += d[key] / len(dicts)
            else:
                result[key] = d[key] / len(dicts)
    return result

people = get_people_in_recordings(recordings)
numRecordingsOfPeopleDict = count_recordings_of_people(recordings)

peopleToLeaveOutPerExpirement = list(filter(lambda person: numRecordingsOfPeopleDict[person]>minimumRecordingsPerLeftOutPerson,people)) #["anja.csv", "florian.csv", "oli.csv", "rauche.csv"]#, "oli.csv", "rauche.csv"



result = [["fold id \ Left out person"]+[["**FOLD "+str(round(i/3))+"**", "without TL", "with TL"][i%3] for i in range(k_fold_splits*3)]+["Average without TL"]+["Average with TL"]]

TLsuccessForUniformDistributionScore = []

for personIndex, personToLeaveOut in enumerate(peopleToLeaveOutPerExpirement):
    print(f"==============================================================================\nLeaving person {personToLeaveOut} out {personIndex}/{len(peopleToLeaveOutPerExpirement)}\n==============================================================================\n")
    personId = people.index(personToLeaveOut)
    model = instanciateModel()
    recordingsOfLeftOutPerson, recordingsTrain = split_list_by_people(recordings, [personToLeaveOut])
    model.n_epochs = numEpochsBeforeTL
    _, yTrainTrue = model.windowize_convert_fit(recordingsTrain)
    activityDistributionFileName = f"subject{personId}_trainActivityDistribution.png"
    save_activity_distribution_pie_chart(yTrainTrue,  activityDistributionFileName)

    resultCol = [f"Subject {personId}<br />Train activity distribution <br />![Base model train activity distribution]({activityDistributionFileName})"]
    resultWithoutTLVals = []
    resultWithTLVals = []
    model.n_epochs = numEpochsForTL
    model.model.save_weights("ckpt")

    xLeftOutPerson, yLeftOutPerson = model.windowize_convert(recordingsOfLeftOutPerson, should_shuffle=False)
    # Evaluate on left out person
    k_fold = StratifiedKFold(n_splits=k_fold_splits, random_state=None)
    for (index, (train_indices, test_indices)) in enumerate(k_fold.split(np.zeros(np.shape(yLeftOutPerson)[0]), np.argmax(yLeftOutPerson, axis=1))):
        # Restore start model state for all folds of this left out person 
        model.model.load_weights("ckpt")

        # Grab data for this fold
        xTrainLeftOutPerson, yTrainLeftOutPerson =  xLeftOutPerson[train_indices], yLeftOutPerson[train_indices]
        xTestLeftOutPerson, yTestLeftOutPerson = xLeftOutPerson[test_indices], yLeftOutPerson[test_indices]

        # Evaluate without transfer learning
        confMatrixWithoutTLFileName = f"subject{personId}_kfold{index}_withoutTL_conf_matrix"
        accuracyWithoutTransferLearning, f1ScoreMacroWithoutTransferLearning,f1ScoreWeightedWithoutTransferLearning, yTestTrue = evaluate(model, xTestLeftOutPerson, yTestLeftOutPerson, confMatrixWithoutTLFileName, f"w/o TL, validated on subject {personId}, fold {index}")
        print(f"Accuracy on test data of left out person {accuracyWithoutTransferLearning}")

        # Store test distribution for this fold
        activityDistributionTestFileName = f"subject{personId}_kfold{index}_testActivityDistribution.png"
        save_activity_distribution_pie_chart(yTestTrue,  activityDistributionTestFileName, title="Test distribution", subtitle=f"Activities of subject {personId} used to test model w/o TL in fold {index}")
        
        # Do transfer learning
        freezeDenseLayers(model)
        model.fit(xTrainLeftOutPerson, yTrainLeftOutPerson)

        # Store TL train distribution
        tlActivityDistributionFileName =  f"subject{personId}_kfold{index}_TL_trainActivityDistribution.png"
        save_activity_distribution_pie_chart(yTrainLeftOutPerson,  tlActivityDistributionFileName, title="TL Train distribution", subtitle=f"Activities of subject {personId} used for transfer learning in fold {index}")

        # Store TL train evaluation
        confMatrixWithTLFileName = f"subject{personId}_kfold{index}_withTL_conf_matrix"
        accuracyWithTransferLearning, f1ScoreMacroWithTransferLearning, f1ScoreWeightedWithTransferLearning, _ = evaluate(model, xTestLeftOutPerson, yTestLeftOutPerson, confMatrixWithTLFileName, f"With TL, validated on subject {personId}, fold {index}")
        print(f"Accuracy on test data of left out person {accuracyWithTransferLearning}")

        TLsuccessForUniformDistributionScore.append((getMeanCountDifferenceFromMeanActivityCount(yTrainLeftOutPerson),accuracyWithTransferLearning-accuracyWithoutTransferLearning))
        # Append report
        resultCol.append("Test Activity Distribution "+f"<br />![Test activity distribution]({activityDistributionTestFileName})")
        resultCol.append("Accuracy: "+str(accuracyWithoutTransferLearning)+f"<br />F1-Score Macro: {f1ScoreMacroWithoutTransferLearning}<br />F1-Score Weighted: {f1ScoreWeightedWithoutTransferLearning}<br />![confusion matrix]({confMatrixWithoutTLFileName}.png)")
        resultCol.append("Accuracy: "+str(accuracyWithTransferLearning)+f"<br />F1-Score Macro: {f1ScoreMacroWithTransferLearning}<br />F1-Score Weighted: {f1ScoreWeightedWithTransferLearning}<br />TL-Train activity distribution <br />![TL-Train activity distribution]({tlActivityDistributionFileName})"+f"<br />![confusion matrix]({confMatrixWithTLFileName}.png)")
        resultWithoutTLVals.append({"Accuracy": accuracyWithoutTransferLearning, "F1-score Macro":f1ScoreMacroWithoutTransferLearning,"F1-score Weighted": f1ScoreWeightedWithoutTransferLearning})
        resultWithTLVals.append({"Accuracy": accuracyWithTransferLearning, "F1-score Macro":f1ScoreMacroWithTransferLearning,"F1-score Weighted": f1ScoreWeightedWithTransferLearning})

    metricAvgWithoutTL = getAveragesOfAttributesInDicts(resultWithoutTLVals)
    metricAvgWithTL = getAveragesOfAttributesInDicts(resultWithTLVals)
    resultCol.append("<br />".join([f"{metric}: {metricAvgWithoutTL[metric]}" for metric in metricAvgWithoutTL]))
    resultCol.append("<br />".join([f"{metric}: {metricAvgWithTL[metric]}" for metric in metricAvgWithTL]))
    result = result + [resultCol]

print("result",result)


resultT = np.array(result).T
print("resultT",resultT)
# save a simple test report to the experiment folder
wholeDataSetActivityDistributionFileName = "wholeDatasetActivityDistribution.png"
_, yAll = instanciateModel().windowize_convert(recordings)
save_activity_distribution_pie_chart(yAll,  wholeDataSetActivityDistributionFileName)
result_md = f"# Experiment"
result_md += f"\nDoing model training with {numEpochsBeforeTL} epochs and transfer learning {numEpochsForTL} with epochs"
result_md += f"\nUsing stratified kfolds for left out person"
result_md += f"\n## Model"
result_md += f"\n```\n"
def append_line(line):
    global result_md
    result_md += line + "\n"
instanciateModel().model.summary(print_fn=append_line)
result_md += f"```"
result_md += f"\n## Dataset"
result_md += f"\n### Whole dataset distribution\n![activityDistribution]({wholeDataSetActivityDistributionFileName})"
result_md += f"\nUsing dataset `{settings.DATA_CONFIG.dataset_path}`"
activitiesPerPersonFilename = "actvitiesPerPerson.png"
plot_activities_per_person(recordings, activitiesPerPersonFilename, "Activities per person")
result_md += f"\n### Activities per subject\n![activityDistribution]({activitiesPerPersonFilename})"
result_md += f"\nLeaving people out with at least  `{minimumRecordingsPerLeftOutPerson}`"
result_md += "\n## Experiments\n"
for index, row in enumerate(resultT):
    for item in row:
        result_md += "|" + str(item)
    result_md += "|\n"
    if index == 0:
        for col in range(len(row)):
            result_md += "| -----------"
        result_md += "|\n"
result_md += "\n## Summary"
plt.clf()
scatterData = np.array(TLsuccessForUniformDistributionScore)
plt.scatter(scatterData[:,0], scatterData[:,1])
plt.title("TL effects according to uniformity of training data distribution") 
plt.xlabel("Mean activity duration difference from mean activity duration") 
plt.ylabel("Accuracy improvement through Transfer learning ") 
scatterFileName = "uniformityOfTrainingDataToTLimprovement.png"
plt.savefig(os.path.join(experiment_folder_path,scatterFileName))
result_md += f"\n# ![TL effects according to uniformity of training data distribution]({scatterFileName})"

with open(os.path.join(experiment_folder_path, "results.md"), "w+") as f:
    f.writelines(result_md)




