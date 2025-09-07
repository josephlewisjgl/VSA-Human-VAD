# VSA-Human-VAD

This code supports an MSc dissertation thesis exploring the application of Vector Symbolic Architectures to human video anomaly detection.

## Dependencies

It is expected that you will work from a root directory i.e. `/VSA-VAD/`

This root directory is where the Avenue Dataset (described below) or ShanghaiTech dataset (below) will be placed.

In this directory clone the repo:

```
git clone https://github.com/josephlewisjgl/VSA-Human-VAD.git
```

Then each Python notebook should be run in the relative directory i.e. with the working directory set to `VSA-VAD/VSA-Human-VAD`.

If you do not do this then please update the file paths in the notebook to your current Avenue Dataset or ShanghaiTech location.

Python dependencies are installed as part of the notebooks used to run code, as opposed to a requirements file, so please ensure you are using a new environment or are content to use your current environment.

### Datasets

To run the pose estimation pipeline you will need the raw datasets for the CUHK-Avenue and ShanghaiTech data.

The CUHK-Avenue data is made available by the author via Kaggle: https://www.kaggle.com/datasets/joelewis/hr-avenueframedata

The ShanghaiTech data can be retrieved from the original authors: https://svip-lab.github.io/dataset/campus_dataset.html

Both should be placed in the `VSA-VAD` directories or have filepaths updated. Note: The ShanghaiTech dataset is very large so may favour placement on a cloud drive/storage space.

As an alternative if you would like to work with pre-extracted poses it is sufficient to just use the datasets cloned in this repo in the `/data/` folder. For validation you may favour extracting labels/frames yourself but the ones provided are used in the paper.

For the ShanghaiTech datasets the pre-extracted poses are also stored on Kaggle: https://www.kaggle.com/datasets/joelewis/shanghaitechtrainandtestposes

### Baselines (optional)

The baseline MoCoDAD reproduction notebook is a part of this repository along with the config used, however the original dataset skeleton poses and pre-trained model/supporting code is not. To reproduce the benchmark users are pointed: [MoCoDAD](https://github.com/aleflabo/MoCoDAD)

The point at which to use the `.yaml` file in `/mocodad/` will be clear from the repo README.

Alternatively you can run the results analysis with just my reproduction of the baseline which is already saved in `/data/mocodad_anomaly_scores.pt`

## Running the code

For simplicity the code is run in notebooks. 

To reproduce the HR-Avenue results of the report, the notebooks should be run in order from 01 to 08.

If you are not interested in collecting poses you can:
* Run your own pose estimator on the data to get it into the same structure as that in `/data/<train/test>_tracked_poses.csv`
* Use the pre-extracted poses and start from 02_AnomalyDetection.ipynb

01_CollectPoses.ipynb: 
* Detect poses and build structured dataset for training frames with the edge and post-hoc systems
* Detect poses and build structured dataset for test frames with just the post-hoc system (edge system poses are detected at runtime)
* Can be skipped if using pre-extracted poses

02_AnomalyDetection.ipynb:
* Runs the experiments and ablations described in results on HR-Avenue
* Skip evaluation of edge system with poses if using pre-extracted poses

03_IsoForest.ipynb:
* Runs the Isolation Forest baseline method for results

04_Ablations.ipynb:
* Produces the analysis and visuals for the results ablations section

005_PerformanceMetricsVSA.ipynb:
* Produces analysis and visuals for the results HR-Avenue core section

006_CombinedMetrics.ipynb:
* Combines metrics for the MoCoDAD (based on pre-saved or reproduced results, see *Note*) with the proposals and baseline forest (visuals: ROC chart and summary plot in summary)

007_Interpretability.ipynb:
* Perform interpretability test qualitative assessment used to build visuals in results Interpretability section

008_YOLOPoseMocodadLatency.ipynb:
* Runs the pose estimation model described in mocodad paper to estimate latency for results comparison

To reproduce the ShanghaiTech results of the report, the notebooks STC-01 and STC-02 should be run:

STC-01.ipynb:
* Runs pose estimation on the STC dataset (canbe skipped if using pre-extracted poses).

STC-02.ipynb:
* Runs the VSA anomaly detection pipeline on the STC dataset outputting the CSV file that is used in the results.

In the paper each of these notebooks are run on Google Colab high RAM instances, for expedience.

## Acknowledgements

The work implements the SORT algorithm first developed by Bewley et al. (2017): 

The algorithm developed implements the _ultralytics_ pre-trained YOLO Pose model: 

The baseline methods use pre-trained MoCoDAD weights provided by Flaborea et al. (2023): 

The method builds on other work which is cited in full in the References section of the written report.

The datasets used come from Liu et al. (2018) (ShanghaiTech) and Lu et al. (2013) (CUHK-Avenue) with filtering rules applied based on Morais et al. (2018). 

