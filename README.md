# Neural Networks and Sign Language
### An Investigation Into Real-Time Sign Language Translating

#### Shoubhit Abhin
#### Centre Number – 51123
#### Candidate Number – 8000

---

Please read and understand this README.md file **before** reading the documentation and exploring the code.
At the beginning of this project, two possible approaches for real-time sign language recognition were considered: a non-machine learning (non-ML) approach and a machine learning (ML) approach. Further details can be found in the documentation (```AQAALevelComputerScienceNEADocumentation```). This message is included here so that references to the 'ML' and 'non-ML' approach from here on in are understood.

---

The video evidence, models, and sample images for the project can be found on the project sharepoint - https://kingsacademiesuk-my.sharepoint.com/:f:/g/personal/j_campbell_kingsacademies_uk/EsYdvlKx5MRNoo3OGHByJX0Bu2HKxMW7jqdluUf9fpMjTA?e=gq1Dm1

---
NOTE: The ```noMLProc.py```, ```unitTestLetterB``` and ```unitTestLetterE``` files are for the non-ML approach only, hence not included in the directory below, which outlines the directory for the ML approach.

The directory for this GitHub Repository is as follows:
```
├── dataCollectionAndAugmentation
  └── dataAugmentation.py
  └── dataCollection.py
├── extraFiles
  └── linkedListGestureQueue.py
  └── saveAsh5.py
├── filesForTesting
  └── binaryClassificationMatrix.py
  └── confusionMatrix.py
  └── dataAugmentationDotPyTest.py
  └── dataCollectionDotPyTest.py
  └── gestureQueueLogicTest.py
  └── modelDotPyTest.py
  └── trainDotPyTest.py
├── finalProduct
  └── app.py
├── modelCreationAndTraining
  └── model.py
  └── train.py 
  └── app.py
├── non-ML Approach
  └── nonMLProc.py
  └── unitTestLetterB.py
  └── unitTestLetterE.py
├── requirements.txt
├── AQAALevelComputerScienceNEADocumentation
├── README.md
  
```

---

The following files are solely for the purpose of the examiner as complete evidence of the investigation, not required to run the final product:
* ```dataAugmentation.py```
* ```dataCollection.py```
* ```model.py```
* ```train.py```
* ```confusionMatrix.py```
* ```binaryClassificationMatrix.py```
* ```dataAugmentationDotPyTest.py```
* ```dataCollectionDotPyTest.py```
* ```gestureQueueLogicTest.py```
* ```modelDotPyTest.py```
* ```trainDotPyTest.py```
* ```saveAsh5.py```
* ```linkedListGestureQueue.py```

---

The following files are the files a user would need installed on their local machine to run the program. These are:
* ```app.py```
* ```model1.keras```
* ```model2.keras```
* ```model3.keras```
* ```model4.keras```
* ```model5.keras```
* ```model6.keras```

The **model** files can be downloaded directly from the project sharepoint - https://kingsacademiesuk-my.sharepoint.com/:f:/g/personal/j_campbell_kingsacademies_uk/EsYdvlKx5MRNoo3OGHByJX0Bu2HKxMW7jqdluUf9fpMjTA?e=gq1Dm1

It is **imperative** that the models (files ending in ```.keras```) are stored inside a folder called ```"CNNModels"```, otherwise the program **will not** run. If they are stored in a folder with a different name, this **must** be changed to ```"CNNModels"```. 

It is recommended to create and use a virtual environment to manage the dependencies for this investigation and avoid conflicts with other Python projects on your system. First create and activate a virtual environment (see specific documentation for your system if unsure), and then install the project dependencies listed ```requirements.txt```. This can be done using the command ```pip install -r requirements.txt```.

---

The video evidence for the project can be found on the project sharepoint - https://kingsacademiesuk-my.sharepoint.com/:f:/g/personal/j_campbell_kingsacademies_uk/EsYdvlKx5MRNoo3OGHByJX0Bu2HKxMW7jqdluUf9fpMjTA?e=gq1Dm1

If there are issues accessing the sharepoint or other files, please contact james.campbell@kingsacademies.uk (Head of Computing - King's Academy Binfield)

For further instruction on how to run the program, as well as complete documentation of the investigative process, see ```AQAALevelComputerScienceNEADocumentation```.



