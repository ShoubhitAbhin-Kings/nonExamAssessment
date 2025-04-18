# Neural Networks and Sign Language
### An Investigation Into Real-Time Sign Language Translating

#### Shoubhit Abhin
#### Centre Number – 51123
#### Candidate Number – 8000

---

Please read and understand this README.md file **before** reading the documentation and exploring the code.

---

The directory for this project is as follows:
```
├── CNNModels
  └── model1.h5
  └── model1.keras
  └── model2.h5
  └── model2.keras
  └── model3.h5
  └── model3.keras
  └── model4.h5
  └── model4.keras
  └── model5.h5
  └── model5.keras
  └── model6.h5
  └── model6.keras
├── dataCollectionAndAugmentation
  └── dataAugmentation.py
  └── dataCollection.py
├── extraFiles
  └── linkedListGestureQueue.py
  └── saveAsh5.py
├── filesForTesting
  └── confusionMatrix.py
  └── fileSavingUnitTest.py
  └── gestureProcessingUnitTest.py
  └── queueOperationUnitTest.py
  └── UIDisplayUnitTest.py
├── finalProduct
  └── app.py
├── modelCreationAndTraining
  └── model.py
  └── train.py
├── useThisFolderToRunTheProgram 
  └── app.py
  └── CNNModels
    └── model1.keras
    └── model2.keras
    └── model3.keras
    └── model4.keras
    └── model5.keras
    └── model6.keras
├── dataForAllModels
  └── model1
    └── augmented
    └── notAugmented
    └── trainOnThese
      └── evaluation
        └── A
        └── B
        └── C
        └── D
        └── E
        └── Unknown
      └── train
        └── A
        └── B
        └── C
        └── D
        └── E
        └── Unknown
  └── model2
    └── ... (same as model1)
  └── model3
    └── ... (same as model1)
  └── model4
    └── ... (same as model1)
  └── model5
    └── ... (same as model1)
  └── model6
    └── ... (same as model1)
├── requirements.txt
  
```

---

The following files are solely for the purpose of the examiner as complete evidence of the investigation, not required to run the final product:
* ```dataAugmentation.py```
* ```dataCollection.py```
* ```model.py```
* ```train.py```
* ```confusionMatrix.py```
* ```fileSavingUnitTest.py```
* ```gestureProcessingUnitTest.py```
* ```queueOperationUnitTest.py```
* ```UIDisplayUnitTest.py```
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

It is **imperative** that the models (files ending in ```.keras```) are stored inside a folder called ```"CNNModels"```, otherwise the program **will not** run. If they are stored in a folder with a different name, this **must** be changed to ```"CNNModels"```. For ease of use, the ```"useThisFolderToRunTheProgram"``` folder includes the files listed above in the structure required for the program to run, hence it is recommended to just download this folder to solely run the program (**NOTE - this will be added at a later date**).

It is recommended to create and use a virtual environment to manage the dependencies for this investigation and avoid conflicts with other Python projects on your system. First create and activate a virtual environment (see specific documentation for your system if unsure), and then install the project dependencies listed ```requirements.txt```. This can be done using the command ```pip install -r requirements.txt```.

---

For further instruction on how to run the program, as well as complete documentation of the investigative process, see ```AQAALevelComputerScienceNEADocumentation```.



