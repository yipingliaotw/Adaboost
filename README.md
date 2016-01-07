- Usage:
  - Generate decision stump as weak classifiers
    - Input
      - x: training data
      - stump_constraint: if the distance between data samples is
    smaller than the constraint, the decision stump won't be geneated.
    
  ```python
    weak_classifier_list=GenWeakClassifier(x,stump_constraint)
  ```
  - Training process
    - Input
      - weak_classifier_list
      - x: training data
      - y: training target
    - Output: strong classifer

  ```python
  g_list = ada.AdaboostTrain(wkclassifier_list,x,y)
  ```

  - Test process
    - Input:
      - strong classifer
      - test data
    - Output:
      - test result
