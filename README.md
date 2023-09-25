
  
# Data Protection Techniques: k-Anonymity, l-Diversity, t-Closeness, and Pseudonymization  
  
This repository contains implementation of several data protection techniques alongside the original solution.   
  
  
### k-Anonymity  
  
k-Anonymity is a data anonymization technique that prevents the re-identification of individuals in a dataset. It ensures that individual data records cannot be easily distinguished from at least k-1 other records based on the quasi-identifiers. The implementation employs the Mondrian algorithm to partition the data based on certain criteria and using aggregation functions like `mean()` to preserve data utility while anonymizing. This is mainly because the data will be used as input to an ML model.

#### Implementation 
For each partition, the algorithm starts by calculating the relative span of each column. Then, it sorts the columns in descending order and choosing the column with the largest span. Using the median of the column as the cut off point, it creates new partitions. If the new partitions satisfy the criterias, they are added to the finished partition list. In the end, the aggregated function is applied to each partition, and the anonymized dataset is generated.

The function sequence: 
- `partition()`
- `get_span()`
- `split()`
- `is_valid()`which checks if the partitions satisfy k-anonymity, l-diversity, or t-closeness

Implementation: `dataprotection_features/anonymize_module.py`  
### l-Diversity  
Building upon k-Anonymity, l-Diversity prevents homogeneity and background knowledge attacks by ensuring that in each k-anonymous group, there are at least l distinct values for the sensitive column. l-Diversity is implemented by modifying `is_valid()` so that it creates diverse groups.
  
Implementation:`dataprotection_features/anonymize_module.py`  
  
  
### t-Closeness  
  
t-Closeness further enhances privacy by ensuring that distribution of the sensitive column in partitions closely match the entire dataset, preventing attacks when the sensitive column is skewed.
  
Implementation:`dataprotection_features/anonymize_module.py`  
  
  
### Other Features  
  
The repository includes functionalities for calculating k-anonymity and l-diversity, as well as pseudonymization for Personally Identifiable Information (PII) data using SHA-256 hashing.  
  
Implementation of metric calculation:`dataprotection_features/calcmetric.py`  
Implementation of pseudonymization:`dataprotection_features/pseudo.py`  
  
## Usage  
```  
pip3 install -r requirements.txt
 ```  
There are several jupyter notebooks available in `dataprotection_features` directory showcasing usage of the features.  
Anonymization (k-anonymity, l-diversity, t-closeness, and calculation of these metrics): `dataprotection_features/adult.ipynb`, `dataprotection_features/diabetes.ipynb`
Pseudonymization: `dataprotection_features/pseudo.ipynb`
  
## References  
- Nithin Prabhu, K-Anonymity, [https://github.com/Nuclearstar/K-Anonymity](https://github.com/Nuclearstar/K-Anonymity)  
- Taisuke Fujita, anonypy, [https://github.com/glassonion1/anonypy](https://github.com/glassonion1/anonypy)  
- Ayala-Rivera, Vanessa & Mcdonagh, Patrick & Cerqueus, Thomas & Murphy, Liam. (2014). A Systematic Comparison and Evaluation of k-Anonymization Algorithms for Practitioners. Transactions on Data Privacy. 7. 337-370.   
- Latanya Sweeney. 2002. K-anonymity: a model for protecting privacy. Int. J. Uncertain. Fuzziness Knowl.-Based Syst. 10, 5 (October 2002), 557–570. https://doi.org/10.1142/S0218488502001648  
- A. Machanavajjhala, J. Gehrke, D. Kifer and M. Venkitasubramaniam, "L-diversity: privacy beyond k-anonymity,"  _22nd International Conference on Data Engineering (ICDE'06)_, Atlanta, GA, USA, 2006, pp. 24-24, doi: 10.1109/ICDE.2006.1.  
- K. LeFevre, D. J. DeWitt and R. Ramakrishnan, "Mondrian Multidimensional K-Anonymity,"  _22nd International Conference on Data Engineering (ICDE'06)_, Atlanta, GA, USA, 2006, pp. 25-25, doi: 10.1109/ICDE.2006.101.