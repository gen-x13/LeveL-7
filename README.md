### **Tsunami & Earthquake Visualization and Prediction**

- **LEVEL 7 - Overview:**  

  This project maps, analyzes, and predicts earthquake and tsunami risks using historical data from 2001 to 2020, along with real-time seismic data.

  It uses structured data : 

    - from the Kaggle dataset : [Tsunami and Earthquake Risk Assessment](https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset/data) (used for the mapping, estimation and prediction pages), 
    - from Geonames (used to simplify geographic mapping) and from the USGS API (used for the early warning page). 

  It includes : 

    - EDA, 
    - Supervised learning algorithm (AdaBoostClassifier and DecisionTreeClassifier), 
    - Machine learning pipelines (with RandomizedSearchCV), 
    - Interactive and animated maps and dashboards (coming soon).

  Main purposes : 

     - to identify the regions most at risk for tsunamis over times and in real-time conditions.
     - to recreate a tsunami warning system with using earthquake magnitude and depth data.


  There is only version available:
  - 🇬🇧 **English version**: Includes English commentary (`-en`).

- **Tech Stack:** Python, Pandas, Streamlit, Scikit-learn, PIL, Plotly.  

---

⚠️ Reminder: This is a beginner-level project : a test, an exploration, a digital sketchbook.         
⚠️ Note: This app might take a few seconds to load if it's been inactive — please be patient while it wakes up!    
⚠️ Note: The real-time prediction is just a warning, it's not in any case a real prediction for an occuring tsunami.

---



---
### **How to Run the Project:**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/gen-x13/Level-7
   ```
---

### **##Demo**

👉 [Click here to try it now !](https://worldoceanhazard.streamlit.app/)

---

### **Requirements**  
Before running the project, make sure you have the following libraries installed:  
```bash
pip install pandas streamlit plotly scikit-learn pillow

```
---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

---

### **💜 A Reminder:**

***Slow Progress beats No Progress***
*First working draft to perfection.*

> ### Link to the bilingual blog talking about this project (french & english):
> ### [Blog Tsunami Risk Project | Projet Risque de Tsunami Link](https://ko-fi.com/post/Tsunami-Risk-Project-Projet-Risque-de-Tsunami-W7W21SAPQQ)
