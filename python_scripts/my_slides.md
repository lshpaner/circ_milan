---
title: Circumcision Techniques in Milan - A Comparitive Study
# theme: gaia
paginate: true
backgroundColor: #FFFFFF
# size: 16:9
marp: true
# theme: gaia
# theme: uncover
theme: academic

style: |
  ::backdrop {
    background-color: #FFFFFF;
  }

  .top-right-image {
    position: absolute;
    top: 10px;
    right: 10px;
    width: 200px; /* Adjust width as needed */
  }

  section .larger-text {
    font-size: 50px !important; /* Force smaller font size */   
    line-height: 1.2 !important; /* Adjust spacing between lines */
    margin: 0 !important; /* Remove extra margins */
    padding: 0 !important; /* Remove extra padding */
  }

  section:nth-of-type(1) .larger-text {
    font-size: 55px !important; /* Force smaller font size */
    color: #224466 !important; /* Change text color */
    font-weight: bold !important; /* Make text bold */
    text-align: center !important; /* Center horizontally */

  }

  section .smaller-text {
    font-size: 30px !important; /* Force smaller font size */
    line-height: 1.2 !important; /* Adjust spacing between lines */
    margin: 0 !important; /* Remove extra margins */
    padding: 0 !important; /* Remove extra padding */
    position: fixed;
  }

  section {
    background-image: none;
    font-family: 'Noto Sans JP', sans-serif;
    padding-top: 20px;
    padding-left: 40px;
    padding-right: 40px;
  }

  @keyframes marp-outgoing-transition-slide-up {
    from { transform: translateY(0%); }
    to { transform: translateY(-100%); }
  }
  @keyframes marp-incoming-transition-slide-up {
    from { transform: translateY(100%); }
    to { transform: translateY(0%); }
  }

 
transition: dissolve


---

<div class="larger-text">
Predicting Blood Loss in Circumcision Procedures: Insights from Data
</div>

<br>

<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

![w:640 center](../assets/CUT_MD.svg)

**Leonid Shpaner, M.S. and Dr. Giuseppe Saitta, M.D.**

---
## Background: Understanding Circumcision Techniques

<div class="smaller-text">

- Circumcision is one of the most common surgical procedures worldwide, with two main techniques:
 
    - **Laser Circumcision:** Promoted for precision and faster recovery.
    - **Traditional Circumcision:** Widely practiced but with potential for higher complications.

- Bleeding is a critical complication, impacting recovery and patient satisfaction.
- **Objective:** Predict bleeding to improve surgical planning and outcomes.

---
# Dataset Overview

<!-- <div class="smaller-text"> -->

**Patients:** Data collected from 202 patients undergoing circumcision.

**Features:**

- Age (years)
- BMI
- Comorbidities
- Preoperative Drugs (Antibiotics)
- Surgical Technique (Laser/Traditional)
- Surgical Technique (Laser/Traditional)
- Surgical Technique (Laser/Traditional)


---

# Dataset Overview

<!-- <div class="smaller-text"> -->

**Patients:** Data collected from 202 patients undergoing circumcision.

**Features:**

- Age (years)
- BMI
- Comorbidities
- Preoperative Drugs (Antibiotics)
- Surgical Technique (Laser/Traditional)

---
# Dataset Overview

<!-- <div class="smaller-text"> -->

**Features:**

- Intraoperative Metrics:
  - Mean Heart Rate (bpm)
  - Mean Pulse Oximetry (%)
  - Surgical Time (minutes)
- Anesthesia Type (Lidocaine)
- Systolic Blood Pressure (SBP)
- Diastolic Blood Pressure (DBP)
Total Variables: 11

---

## Methodology: Study Design and Analysis

<!-- <div class="smaller-text"> -->

Comparative analysis of bleeding rates between:
- Laser Circumcision
- Traditional Circumcision

Statistical techniques applied:
- Descriptive analysis of patient and procedural characteristics.
- Correlation analysis to identify key predictors of bleeding.
- Predictive modeling to quantify bleeding risk.
- Tools: `Python 3.11` (Libraries: `eda_toolkit`, `model_tuner`), statistical modeling.
--- 

# Distributions of All Data

![h:560](../images/svg_images/numeric_distributions_1.svg)

---
# Distributions of All Data

![h:560](../images/svg_images/numeric_distributions_2.svg)

---

# Distributions of All Data

![h:560](../images/svg_images/numeric_distributions_3.svg)


---
# Distributions of All Data
**Why Focus on Age and Surgical Times?**
- As we will see on the next slides, age and surgical time are critical factors influencing bleeding risk: 
  - Age distributions differ significantly between patients experiencing bleeding and those who do not.
  - Surgical times vary substantially between techniques, impacting outcomes.
- By zooming in on these distributions, we:
  - Highlight subtle trends.
  - Establish a foundation for deeper predictive modeling.


---
# Distribution of Age

<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

![w:950 center](../images/svg_images/age_distribution.svg)


---
# Distribution of Surgical Time (min)

<style scoped>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

![w:950 center](../images/svg_images/surgical_time_distribution.svg)

---
# Correlation Analysis
### Identifying variables of Interest
- A correlation matrix allows us to:
  - Examine relationships between clinical variables.
  - Highlight key predictors strongly associated with bleeding.

- We will:
  - Focus on high-correlation variables for detailed analysis.
  - Determine which metrics require further investigation in predictive modeling.

---
# Correlation Analysis

![w:900 h:570 center](../images/svg_images/correlation_matrix.svg)

---

# Bleeding Rates

- Laser Circumcision:

  - Lower average bleeding rates.

  - Faster recovery times reported.

- Traditional Circumcision:

  - Higher rates of bleeding complications.

  - More variability in outcomes.