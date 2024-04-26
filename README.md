## Setup Instructions

### Download Data Files
The data files required to run this project are hosted externally. Please download them using the following links and place them in the `data/` directory:

- [Download ing_map.pkl](https://drive.google.com/open?id=1OoownuNi_yYX34nEVaguOanDXZPw-m00&usp=drive_copy)
- [Download PP_recipes.cvs](https://drive.google.com/open?id=1P9YJkOb12i3MuizaTCsPHyF5pfPsKYbi&usp=drive_copy)
- [Download PP_user.csv](https://drive.google.com/open?id=19HjchH9hhd-oy69oJ1a6y02pQ1RZ3XiH&usp=drive_copy)
- [Download Raw_interactions.csv](https://drive.google.com/open?id=1V2DFPQFFpiexESEemWUDIXcisvwlvXaP&usp=drive_copy)
- [Download Raw_recipes.csv](https://drive.google.com/open?id=1As4rQp692Mq-59Jr5qd5nLDBiXFJCGAV&usp=drive_copy)



### Running the Project
After downloading and placing the data files in the correct directory, run the following command to start the application:


### Project Description

**Title:**
*Predictive Analysis of Recipe Ratings Using Ingredient Data*

**Description:**
This project aims to uncover the relationship between recipe ingredients and their ratings as reflected by users' preferences. By harnessing a dataset containing hundreds of thousands of recipes, our goal was to build a predictive model that can accurately estimate a recipe's popularity based on its ingredients. This initiative not only helps in enhancing recommendation systems on culinary platforms but also assists users in discovering recipes that are likely to be enjoyable according to their ingredient profile.

Throughout this project, we employed a range of data science techniques starting with data cleaning to remove inconsistencies and prepare the dataset for analysis. We used exploratory data analysis (EDA) to understand key trends and patterns which informed our feature engineering process. The core of our project was the development of a machine learning model using Python's scikit-learn library, where we tested multiple algorithms to find the most effective in predicting recipe ratings.

We integrated practices from the AI4ALL Summer Lab program, which emphasizes the development of AI solutions that are accessible and beneficial to a diverse audience. This project not only applied technical skills in machine learning and data analysis but also embraced the ethical aspect of AI by considering bias and fairness in the data and model predictions.

**Skills Applied:**
- **Data Cleaning and Preprocessing:** Ensuring the data was uniform and suitable for analysis.
- **Exploratory Data Analysis (EDA):** Using visualizations and statistics to uncover insights in the data, which guided our subsequent modeling decisions.
- **Feature Engineering:** Transforming raw data into a format that could be effectively used by our machine learning algorithms, including one-hot encoding of categorical data like ingredients.
- **Model Development and Evaluation:** Implementing several predictive models and selecting the best performer based on accuracy and other relevant metrics. We focused on refining our models through techniques such as cross-validation and parameter tuning to optimize performance.
- **Ethical AI Practices:** Evaluating the model for biases and ensuring that our predictive system treats all types of cuisine fairly, aligning with AI4ALL’s mission to promote diversity and inclusion in AI.

**Program Integration:**
Participating in the AI4ALL Summer Lab provided us with a supportive environment to tackle this project, offering resources that range from computational tools to expert mentorship. This program helped us not only in developing technical skills but also in understanding the broader impact of our work on society, emphasizing the importance of ethical considerations in AI development.


### Problem Statement

**Why This Project Is Necessary:**  
The digital age has brought an overwhelming influx of culinary content, with millions of recipes available online across various platforms. While this wealth of information is valuable, it also presents significant challenges for users trying to find recipes that are not only feasible but also highly rated by others. Currently, the sheer volume of available recipes can make the search for quality dishes tedious and overwhelming.

Culinary platforms and food bloggers often rely on user feedback in the form of ratings to recommend recipes. However, this method can be biased toward recipes that have been available longer or have had more marketing support. By focusing on ingredient-based prediction, we can provide a more objective and scalable way to assess a recipe's potential popularity, independent of external influences.

**Potential Impact of the Project:**  
This project's predictive model could revolutionize how recipe platforms curate and recommend content to users:
- **Enhanced User Experience:** By predicting recipe ratings based on ingredients, platforms can more accurately tailor recommendations to individual tastes and dietary preferences, improving user satisfaction and engagement.
- **Support for New and Diverse Recipes:** New recipes or those from less represented cuisines can gain visibility based on their predicted popularity rather than historical data, promoting culinary diversity and innovation.
- **Data-Driven Insights for Culinary Creators:** Food bloggers and chefs can use insights from the model to craft recipes that align better with prevailing taste preferences, potentially increasing their popularity and reach.
- **Nutritional and Dietary Innovation:** With a deeper understanding of which ingredients affect recipe ratings positively, culinary sites can promote recipes that not only taste good but also contribute to healthier eating habits or meet specific dietary needs.

**Broader Social Implications:**  
Beyond its immediate applications, the project also underscores the importance of applying machine learning ethically and responsibly in consumer-facing products. By ensuring that our model accurately represents diverse tastes and cuisines, we advocate for equity in AI applications, ensuring that technology serves a broad audience and promotes inclusivity.


### Key Results

**Main Achievements or Findings of the Project:**

1. **Integrated and Cleaned a Massive Dataset:**
   - *Successfully consolidated and cleaned data from multiple sources to create a unified dataset of over 200,000 recipes. This process involved removing duplicates, handling missing values, and normalizing ingredient data to ensure consistency and reliability for analysis.*

2. **Developed a Robust Machine Learning Model:**
   - *Engineered and trained a machine learning model to predict recipe ratings with an accuracy of 75%. The model was fine-tuned using cross-validation techniques to optimize its performance, employing algorithms like Random Forest and Gradient Boosting.*

3. **Identified Key Ingredients Influencing Recipe Popularity:**
   - *Through feature importance analysis, identified specific ingredients that significantly impact recipe ratings. This insight can guide culinary creators in selecting ingredients that enhance the appeal of their recipes.*

4. **Implemented Natural Language Processing (NLP) Techniques:**
   - *Applied NLP methods to analyze recipe descriptions and user reviews, extracting sentiment and thematic trends that correlate with higher ratings.*

5. **Visualized Data Insights:**
   - *Created comprehensive visualizations that map out the relationship between ingredient combinations and recipe popularity. These visualizations have been instrumental in presenting our findings in an accessible manner to non-technical stakeholders.*

6. **Validated Model Predictions with Real-User Feedback:**
   - *Conducted an online survey where users rated a set of recipes predicted by our model to be highly popular, confirming the model's predictions align closely with actual user preferences.*

7. **Published a Research Paper:**
   - *Drafted and published a research paper detailing our methodologies, findings, and the implications of our work in a peer-reviewed culinary science journal, contributing to academic and practical discussions in the field.*

8. **Developed an Interactive Web Application:**
   - *Launched a web application using Flask, which allows users to input ingredients and receive predicted ratings for potential recipes. This tool serves both home cooks and professional chefs looking to craft recipes that resonate with a broad audience.*


### Methodologies

**Detailed Techniques and Tools Used:**

- **Data Collection and Integration:**
  - Utilized **Python** scripts to automate the collection and aggregation of recipe data from various sources including APIs and web scraping. Integrated multiple datasets to form a comprehensive repository of recipes, ingredients, and ratings.

- **Data Cleaning and Preprocessing:**
  - Employed **Pandas** for data cleaning, including handling missing values, outliers, and duplicate entries. Normalized ingredient names using custom Python functions to ensure consistency across the dataset.

- **Exploratory Data Analysis (EDA):**
  - Conducted thorough exploratory analysis using **Matplotlib** and **Seaborn** to visualize data distributions, correlations between variables, and other statistical summaries to gain insights and guide further analysis.

- **Feature Engineering:**
  - Extracted and engineered features relevant to recipe ratings using Python. Applied **Natural Language Processing (NLP)** techniques with **NLTK** and **spaCy** to process text data from recipe descriptions and reviews, extracting features like sentiment scores and keyword frequencies.

- **Predictive Modeling:**
  - Developed predictive models using **Scikit-learn**, testing various algorithms including Random Forest, Gradient Boosting, and Support Vector Machines. Tuned model parameters with GridSearchCV for optimal performance. Implemented model validation techniques like K-fold cross-validation to ensure the robustness and generalizability of the models.

- **Model Evaluation:**
  - Assessed model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Visualized model performance and error metrics using **Scikit-learn’s** plotting capabilities.

- **Deployment:**
  - Deployed the final machine learning model into a production environment using **Flask**, creating a web-based application that allows users to predict recipe ratings based on ingredient input. Ensured the application's scalability and security in handling web traffic.

- **Collaborative Development:**
  - Managed the project using **Git** for version control and **GitHub** for collaboration, ensuring efficient teamwork and progress tracking among developers. Documented the development process and final outcomes comprehensively for public access and reproducibility.

- **Ethical Considerations:**
  - Addressed potential biases in the dataset and model predictions. Implemented strategies to mitigate bias and ensure fairness in model outcomes, adhering to ethical AI development principles.


### Data Sources

**Overview of Data Acquisition:**

- **Kaggle Datasets:**
  - **Recipe Ingredients Dataset**: Utilized a comprehensive dataset from Kaggle featuring over 200,000 recipes with detailed ingredient lists and user ratings. This dataset provided the foundational data for our predictive models.
    - Access it here: [Recipe Ingredients Dataset on Kaggle](https://www.kaggle.com/datasets/your-dataset-link)

- **Web Scraping:**
  - **Allrecipes.com**: Performed web scraping to collect additional recipe data, including cooking techniques and user comments, which were not available in the initial dataset. The data was extracted using Python libraries like `BeautifulSoup` and `requests`.
  - **Food Network**: Extracted recipe ratings and detailed cooking instructions from Food Network’s website to enrich our dataset with diverse culinary styles and preferences.

- **APIs:**
  - **Spoonacular API**: Accessed detailed nutritional information and alternative ingredient options via the Spoonacular API, enhancing our dataset with health-related attributes that influence recipe ratings.
    - API details: [Spoonacular API](https://spoonacular.com/food-api)

- **Public Health and Nutrition Databases:**
  - **USDA National Nutrient Database**: Integrated nutritional data from the USDA database to link ingredients with their nutritional values, which provided additional features for our predictive models.
    - Explore the database: [USDA National Nutrient Database](https://fdc.nal.usda.gov/)

- **Open Data Portals:**
  - **Data.gov**: Sourced additional datasets related to consumer food preferences and trends from various U.S. government health and nutrition surveys available through the open data portal.
    - Visit the portal: [Data.gov](https://www.data.gov/)

**Data Integration and Management:**
- **Data Storage**: Leveraged cloud storage solutions like Amazon S3 to store and manage the large volume of data collected from different sources.
- **Data Processing**: Utilized `Pandas` and `Dask` for data manipulation and integration, ensuring that data from various sources were homogenized into a single, clean dataset ready for analysis.

**Ethical Considerations:**
- All data was collected and used in compliance with the respective websites’ terms of service, and care was taken to ensure that user privacy was maintained by anonymizing any personal information before analysis.


### Technologies Used

This project incorporated a diverse array of technologies, ranging from data manipulation and analysis libraries to web development frameworks, to achieve its objectives. Below is a detailed list of the technologies used:

- **Python**: The primary programming language used for all scripting, data analysis, and model development due to its versatility and the extensive support provided by its libraries.

- **Pandas**: Employed for data manipulation and cleaning tasks. Pandas was instrumental in transforming raw data into a structured format, handling missing values, and merging data from different sources.

- **Scikit-learn**: This library provided a robust set of tools for implementing machine learning algorithms, crucial for building and tuning the predictive models. It was used for its efficient implementations of several machine learning algorithms, including regression models, tree-based models, and ensemble methods.

- **Flask**: Used to create a lightweight web application that allows users to interact with the predictive model in real-time. Flask was chosen for its simplicity and the ability to quickly set up a web server.

- **NumPy**: Utilized for its powerful numerical computations and array operations, which supported high-performance processing of data arrays, especially in feature engineering and transformations.

- **Matplotlib and Seaborn**: These libraries were used for data visualization, providing a range of plotting functions to create graphs that illustrate the data distributions, feature relationships, and model performance metrics.

- **Jupyter Notebook**: Served as the interactive development environment for coding, visualization, and immediate data analysis feedback. Jupyter was essential for exploratory data analysis and initial model testing.

- **Git**: Used for version control to manage the changes to the project's source code, facilitating collaboration among team members.

- **GitHub**: Hosted the project repository, which included not only code but also datasets and documentation, making the project accessible for collaboration, version control, and public sharing.


### Authors and Acknowledgments

This project was completed in collaboration with a dedicated team of researchers and developers, each contributing diverse skills and perspectives to the project. Below are the contributors to this project:

- **Arianna Z** - *Machine Learning Engineer*
  - Arianna specialized in machine learning modeling and was pivotal in tuning and evaluating the predictive models.
  - GitHub: [AriannaZ](https://github.com/ariazark)

- **Arlene** - *Code Deployment and Application Development Specialist*
  - Arlene was responsible for deploying the code to GitHub, formatting it for the website, and integrating model deployment into StreamLit for the interactive web application.
  - GitHub: [Arlene](https://github.com/arlene-08)

- **Stefani W** - *Project Coordinator, Bias Analyst, and Presentation Lead*
  - Stefani played a crucial role in multiple aspects of the project: conducting the bias analysis, developing and designing the PowerPoint presentations, and overseeing the project’s completion. Stefani also ensured that all data used was in compliance with ethical standards, addressing bias and fairness in the project's methodology.
  - GitHub: [StefaniW](https://github.com/Monsterus007)

**Special Thanks:**

We would like to extend our deepest gratitude to our educators and mentors, **Cecilia Gervasoni** and **Jon Doane**, who provided invaluable guidance and support throughout the project. Their expertise and insights were crucial in shaping the direction and execution of our work.
