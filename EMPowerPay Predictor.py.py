import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io

# Function to load and preprocess the data
@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Handle '?' values by replacing them with 'Others' or 'Not Listed'
    df['occupation'] = df['occupation'].replace('?', 'Others')
    df['workclass'] = df['workclass'].replace('?', 'Not Listed')
    df['native-country'] = df['native-country'].replace('?', 'Others')


    # Remove specified workclass categories
    df = df[df['workclass'] != 'Without-pay']
    df = df[df['workclass'] != 'Never-worked']

    # Remove specified education categories
    df = df[df['education'] != 'Preschool']
    df = df[df['education'] != '1st-4th']
    df = df[df['education'] != '5th-6th']

    # Drop original 'education', 'relationship', and 'race' columns
    df.drop('education', axis=1, inplace=True)
    df.drop('relationship', axis=1, inplace=True)
    df.drop('race', axis=1, inplace=True)

    # Outlier handling for 'age'
    df = df[(df['age'] >= 17) & (df['age'] <= 75)]

    # Outlier handling for 'fnlwgt'
    q1_fnlwgt = df['fnlwgt'].quantile(0.25)
    q3_fnlwgt = df['fnlwgt'].quantile(0.75)
    iqr_fnlwgt = q3_fnlwgt - q1_fnlwgt
    lower_bound_fnlwgt = q1_fnlwgt - 1.5 * iqr_fnlwgt
    upper_bound_fnlwgt = q3_fnlwgt + 1.5 * iqr_fnlwgt
    df = df[(df['fnlwgt'] >= lower_bound_fnlwgt) & (df['fnlwgt'] <= upper_bound_fnlwgt)]

    # Outlier handling for 'capital-gain'
    q1_cg = df['capital-gain'].quantile(0.25)
    q3_cg = df['capital-gain'].quantile(0.75)
    iqr_cg = q3_cg - q1_cg
    lower_bound_cg = q1_cg - 1.5 * iqr_cg
    upper_bound_cg = q3_cg + 1.5 * iqr_cg
    df = df[(df['capital-gain'] >= lower_bound_cg) & (df['capital-gain'] <= upper_bound_cg)]

    # Outlier handling for 'capital-loss'
    q1_cl = df['capital-loss'].quantile(0.25)
    q3_cl = df['capital-loss'].quantile(0.75)
    iqr_cl = q3_cl - q1_cl
    lower_bound_cl = q1_cl - 1.5 * iqr_cl
    upper_bound_cl = q3_cl + 1.5 * iqr_cl
    df = df[(df['capital-loss'] >= lower_bound_cl) & (df['capital-loss'] <= upper_bound_cl)]

    # Encode categorical features
    # Initialize a dictionary to store encoders for inverse transformation later
    encoders = {}
    for column in ['workclass', 'marital-status', 'occupation', 'native-country', 'gender']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoders[column] = le

    X = df.drop('income', axis=1)
    y = df['income']

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns, scaler, encoders

# Streamlit UI
st.set_page_config(page_title="Employee Income Prediction", layout="wide")

st.title("Employee Income Prediction")
st.write("Upload your `adult 3.csv` file to predict employee income.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # To convert to a string based IO:
    string_data = io.StringIO(bytes_data.decode('utf-8'))

    with st.spinner("Processing data..."):
        X, y, feature_names, scaler, encoders = load_and_preprocess_data(string_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    st.success("Data processed and split successfully!")

    st.subheader("Train and Evaluate Models")

    # K-Nearest Neighbors
    st.write("### K-Nearest Neighbors (KNN)")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_predict = knn.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_predict)
    st.write(f"KNN Accuracy: {knn_accuracy:.4f}")

    # Logistic Regression
    st.write("### Logistic Regression")
    lr = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
    lr.fit(X_train, y_train)
    lr_predict = lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_predict)
    st.write(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

    # Multi-layer Perceptron (Neural Network)
    st.write("### Multi-layer Perceptron (MLP)")
    mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(5, 2), max_iter=2000, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_predict = mlp.predict(X_test)
    mlp_accuracy = accuracy_score(y_test, mlp_predict)
    st.write(f"MLP Accuracy: {mlp_accuracy:.4f}")

    st.subheader("Make a Prediction")
    st.write("Enter the details below to predict income for a new employee.")

    with st.form("prediction_form"):
        age = st.slider("Age", 17, 75, 30)
        
        # Reverse encode workclass for display
        workclass_options = encoders['workclass'].classes_
        workclass_display = st.selectbox("Workclass", workclass_options)
        workclass = encoders['workclass'].transform([workclass_display])[0]

        fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=12285, max_value=390000, value=150000)
        educational_num = st.slider("Educational Num", 1, 16, 9)
        
        # Reverse encode marital-status for display
        marital_status_options = encoders['marital-status'].classes_
        marital_status_display = st.selectbox("Marital Status", marital_status_options)
        marital_status = encoders['marital-status'].transform([marital_status_display])[0]

        # Reverse encode occupation for display
        occupation_options = encoders['occupation'].classes_
        occupation_display = st.selectbox("Occupation", occupation_options)
        occupation = encoders['occupation'].transform([occupation_display])[0]

        gender_options = encoders['gender'].classes_
        gender_display = st.selectbox("Gender", gender_options)
        gender = encoders['gender'].transform([gender_display])[0]

        capital_gain = st.number_input("Capital Gain", min_value=0, max_value=0, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, max_value=0, value=0)
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
        
        # Reverse encode native-country for display
        native_country_options = encoders['native-country'].classes_
        native_country_display = st.selectbox("Native Country", native_country_options)
        native_country = encoders['native-country'].transform([native_country_display])[0]

        submitted = st.form_submit_button("Predict Income")

        if submitted:
            # Create a DataFrame for the new input
            new_data = pd.DataFrame([[age, workclass, fnlwgt, educational_num, marital_status,
                                      occupation, gender, capital_gain, capital_loss,
                                      hours_per_week, native_country]],
                                    columns=feature_names)

            # Scale the new data using the same scaler fitted on training data
            new_data_scaled = scaler.transform(new_data)

            # Make predictions using the trained MLP model (which had the highest accuracy)
            prediction = mlp.predict(new_data_scaled)

            st.write("### Prediction Result")
            if prediction[0] == '<=50K':
                st.success("Predicted Income: <=50K")
            else:
                st.success("Predicted Income: >50K")

            st.write("---")
            st.write("Disclaimer: The prediction is based on the trained model and the provided input data. Accuracy may vary.")

else:
    st.info("Please upload your `adult 3.csv` file to proceed.")