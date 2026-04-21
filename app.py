import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
# This sets the title and icon in the browser tab
st.set_page_config(page_title="Student Marks Predictor", page_icon="🎓", layout="centered")

# --- CUSTOM CSS FOR PREMIUM LOOK ---
# Adding some custom styling to make the app look modern and clean
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #6c5ce7;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #a29bfe;
        color: white;
        border: none;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
        margin-top: 20px;
        border: 1px solid #eee;
    }
    h1 {
        color: #2d3436;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1>🎓 Student Marks Predictor</h1>", unsafe_allow_html=True)
st.write("### A simple AI tool to estimate academic performance")
st.write("This app uses a **Linear Regression** model to predict final marks based on study hours, attendance, and previous scores.")

# --- STEP 1: DATA GENERATION ---
# We create a synthetic dataset to train our model
@st.cache_data
def generate_student_data():
    np.random.seed(42)
    n_samples = 150
    
    # Feature 1: Study Hours (1 to 12 hours)
    study_hours = np.random.randint(1, 12, n_samples)
    # Feature 2: Attendance (60% to 100%)
    attendance = np.random.randint(60, 100, n_samples)
    # Feature 3: Previous Marks (40% to 98%)
    prev_marks = np.random.randint(40, 98, n_samples)
    
    # Target: Final Marks (Logic: Hours weigh most, followed by prev marks and attendance)
    # Formula: Marks = (4 * Hours) + (0.5 * PrevMarks) + (0.1 * Attendance) + Noise
    noise = np.random.normal(0, 3, n_samples)
    marks = (4 * study_hours) + (0.4 * prev_marks) + (0.15 * attendance) + noise
    marks = np.clip(marks, 0, 100) # Ensure marks stay within 0-100 range
    
    df = pd.DataFrame({
        'Study_Hours': study_hours,
        'Attendance': attendance,
        'Previous_Marks': prev_marks,
        'Final_Marks': marks
    })
    return df

# Load the data
df = generate_student_data()

# --- STEP 2: MODEL TRAINING ---
# 1. Define Features (X) and Target (y)
X = df[['Study_Hours', 'Attendance', 'Previous_Marks']]
y = df['Final_Marks']

# 2. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# --- SIDEBAR: USER INPUTS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3413/3413535.png", width=100)
st.sidebar.title("Configuration")
st.sidebar.write("Adjust the sliders below to predict a student's marks.")

input_hours = st.sidebar.slider("Daily Study Hours", 0.0, 12.0, 6.0, 0.5)
input_attendance = st.sidebar.slider("Attendance Percentage (%)", 0, 100, 85)
input_prev_marks = st.sidebar.slider("Previous Exam Marks (%)", 0, 100, 70)

# --- MAIN CONTENT: PREDICTION ---
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Predict Now")
    st.write("Click the button below to see the magic!")
    predict_btn = st.button("Predict Marks")

with col2:
    if predict_btn:
        # Prepare input for prediction
        user_input = np.array([[input_hours, input_attendance, input_prev_marks]])
        prediction = model.predict(user_input)[0]
        prediction = min(max(prediction, 0), 100) # Double check range
        
        # Display the result in a nice box
        st.markdown(f"""
            <div class="prediction-box">
                <p style="font-size: 1.2rem; color: #636e72;">Predicted Final Score</p>
                <h1 style="color: #6c5ce7; font-size: 3rem;">{prediction:.1f}%</h1>
            </div>
        """, unsafe_allow_html=True)

# --- VISUALIZATION ---
st.markdown("---")
st.subheader("Data Insights")
tab1, tab2 = st.tabs(["📊 Training Data", "📈 Trends"])

with tab1:
    st.write("Here is a sample of the synthetic dataset we generated:")
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    st.write("Relationship between Study Hours and Final Marks:")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df['Study_Hours'], df['Final_Marks'], color='#a29bfe', alpha=0.7, edgecolors='white')
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Final Marks")
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)

st.write("")
st.info("💡 **Tip:** Linear Regression finds the 'best fit' line through the data points to make its predictions.")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<center><p style='color: #b2bec3;'>Build with ❤️ for beginner ML enthusiasts</p></center>", unsafe_allow_html=True)
