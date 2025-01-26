import streamlit as st
import pickle as pkl
import pandas as pd

st.set_page_config(layout="wide")

st.title("IPL Win Predictor")

teams = pkl.load(open('team.pkl', 'rb'))
cities = pkl.load(open('city.pkl', 'rb'))
model = pkl.load(open('model.pkl', 'rb'))

# Ensure cities only contain valid strings
cities = [city for city in cities if isinstance(city, str)]

# First Row and Columns
col1, col2, col3 = st.columns(3)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))
with col3:
    selected_city = st.selectbox('Select the host city', sorted(cities))

# Target score input
target = st.number_input('Target Score', min_value=0, max_value=720, step=1)

# Second Row and Columns
col4, col5, col6 = st.columns(3)
with col4:
    score = st.number_input('Score', min_value=0, max_value=720, step=1)
with col5:
    overs = st.number_input('Overs Done', min_value=0.1, max_value=20.0, step=0.1)  # Minimum overs cannot be zero
with col6:
    wickets_fell = st.number_input('Wickets Fell', min_value=0, max_value=10, step=1)

# Predict button logic
if st.button('Predict Probabilities'):
    # Calculate derived features
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_remaining = 10 - wickets_fell
    crr = score / overs
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0  # Avoid division by zero

    # Prepare input data for the model
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'Score': [score],
        'Wickets': [wickets_remaining],
        'Remaining Balls': [balls_left],
        'target_left': [runs_left],
        'crr': [crr],
        'rrr': [rrr]
    })

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Caveat&display=swap');
    .caveat-font {
        font-family: 'Caveat', cursive;
        text-align: center;
    }
    .green {
        color: green;
    }
    .red {
        color: red;
    }
    </style>
    """, unsafe_allow_html=True)

    # Get predictions
    try:
        result = model.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Display results with color-coded percentages
        if win > loss:
            st.markdown(f"<h2 class='Courier New-font green' style='text-align: center; color: green;'>{batting_team} - {round(win * 100, 2)}%</h2>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='Courier New-font red'style='text-align: center; color: red;'>{bowling_team} - {round(loss * 100, 2)}%</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 class='Courier New-font red'style='text-align: center; color: red;'>{batting_team} - {round(win * 100, 2)}%</h2>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='Courier New-font green'style='text-align: center; color: green;'>{bowling_team} - {round(loss * 100, 2)}%</h2>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
     
st.markdown("<hr style='border: 1px solid #ff4d4d;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center;">
        <p style="font-size: 14px; color: grey;">by Tejas Darunte.</p>
        <p style="font-size: 12px; color: grey;">&copy; 2024 Tejas Darunte. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)