import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load mô hình hồi quy tuyến tính đã được lưu
model = pickle.load(open('linear.pkl', 'rb'))
# load mô hình chuẩn hóa Min-Max từ tệp "minmax_scaler_x.pkl"
with open("minmax_scaler_X.pkl", "rb") as scaler_file:
    loaded_minmax_scale = pickle.load(scaler_file)

# Define a function to make predictions
def predict_house(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity):
    # Tạo mảng 1 chiều từ input_data
    input_data = np.array([longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, 0, 0, 0, 0, 0])
    if ocean_proximity == '<1H OCEAN':
        input_data [8] = 1
    elif ocean_proximity == 'INLAND':
        input_data [9] = 1
    elif ocean_proximity == 'ISLAND':
        input_data [10] = 1
    elif ocean_proximity == 'NEAR BAY':
        input_data [11] = 1
    elif ocean_proximity == 'NEAR OCEAN':
        input_data [12] = 1

    # Chuẩn hóa dữ liệu input_data bằng mô hình chuẩn hóa Min-Max
    input_data_normalized = loaded_minmax_scale.transform(input_data.reshape(1, -1))

    # Dự đoán giá trị bằng mô hình Linear Regressor đã nạp
    predicted_house = model.predict(input_data_normalized)

    return predicted_house[0]

# Create a Streamlit web app
st.title('House Price Forecast')
st.sidebar.header('Input Features')

# Input fields for user to enter feature values
longitude = st.sidebar.number_input('longitude', value=0.0)
latitude = st.sidebar.number_input('latitude', value=0.0)
housing_median_age  = st.sidebar.number_input('housing_median_age', value=0)
total_rooms = st.sidebar.number_input('total_rooms', value=0)
total_bedrooms = st.sidebar.number_input('total_bedrooms', value=0)
population = st.sidebar.number_input('population', value=0.0)
households = st.sidebar.number_input('households', value=0.0)
median_income = st.sidebar.number_input('median_income', value=0.0)
ocean_proximity = st.sidebar.selectbox('ocean_proximity', ('<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'))

# Định nghĩa CSS trực tiếp bằng cách sử dụng st.markdown
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f0; /* Nền màu xám nhạt */
    }
    .sidebar {
        background-color: #e0e0e0; /* Nền màu xám đậm hơn */
    }
    .title {
        color: #007bff; /* Màu xanh dương */
    }
    </style>
     <style>
    h1 {
        color: red;
        font-size: 40px;
    }
    </style>
    <style>
    .red-text {
        color: red;
        font-size: 30px;  /* Thay đổi cỡ chữ thành 30px */
    }
    </style>

    <style>
    .edit-text_blue {
    color: #007FFF; /* Xanh dương sáng */
    font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#Tạo 1 list để lưu dữ liệu dự đoán sau mỗi lần ấn nút predict bằng session state
if "predicted_houses" not in st.session_state:
    st.session_state.predicted_houses = [] # tao list ể lưu trữ các giá trị dự đoán


if st.sidebar.button('Predict'):
    predicted_house = predict_house(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity )
    predicted_house = float(predicted_house)
    st.session_state.predicted_houses.append(predicted_house)  # Lưu trữ kết quả dự báo vào mảng
    st.markdown(f'<p class="red-text">Predicted Price House: ${(predicted_house):,.2f}</p>', unsafe_allow_html=True)


    plt.figure(figsize=(8, 6), facecolor='black')
    plt.gcf().set_facecolor('black')
    plt.plot(range(1, len(st.session_state.predicted_houses) + 1), st.session_state.predicted_houses, color='lime')
    plt.xlabel('Prediction Number', color='white')
    plt.ylabel('Predicted Price House', color='white')
    plt.title('House price prediction chart ', color='white')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    st.pyplot(plt)
    # Đặt lại biểu đồ để tránh trùng lặp
    plt.clf()
    # Hiển thị danh sách  dự đoán từ các lần nhấn trước đó
    if st.session_state.predicted_houses:
        st.write("Danh sách dự đoán từ các lần nhấn trước đó:")
        for i, salary in enumerate(st.session_state.predicted_houses):
            #salary = float(salary)
            st.markdown(f'<p class="edit-text_blue">Dự đoán {i + 1}: ${salary:,.2f}</p>', unsafe_allow_html=True)

        # Hiển thị hình ảnh dựa trên giá trị predicted_price
        if predicted_house < 300000:
            st.image("image1.png", use_column_width=True)
        elif 300000 <= predicted_house <= 500000:
            st.image("image2.png", use_column_width=True)
        else:
            st.image("image3.png", use_column_width=True)

# This will clear the user inputs
if st.sidebar.button('Reset'):
    # Đặt lại tất cả các giá trị về mặc định
    st.session_state['longitude'] = 0.0
    st.session_state['latitude'] = 0.0
    st.session_state['housing_median_age'] = 0.0
    st.session_state['total_rooms'] = 0
    st.session_state['total_bedrooms'] = 0
    st.session_state['population'] = 0.0
    st.session_state['households'] = 0.0
    st.session_state['median_income'] = 0.0
    st.session_state['ocean_proximity'] = '<1H OCEAN'

    # Xóa danh sách predicted_prices
    st.session_state.predicted_houses = []

# Provide some information about the app
st.write('This app predicts House Price using a Linear Regression model.')

