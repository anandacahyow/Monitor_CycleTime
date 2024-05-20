import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np
import math
from PIL import Image

img = Image.open('Nestle_Logo.png')
st.set_page_config(page_title="Maggi Cycle Count", page_icon=img,layout="wide")

def seconds_to_time(seconds):
    try:
        seconds = float(seconds)
        if math.isnan(seconds):
            return "00:00:00"
        hours = int(seconds) // 3600
        minutes = (int(seconds) % 3600) // 60
        seconds = int(seconds) % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except (ValueError, TypeError):
        return "00:00:00"

def load_data(file):
    header = [
        'RECORD_Time', 'RECORD_Date', 'RECORD_Shift', 'BATCHING_No', 'BATCHING_Recipe code', 'BATCHING_Batch size', 
        'BATCHING_Batch Status', 'BATCHING_Batch Time_Start', 'BATCHING_Batch Time_Ready', 'BATCHING_Material ID 1_Name', 
        'BATCHING_Material ID 1_Set', 'BATCHING_Material ID 1_Act', 'BATCHING_Material ID 1_Diff.', 'BATCHING_Material ID 2_Name', 
        'BATCHING_Material ID 2_Set', 'BATCHING_Material ID 2_Act', 'BATCHING_Material ID 2_Diff.', 'BATCHING_Material ID 3_Name', 
        'BATCHING_Material ID 3_Set', 'BATCHING_Material ID 3_Act', 'BATCHING_Material ID 3_Diff.', 'BATCHING_Material ID 4_Name', 
        'BATCHING_Material ID 4_Set', 'BATCHING_Material ID 4_Act', 'BATCHING_Material ID 4_Diff.', 'BATCHING_Material ID 5_Name', 
        'BATCHING_Material ID 5_Set', 'BATCHING_Material ID 5_Act', 'BATCHING_Material ID 5_Diff.', 'BATCHING_Material ID 6_Name', 
        'BATCHING_Material ID 6_Set', 'BATCHING_Material ID 6_Act', 'BATCHING_Material ID 6_Diff.', 'BATCHING_Actual Size', 
        'BATCHING_Material ID 7_Name', 'BATCHING_Material ID 7_Set', 'BATCHING_Material ID 7_Act', 'BATCHING_Material ID 7_Diff.', 'BATCHING_Material ID 7_Spare',
        'TIME_WT Discharge', 'TIME_Start', 'TIME_Mixer Discharge', 'TIME_Discharge Complete', 'TIME_Spare', 'Cycle Time_ID 1', 'Cycle Time_ID 2', 
        'Cycle Time_ID 3', 'Cycle Time_ID 7', 'Cycle Time_Batching', 'Cycle Time_WT Discharge', 'Cycle Time_ID 4', 'Cycle Time_ID 5', 
        'Cycle Time_ID 6', 'Cycle Time_Mixing', 'Cycle Time_Discharging', 'Cycle Time_Total Batch'
    ]
    try:
        df = pd.read_csv(file, header=[0, 1, 2])
        df.columns = header
        df['Datetime'] = pd.to_datetime(df['RECORD_Date'] + ' ' + df['RECORD_Time'])

        for i in range(1, len(df)):
            if df.loc[i, 'Datetime'] < df.loc[i - 1, 'Datetime']:
                df.loc[i, 'Datetime'] += timedelta(days=1)
        
        averages = []
        current_shift = None
        shift_data = []
        for index, row in df.iterrows():
            if row['RECORD_Shift'] != current_shift:
                if current_shift is not None:
                    # Calculate average for the previous shift
                    avg = sum(shift_data) / len(shift_data)
                    averages.extend([avg] * len(shift_data))
                current_shift = row['RECORD_Shift']
                shift_data = [row['Cycle Time_ID 1']]
            else:
                shift_data.append(row['Cycle Time_ID 1'])

        # Calculate average for the last shift
        if shift_data:
            avg = sum(shift_data) / len(shift_data)
            averages.extend([avg] * len(shift_data))

        # Add 'Average' column to DataFrame
        df['Average'] = averages

        df.drop(columns=['RECORD_Date', 'RECORD_Time'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def plot_data(df):
    fig = go.Figure()
    cumulative_sum = [0] * len(df)

    for i in range(1, 8):
        # Convert the column to numeric to ensure it's an integer
        df[f'Cycle Time_ID {i}'] = pd.to_numeric(df[f'Cycle Time_ID {i}'], errors='coerce')
        
        cumulative_sum = [a + b for a, b in zip(cumulative_sum, df[f'Cycle Time_ID {i}'])]
        fig.add_trace(
            go.Scatter(
                x=df['Datetime'],
                y=cumulative_sum,
                mode='lines',
                name=f'Cycle Time {i}',
                stackgroup='one',
                hovertemplate='Datetime: %{x}<br>Duration: %{text}'
            )
        )

    for trace in fig.data:
        trace.text = [seconds_to_time(y) for y in trace.y]
    
    fig.add_trace(
            go.Scatter(
                x=df['Datetime'],
                y=df['Average'],
                mode='lines',
                name=f'Average Line',
                line=dict(width=4)
            )
        )

    fig.update_layout(title='Stacked Line Chart of Cycle Time IDs 1 to 7',
                    xaxis_title='Datetime',
                    yaxis_title='Cumulative Cycle Time',
                    hovermode='x',
                    template='plotly_dark',
                    width = 1250,
                    height = 800)
    #fig.update_traces(mode='markers')

    st.plotly_chart(fig)


def main():
    st.title("Cycle Time Analysis")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None:
            # Set default date value to minimum date in the dataset
            min_date = df['Datetime'].min().date()
            max_date = df['Datetime'].max().date()
            default_date = min_date

            # Date filter
            # Set default date value to May 1, 2024
            default_start_date = datetime(2024, 1, 1).date()
            start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=default_start_date)
            end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

            filtered_df = df[(df['Datetime'] >= pd.Timestamp(start_date)) & (df['Datetime'] <= pd.Timestamp(end_date))]

            # Shift filter
            shifts = filtered_df['RECORD_Shift'].unique().tolist()
            selected_shift = st.sidebar.selectbox("Select Shift", ['All'] + shifts)

            if selected_shift != 'All':
                filtered_df = filtered_df[filtered_df['RECORD_Shift'] == selected_shift]
            
            st.write(filtered_df)

            plot_data(filtered_df)
    st.sidebar.image("Nestle_Signature.png")
    st.sidebar.write("""<p style='font-size: 14px;'>This Web-App is designed to facilitate monitoring of Maggi Mixing Cycle Count Report""")
    st.sidebar.write("""<p style='font-size: 13px;'>For any inquiries, error handling, or assistance, please feel free to reach us through Email: <br>
        <a href="mailto:Ananda.Cahyo@id.nestle.com">Ananda.Cahyo@id.nestle.com <br></p>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
