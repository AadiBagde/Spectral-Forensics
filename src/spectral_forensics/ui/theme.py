import streamlit as st

def inject_custom_css():
    st.markdown("""
        <style>
        /* General background and typography */
        .stApp {
            background-color: #1a1b1e;
            color: #d1d5db;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #f3f4f6;
            font-weight: 500;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #374151;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #9ca3af;
        }

        /* Metrics and Results Panel */
        [data-testid="stMetricValue"] {
            color: #e5e7eb;
            font-family: monospace;
        }
        
        [data-testid="stMetricLabel"] {
            color: #9ca3af;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }

        /* Buttons */
        .stButton button {
            background-color: #1f2937;
            color: #e5e7eb;
            border: 1px solid #4b5563;
            border-radius: 4px;
            transition: all 0.2s ease;
        }
        
        .stButton button:hover {
            background-color: #374151;
            border-color: #60a5fa;
            color: #ffffff;
        }
        
        .stDownloadButton button {
            background-color: #0f766e;
            color: #ffffff;
            border: 1px solid #115e59;
        }
        
        .stDownloadButton button:hover {
            background-color: #115e59;
            border-color: #134e4a;
            color: #ffffff;
        }

        /* Alerts and Boxes */
        .stAlert {
            border-radius: 4px;
            border: 1px solid transparent;
        }
        
        [data-testid="stAlert"] {
            background-color: #1f2937;
            border-color: #374151;
            color: #d1d5db;
        }

        /* Success/Error specifics if possible, though Streamlit isolates them */
        
        /* Dataframes */
        [data-testid="stDataFrame"] {
            border: 1px solid #374151;
            border-radius: 4px;
        }
        
        hr {
            border-top: 1px solid #374151;
        }
        </style>
    """, unsafe_allow_html=True)
