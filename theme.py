import streamlit as st
 
 
def apply_theme():
    st.markdown(
        """
        <style>
        /* =========================================================
           GLOBAL PAGE COLOURS
        ========================================================= */
 
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
 
        /* =========================================================
           NORMAL TEXT, LABELS AND DESCRIPTIONS
        ========================================================= */
 
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp div,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span {
            color: #ffffff;
        }
 
        /* Widget labels */
        .stCheckbox label,
        .stRadio label,
        .stSelectbox label,
        .stMultiSelect label,
        .stNumberInput label,
        .stTextInput label,
        .stTextArea label,
        .stSlider label,
        .stFileUploader label,
        .stDateInput label,
        .stTimeInput label {
            color: #ffffff !important;
        }
 
        /* Checkbox and radio option text */
        [data-testid="stCheckbox"] label p,
        [data-testid="stCheckbox"] label span,
        [data-testid="stRadio"] label p,
        [data-testid="stRadio"] label span {
            color: #ffffff !important;
            opacity: 1 !important;
        }
 
        /* Small captions, help text and file-upload instructions */
        .stCaption,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] *,
        small {
            color: #d6d6d6 !important;
            opacity: 1 !important;
        }
 
        /* =========================================================
           NUMBER INPUTS AND TEXT INPUTS
        ========================================================= */
 
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            background-color: #f4f5f8 !important;
            color: #171717 !important;
            -webkit-text-fill-color: #171717 !important;
            opacity: 1 !important;
        }
 
        /* Number input plus/minus buttons */
        [data-testid="stNumberInput"] button {
            background-color: #f4f5f8 !important;
            color: #171717 !important;
        }
 
        [data-testid="stNumberInput"] button svg {
            fill: #171717 !important;
            color: #171717 !important;
        }
 
        /* =========================================================
           DISABLED WIDGETS
        ========================================================= */
 
        input:disabled,
        textarea:disabled,
        button:disabled,
        [aria-disabled="true"] {
            opacity: 1 !important;
        }
 
        input:disabled,
        textarea:disabled {
            background-color: #dedfe3 !important;
            color: #555555 !important;
            -webkit-text-fill-color: #555555 !important;
        }
 
        /* Keep disabled widget labels readable */
        [data-testid="stCheckbox"]:has(input:disabled) label p,
        [data-testid="stCheckbox"]:has(input:disabled) label span,
        [data-testid="stNumberInput"]:has(input:disabled)
        [data-testid="stWidgetLabel"] p {
            color: #bdbdbd !important;
            opacity: 1 !important;
        }
 
        /* =========================================================
           SELECTBOX AND MULTISELECT
        ========================================================= */
 
        [data-baseweb="select"] > div {
            background-color: #f4f5f8 !important;
            color: #171717 !important;
        }
 
        [data-baseweb="select"] input,
        [data-baseweb="select"] span,
        [data-baseweb="select"] div {
            color: #171717 !important;
            -webkit-text-fill-color: #171717 !important;
        }
 
        /* Dropdown menu */
        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [role="listbox"] {
            background-color: #ffffff !important;
        }
 
        [role="option"],
        [role="option"] * {
            color: #171717 !important;
            background-color: #ffffff !important;
        }
 
        [role="option"]:hover,
        [role="option"]:hover * {
            background-color: #e8e8e8 !important;
            color: #000000 !important;
        }
 
        /* =========================================================
           SLIDERS
        ========================================================= */
 
        [data-testid="stSlider"] p,
        [data-testid="stSlider"] span {
            color: #ffffff !important;
        }
 
        /* =========================================================
           EXPANDERS, TABS AND DATAFRAMES
        ========================================================= */
 
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary p {
            color: #ffffff !important;
        }
 
        button[data-baseweb="tab"] p,
        button[data-baseweb="tab"] span {
            color: #ffffff !important;
        }
 
        /* =========================================================
           HEADINGS
        ========================================================= */
 
        h1 {
            color: #ef4b4b !important;
        }
 
        h2,
        h3,
        h4,
        h5,
        h6 {
            color: #ffffff !important;
        }
 
        /* =========================================================
           SAFE WIDGET LABEL FIX
           Keeps main-page widget labels readable without recoloring
           the sidebar, selectbox contents, buttons, or input values.
        ========================================================= */
 
        [data-testid="stMain"] [data-testid="stWidgetLabel"],
        [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
        [data-testid="stMain"] [data-testid="stWidgetLabel"] span,
        [data-testid="stMain"] [data-testid="stCheckbox"] label p,
        [data-testid="stMain"] [data-testid="stCheckbox"] label span,
        [data-testid="stMain"] [data-testid="stRadio"] label p,
        [data-testid="stMain"] [data-testid="stRadio"] label span {
            color: #ffffff !important;
            opacity: 1 !important;
        }
 
        /* Disabled widget labels should remain visible but softer */
        [data-testid="stMain"] [aria-disabled="true"]
        [data-testid="stWidgetLabel"],
        [data-testid="stMain"] [aria-disabled="true"]
        [data-testid="stWidgetLabel"] * {
            color: #bdbdbd !important;
            opacity: 1 !important;
        }
 
        /* Restore readable sidebar colours */
        [data-testid="stSidebar"] {
            background-color: #111111 !important;
        }
 
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
 
        [data-testid="stSidebar"] [aria-current="page"] {
            background-color: #ef4b4b !important;
        }
 
        /* Keep Streamlit's top header dark */
        [data-testid="stHeader"] {
            background-color: #000000 !important;
        }
 
 
        /* =========================================================
           PAGE TITLE BETWEEN LOGOS
        ========================================================= */
 
        .icaav-page-title {
            text-align: center;
            color: #ef4b4b !important;
            font-size: 2rem;
            font-weight: 700;
        }
 
        .icaav-page-subtitle {
            text-align: center;
            color: #ef4b4b !important;
            font-size: 1rem;
        }
 
 
 
        /* =========================================================
           FILE UPLOADER VISIBILITY FIX
        ========================================================= */
 
        [data-testid="stFileUploaderDropzone"] {
            background-color: #f4f5f8 !important;
            border: 1px solid #d8dbe2 !important;
        }
 
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploaderFile"] *,
        [data-testid="stFileUploaderDropzoneInstructions"] *,
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small {
            color: #2b2f36 !important;
            -webkit-text-fill-color: #2b2f36 !important;
            opacity: 1 !important;
        }
 
        [data-testid="stFileUploaderFile"] {
            background-color: #ffffff !important;
            border: 1px solid #d8dbe2 !important;
            border-radius: 0.5rem !important;
        }
 
        [data-testid="stFileUploaderDropzone"] button {
            background-color: #ffffff !important;
            color: #20242b !important;
            border: 1px solid #cfd3da !important;
        }
 
        [data-testid="stFileUploaderDropzone"] button * {
            color: #20242b !important;
            -webkit-text-fill-color: #20242b !important;
            opacity: 1 !important;
        }
 
        [data-testid="stFileUploader"] svg,
        [data-testid="stFileUploader"] button svg {
            color: #4b5563 !important;
            fill: #4b5563 !important;
            opacity: 1 !important;
        }
 
 
        /* =========================================================
           BUTTONS (st.button and st.download_button)
           Red button with white text, readable without hover.
        ========================================================= */
 
        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button {
            background-color: #ef4b4b !important;
            color: #ffffff !important;
            border: 1px solid #ef4b4b !important;
            border-radius: 0.5rem !important;
            font-weight: 600 !important;
        }
 
        [data-testid="stButton"] button p,
        [data-testid="stButton"] button span,
        [data-testid="stDownloadButton"] button p,
        [data-testid="stDownloadButton"] button span {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            opacity: 1 !important;
        }
 
        [data-testid="stButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover {
            background-color: #c93b3b !important;
            border-color: #c93b3b !important;
        }
 
        [data-testid="stButton"] button:hover p,
        [data-testid="stButton"] button:hover span,
        [data-testid="stDownloadButton"] button:hover p,
        [data-testid="stDownloadButton"] button:hover span {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
 
        </style>
        """,
        unsafe_allow_html=True,
    )
