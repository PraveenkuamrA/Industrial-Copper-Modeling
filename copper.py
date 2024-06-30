import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


st.set_page_config(layout="wide")
st.title('Industrial Copper Modeling Application')


tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"])

with tab1:
    # Define the possible values for the dropdown menus
    status_options = ['Won','Draft','To be approved','Lost','Not lost for AM','Wonderful','Revised','Offered','Offerable']
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28.,25.,30.,32.,38.,78.,27.,77.,113.,79.,26.,39.,40.,84.,80.,107.,89.,]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67.,
                           79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product = [1670798778, 1668701718,628377,640665,611993,1668701376,164141591,1671863738,1332077137,640405,
       1693867550,1665572374,1282007633,1668701698,628117,1690738206,628112,640400,1671876026,164336407,164337175, 
       1668701725,1665572032,611728,1721130331,1693867563,611733,1690738219,1722207579,929423819,
       1665584320, 1665584662, 1665584642]
    
        # Define the widgets for user input
    with st.form("my_form"):
        col1, col2, col3 = st.columns([5, 2, 5])
        with col1:
            st.write(' ')
            Status = st.selectbox("Status", status_options, key=1)
            Item_Type = st.selectbox("Item Type", item_type_options, key=2)
            Country = st.selectbox("Country", sorted(country_options), key=3)
            Application = st.selectbox("Application", sorted(application_options), key=4)
            Product_ref = st.selectbox("Product Reference", product, key=5)
            Delivery_Date = st.text_input("Enter the Delivery Date(Min:20190401 & Max:30310101 / YYYYMMDD)")
        with col3:
            st.write(
                f'<h5 style="color:#ee4647;">NOTE: Min & Max given for reference, you can enter any value</h5>',
                unsafe_allow_html=True)
            Item_Date = st.text_input("Enter the Date (Min:19950101 & Max:20210401/YYYYMMDD)")
            Quantity_tons = st.text_input("Enter Quantity Tons (Min:1 & Max:1000000000)")
            Thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            Width = st.text_input("Enter width (Min:1, Max:2990)")
            Customer = st.text_input("Customer ID (Min:12458, Max:30408185)")
            submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
            st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #004aad;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)
            
    if submit_button:

        with open(r"C:\Users\user\Downloads\copper_selling_price.pkl", 'rb') as file:
            guvi = pickle.load(file)

        with open(r"C:\Users\user\Downloads\preprocessor.pkl", 'rb') as f:
            preprocessor = pickle.load(f)

        user_input=pd.DataFrame({
            'item_date':[Item_Date],
            'quantity tons':[Quantity_tons],
            'customer':[Customer],
            'country':[Country],
            'status':[Status],
            'item type':[Item_Type],
            'application':[Application],
            'thickness':[Thickness],
            'width':[Width],
            'product_ref':[Product_ref],
            'delivery date':[Delivery_Date]
            })
        try: 
            transformed_input = preprocessor.transform(user_input)
            # Make predictions using the best_model
            prediction = guvi.predict(transformed_input)

            st.write('## :orange[Predicted selling price:] ',prediction[0])
        except: 
            st.write(':orange[You have entered an invalid value]')
with tab2:
    with st.form("my_form1"):
        col1, col2, col3 = st.columns([5, 1, 5])
        with col1:
            cquantity_tons = st.text_input("Enter Quantity Tons (Min:1 & Max:10000000)")
            cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            cwidth = st.text_input("Enter width (Min:1, Max:2990)")
            ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
            cselling = st.text_input("Selling Price (Min:1, Max:100001015)")

        with col3:
            st.write(' ')
            citem_type = st.selectbox("Item Type", item_type_options, key=21)
            ccountry = st.selectbox("Country", sorted(country_options), key=31)
            capplication = st.selectbox("Application", sorted(application_options), key=41)
            cproduct_ref = st.selectbox("Product Reference", product, key=51)
            csubmit_button = st.form_submit_button(label="PREDICT STATUS")
    
    if csubmit_button: 

        with open(r"C:\Users\user\Downloads\copper_status.pkl", 'rb') as file:
            model = pickle.load(file)

        user_data = pd.DataFrame({
            'quantity tons': [cquantity_tons],
            'customer': [ccustomer],
            'country': [ccountry],
            'application': [capplication],
            'thickness': [cthickness],
            'width': [cwidth],
            'product_ref': [cproduct_ref],
            'selling_price': [cselling],
            'item type': [citem_type]
            })
        
        try:
            prediction = model.predict(user_data)
            st.write('## :orange[Predicted selling price:] ',prediction[0])
        except:
            st.write(':orange[You have entered an invalid value]')

