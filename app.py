# 1. import and page configuration
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go


st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="🏦",
    layout="wide"         
)
# 2. load pickle files with cached
@st.cache_resource
def load_pickle_files():
    base = "03_models/"
    with open(base + "median_imputer.pkl", "rb") as f:
        median_imputer = pickle.load(f)
    with open(base + "binner.pkl", "rb") as f:
        binner = pickle.load(f)
    with open(base + "woe_encoder.pkl", "rb") as f:
        woe_encoder = pickle.load(f)
    with open(base + "logreg_model.pkl", "rb") as f:
        logreg = pickle.load(f)
    with open(base + "config.pkl", "rb") as f:
        config = pickle.load(f)
    return median_imputer, binner, woe_encoder, logreg, config

median_imputer, binner, woe_encoder, logreg, config = load_pickle_files()

# 3. Predication function
    # median imputer -> binining -> convert to cateogries all cols -> woe
def preprocess_input(user_df):
    df = user_df.copy()
    
    # Patch: median_imputer was fit with 2 extra missing-indicator columns
    # that this app's form never produces (user always fills every field,
    # so these would always be 0 in a live prediction anyway)
    for na_col in ['employment_years_na', 'interest_rate_na']:
        if na_col not in df.columns:
            df[na_col] = 0
    
    df = median_imputer.transform(df)
    
    # Drop the indicator columns again after imputation — binner/woe_encoder
    # were fit WITHOUT them (confirm this matches your config['all_encode_columns'])
    df = df.drop(columns=['employment_years_na', 'interest_rate_na'], errors='ignore')
    
    df = binner.transform(df)
    for col in config['all_encode_columns']:
        df[col] = df[col].astype('category')
    df = woe_encoder.transform(df)
    return df

def predict_probability(user_df):
    df_t = preprocess_input(user_df)
    return logreg.predict_proba(df_t[config['all_encode_columns']])[:, 1][0]

def pd_to_score(pd_value):
    pd_value = np.clip(pd_value, 1e-6, 1 - 1e-6)
    odds = (1 - pd_value) / pd_value
    return config['score_offset'] + config['score_factor'] * np.log(odds)

def get_decision(pd_value):
    if pd_value <= config['approve_threshold']:
        return "APPROVE"
    elif pd_value >= config['reject_threshold']:
        return "REJECT"
    else:
        return "REVIEW"


def get_risk_tier(pd_value):
    if pd_value < 0.15:
        return "LOW"
    elif pd_value < 0.40:
        return "MEDIUM"
    else:
        return "HIGH"
    
def get_reason_codes(user_df):
    df_t = preprocess_input(user_df)
    coefs = dict(zip(config['all_encode_columns'], logreg.coef_[0]))
    contributions = {f: df_t[f].values[0] * coefs[f] for f in config['all_encode_columns']}
    return pd.DataFrame(sorted(contributions.items(), key=lambda x: x[1]),
                        columns=['feature', 'contribution'])
    
    
# 4. app Header + disclaimer
st.title("🏦 Credit Risk Application Scorecard")
st.caption(
    "This model is trained on a public, simulated dataset (Kaggle: Credit Risk "
    "Dataset by Lao Tse) for demonstration purposes only. Not trained on real "
    "bank data — not for actual credit decisions."
)

tab1, tab2 = st.tabs(["🎯 Predict & Explain", "📊 Model Dashboard"])
#tab1 = st.tabs(["🎯 Predict & Explain"])

with tab1:
    st.subheader("Applicant Information")
    
    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Demographics**")
            age = st.number_input("Age" ,18,100,25)
            home_ownership = st.selectbox("Home Ownership",config['dropdown_options']['home_ownership'])
            employment_years = st.number_input('Employment Lenght(years)',0,50,3)
            
        with col2:
            st.markdown('**Finacial & loan**')
            annual_income = st.slider('Annual Income (₹)',0,1000000,500000)
            loan_amount = st.slider('Loan Amount Requested (₹)',0,5000000,200000)
            loan_purpose = st.selectbox("loan Purpose",config['dropdown_options']['loan_purpose'])
            
        with col3:
            st.markdown('Credit Burea')
            credit_grade = st.selectbox('Credit Grade',config['dropdown_options']['credit_grade'])
            prior_default_flag = st.radio("Prior Default on File?", config['dropdown_options']['prior_default_flag'])
            credit_history_years = st.number_input("Credit History (years)", 0, 40, 5)
            
        submitted = st.form_submit_button("Run Credit Risk Assessment",use_container_width=True )
        
# 5. validation
    if submitted:
        # Derive interest_rate from credit_grade (user doesn't know their rate)
        interest_rate = config['grade_rate_map'][credit_grade]
        
        # Compute engineered feature
        loan_income_ratio = loan_amount / annual_income if annual_income > 0 else 0
        
        # Validation
        if annual_income == 0:
            st.error("Annual income must be greater than 0.")
            st.stop()
        
        # Assemble into exact column structure the pipeline expects
        user_input = pd.DataFrame([{
            'age': age, 'annual_income': annual_income, 'employment_years': employment_years,
            'loan_amount': loan_amount, 'interest_rate': interest_rate,
            'loan_income_ratio': loan_income_ratio, 'credit_history_years': credit_history_years,
            'home_ownership': home_ownership, 'loan_purpose': loan_purpose,
            'credit_grade': credit_grade, 'prior_default_flag': prior_default_flag,
        }])
        
        # Run prediction
        pd_value = predict_probability(user_input)
        score = int(pd_to_score(pd_value))
        decision = get_decision(pd_value)
        risk_tier = get_risk_tier(pd_value)
        reason_codes = get_reason_codes(user_input)
    
    

        st.divider()
        st.subheader("Model Decision")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric('Risk Decesion',decision)
        t2.metric("Risk Tier",risk_tier)
        t3.metric('Probability of Default',f"{pd_value : .0%}")
        t4.metric("Credit Score",score)



        g1, g2 = st.columns(2)
        
        with g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pd_value * 100,
                title={'text': "Default Risk (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "crimson" if pd_value > 0.4 else "orange" if pd_value > 0.15 else "seagreen"},
                    'steps': [
                        {'range': [0, 15], 'color': "#d1e7dd"},
                        {'range': [15, 40], 'color': "#fff3cd"},
                        {'range': [40, 100], 'color': "#f8d7da"},
                    ],
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with g2:
            colors = ['crimson' if v < 0 else 'seagreen' for v in reason_codes['contribution']]
            fig_reason = go.Figure(go.Bar(
                x=reason_codes['contribution'], y=reason_codes['feature'],
                orientation='h', marker_color=colors
            ))
            fig_reason.update_layout(title="Why This Decision — Feature Contributions",
                                        xaxis_title="Contribution to Risk (log-odds)")
            st.plotly_chart(fig_reason, use_container_width=True)
        
        # Plain-language summary
        top_risk_factor = reason_codes[reason_codes['contribution'] < 0].sort_values('contribution').head(1)
        if len(top_risk_factor) > 0 and decision != "APPROVE":
            st.info(f"The strongest factor increasing this applicant's risk was **{top_risk_factor.iloc[0]['feature']}**.")
            
            
# with tab2:
#     st.subheader("Model Performance & Feature Importance")
    
#     # You'd load a separately-pickled evaluation artifact here, e.g.:
#     # with open("03_models/evaluation_metrics.pkl", "rb") as f:
#     #     eval_data = pickle.load(f)
    
#     st.markdown("**IV Ranking, AUC, KS Chart, Confusion Matrix go here** — "
#                 "these use your pre-computed test-set results, not live predictions.")
    
#     # Example: static image or pre-generated plotly figure loaded from disk
#     # st.plotly_chart(eval_data['iv_ranking_fig'])
#     # st.plotly_chart(eval_data['ks_chart_fig'])