#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#----------------------------------------
from PIL import Image
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
#----------------------------------------
import scipy.stats as stats
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Statistics Playground | v0.1",
                    layout="wide",
                    page_icon="📊",            
                    initial_sidebar_state="auto")
#---------------------------------------
st.markdown(
    """
    <style>
    .title-large {
        text-align: center;
        font-size: 35px;
        font-weight: bold;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .title-small {
        text-align: center;
        font-size: 20px;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    <div class="title-large">Statistics Playground</div>
    <div class="title-small">Play with Data</div>
    """,
    unsafe_allow_html=True
)
#----------------------------------------

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F0F2F6;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
        z-index: 100;
    }
    .footer p {
        margin: 0;
    }
    .footer .highlight {
        font-weight: bold;
        color: blue;
    }
    </style>

    <div class="footer">
        <p>© 2025 | Created by : <span class="highlight">Avijit Chakraborty</span> | Prepared by: <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a></p> <span class="highlight">Thank you for visiting the app | Unauthorized uses or copying is strictly prohibited | For best view of the app, please zoom out the browser to 75%.</span>
    </div>
    """,
    unsafe_allow_html=True)

#----------------------------------------
#st.divider()
#----------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------
### knowledge 
#---------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

@st.cache_data(ttl="2h")
def load_file(file):
    file_extension = file.name.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(file, sep=None, engine='python', encoding='utf-8', parse_dates=True, infer_datetime_format=True)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format")
        df = pd.DataFrame()
    return df

#---------------------------------------------------------------------------------------------------------------------------------
### Main app
#---------------------------------------------------------------------------------------------------------------------------------

if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

col1, col2 = st.columns(2)
with col1:
    if st.button("🏠 **Home**",use_container_width=True):
        st.session_state.current_page = "home"
with col2:
    if st.button("📈 **Analysis**",use_container_width=True):
        st.session_state.current_page = "analysis" 
        
page = st.session_state.current_page

#---------------------------------------------------------------------------------------------------------------------------------

if page == "analysis":
    
    st.markdown(
            """
            <style>
                .centered-info {
                display: flex;
                justify-content: center;
                align-items: center;
                font-weight: bold;
                font-size: 15px;
                color: #007BFF; 
                padding: 5px;
                background-color: #E8F4FF; 
                border-radius: 5px;
                border: 1px solid #007BFF;
                margin-top: 5px;
                }
            </style>
            """,unsafe_allow_html=True,)
    st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Filters</span></div>',unsafe_allow_html=True,)
    #---------------------------------------------------------------
    file = st.sidebar.file_uploader("**:blue[Choose a file]**",type=["csv", "xls", "xlsx"], accept_multiple_files=False, key="file_upload")
    st.sidebar.divider()
    if file is not None:
        df = load_file(file)        #for filter
        df1 = df.copy()             #for analysis
        
        def initialize_session_state():
            if 'selected_columns' not in st.session_state:
                st.session_state.selected_columns = df.columns.tolist()
            if 'numeric_filters' not in st.session_state:
                st.session_state.numeric_filters = {}
            if 'filter_mode' not in st.session_state:
                st.session_state.filter_mode = {}
            if 'categorical_filters' not in st.session_state:
                st.session_state.categorical_filters = {}
                
        initialize_session_state()
        
        if st.sidebar.button("Reset Filters"):
            st.session_state.clear()
            initialize_session_state()
        #---------------------------------------------------------------
        col1, col2 = st.columns((0.8,0.2))
        with col1:
            with st.container(border=True):
                selected_columns = st.multiselect("**:blue[Columns to display]**", options=df.columns, default=st.session_state.selected_columns, key="selected_columns") 
        with col2:   
            with st.container(border=True):
                subset_option = st.selectbox("**:blue[Subset Data Options]**", ["No Subset", "Enable Subset"])
        #---------------------------------------------------------------
        selected_df = df[selected_columns].copy()
        #---------------------------------------------------------------
        if subset_option == "Enable Subset":
            #st.header("Data Subset Filters")
            with st.popover("**:blue[Subset Filters]**", disabled=False, use_container_width=True):

                for col in selected_df.columns:
                    if pd.api.types.is_numeric_dtype(selected_df[col]):

                        st.write(f"#### Filter for Numeric Column: {col}")
                        filter_mode = st.selectbox(f"Filter type for {col}", ["Range", "Single Value"], key=f"filter_mode_{col}")
                        st.session_state.filter_mode[col] = filter_mode

                        if filter_mode == "Range":
                            from_col, to_col = st.columns(2)
                            with from_col:
                                from_value = st.number_input(f"From", key=f"from_{col}", value=float(selected_df[col].min()) if not pd.isna(selected_df[col].min()) else 0.0)
                            with to_col:
                                to_value = st.number_input(f"To", key=f"to_{col}", value=float(selected_df[col].max()) if not pd.isna(selected_df[col].max()) else 1.0)
                        
                            selected_ddf = selected_df[(selected_df[col] >= from_value) & (selected_df[col] <= to_value)]

                        elif filter_mode == "Single Value":
                            operator_col, value_col = st.columns(2)
                            with operator_col:
                                operator = st.selectbox(f"Operator", ["<", ">", "="], key=f"operator_{col}")
                            with value_col:
                                single_val = st.number_input(f"Value", key=f"value_{col}", value=float(selected_df[col].median()) if not pd.isna(selected_df[col].median()) else 0.0)

                            if operator == "<":
                                selected_data = selected_data[selected_data[col] < single_val]
                        elif operator == ">":
                            selected_data = selected_data[selected_data[col] > single_val]
                        elif operator == "=":
                            selected_data = selected_data[selected_data[col] == single_val]

                    elif pd.api.types.is_object_dtype(selected_df[col]) or pd.api.types.is_categorical_dtype(selected_df[col]):
                    
                        st.write(f"#### Filter for Categorical Column: {col}")
                        unique_vals = selected_df[col].dropna().unique()
                        selected_vals = st.multiselect(f"Select values for {col}:", options=unique_vals, default=unique_vals, key=f"filter_{col}")
                        st.session_state.categorical_filters[col] = selected_vals
                        selected_df = selected_df[selected_df[col].isin(selected_vals)]
        #---------------------------------------------------------------                
        stats_expander = st.expander("**:blue[Preview]**", expanded=True)
        with stats_expander: 
            st.dataframe(selected_df.head(2),use_container_width=True)
        #-----------------------------------------------------------------------------------------------------------------------------------------------    
        st.markdown(
            """
            <style>
                .centered-info {
                display: flex;
                justify-content: center;
                align-items: center;
                font-weight: bold;
                font-size: 15px;
                color: #007BFF; 
                padding: 5px;
                background-color: #E8F4FF; 
                border-radius: 5px;
                border: 1px solid #007BFF;
                margin-top: 5px;
                }
            </style>
            """,unsafe_allow_html=True,)
        st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Summary Statistics</span></div>',unsafe_allow_html=True,)
        #--------------------------------------------------------------- 
        colSUM, colType = st.columns(2)
        with colSUM:
            
            described_df = selected_df.describe().T
            st.dataframe(described_df, height=250, use_container_width=True)
            
        with colType:
            
            df_info = pd.DataFrame({"Column Name": selected_df.columns,
                                      "Data Type": selected_df.dtypes,
                                      "NA Count": selected_df.isna().sum()}).reset_index(drop=True)
            st.dataframe(df_info, height=250, use_container_width=True)
        #-----------------------------------------------------------------------------------------------------------------------------------------------    
        st.markdown(
            """
            <style>
                .centered-info {
                display: flex;
                justify-content: center;
                align-items: center;
                font-weight: bold;
                font-size: 15px;
                color: #007BFF; 
                padding: 5px;
                background-color: #E8F4FF; 
                border-radius: 5px;
                border: 1px solid #007BFF;
                margin-top: 5px;
                }
            </style>
            """,unsafe_allow_html=True,)
        st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Proportions</span></div>',unsafe_allow_html=True,)
        #--------------------------------------------------------------- 
        colProp, colcatPropTable,colnumPropTable = st.columns((0.15,0.25,0.6))
        with colProp:
            with st.container(border=True):
                
                    cat_var1 = st.selectbox("**:blue[First categorical variable]**", ["None"] + list(selected_df.select_dtypes(include=['object', 'category']).columns))
                    cat_var2 = st.selectbox("**:blue[Second categorical variable]**", ["None"] + [col for col in selected_df.select_dtypes(include=['object', 'category']).columns if col != cat_var1])
                    weight_column = st.selectbox("**:blue[Choose a weight column (optional)]**", ["None"] + list(selected_df.select_dtypes(include=['int64', 'float64']).columns))
                    normalize_option = st.selectbox("**:blue[Normalize by]**", ["Total", "Row", "Column"], index=0)
                    proportion_table = None

                    if cat_var1 != "None":
                        if cat_var2 == "None":

                            if weight_column != "None":
                                weighted_counts = selected_df.groupby(cat_var1)[weight_column].sum()
                                if normalize_option == "Total":
                                    proportion_table = weighted_counts / weighted_counts.sum()
                                else:
                                    st.warning("Row and Column normalization not applicable for single variable.")
                            else:
                                proportion_table = selected_df[cat_var1].value_counts(normalize=True)
                        else:
                        
                            if weight_column != "None":
                                weighted_counts = pd.crosstab(selected_df[cat_var1],
                                                          selected_df[cat_var2],
                                                          values=selected_df[weight_column],
                                                          aggfunc='sum')
                                if normalize_option == "Total":
                                    proportion_table = weighted_counts / weighted_counts.sum().sum()
                                elif normalize_option == "Row":
                                    proportion_table = weighted_counts.div(weighted_counts.sum(axis=1), axis=0)
                                elif normalize_option == "Column":
                                    proportion_table = weighted_counts.div(weighted_counts.sum(axis=0), axis=1)
                            else:
                        
                                normalize_map = {"Total": "all", "Row": "index", "Column": "columns"}
                                normalize_arg = normalize_map[normalize_option]
                                proportion_table = pd.crosstab(selected_df[cat_var1], selected_df[cat_var2], normalize=normalize_arg)

            with colcatPropTable: 
                with st.container(border=True):   
                    
                    if proportion_table is not None:
                        st.dataframe(proportion_table)
                    else:
                        st.warning("Please select at least one categorical variable to generate the proportion table.")
                        
            with colnumPropTable: 
                with st.container(border=True):   
                    
                    selected_df_num = selected_df.select_dtypes(include=['int64', 'float64'])
                    if not selected_df_num.empty:
                        fig, ax = plt.subplots(figsize=(14,2.5))
                        sns.heatmap(selected_df_num.corr(), ax=ax, annot=True, linewidths=0.05, fmt='.2f', cmap="magma")
                        st.pyplot(fig)
                    else:
                        st.warning("No numerical columns found in the dataset.")
                        
        #-----------------------------------------------------------------------------------------------------------------------------------------------    
        st.markdown(
            """
            <style>
                .centered-info {
                display: flex;
                justify-content: center;
                align-items: center;
                font-weight: bold;
                font-size: 15px;
                color: #007BFF; 
                padding: 5px;
                background-color: #E8F4FF; 
                border-radius: 5px;
                border: 1px solid #007BFF;
                margin-top: 5px;
                }
            </style>
            """,unsafe_allow_html=True,)
        st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Hypothesis Testing</span></div>',unsafe_allow_html=True,)
        #--------------------------------------------------------------- 
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["**T-Test**","**Chi-Square Test**","**Anova (F Statistics)**","**Mann-Whitney Test (U Statistics)**","**Kruskal-Wallis Test (h Statistics)**"])   
        #---------------------------------------------------------------         
        with tab1:
            
            st.info("Select one numeric variable and one categorical variable to perform the T-Test.")
            col1, col2= st.columns((0.2,0.8))
            with col1:
                with st.container(border=True):
                        
                    numeric_var = st.selectbox("Select numeric variable", ["None"] + list(selected_df.select_dtypes(include=['int64', 'float64']).columns))
                    categorical_var = st.selectbox("Select categorical variable", ["None"] + list(selected_df.select_dtypes(include=['object', 'category']).columns))
                    tail_option = st.selectbox("Select test type", ["Two-Tailed", "One-Tailed"])

                    with col2:
                        with st.container(border=True):
                    
                            if numeric_var != "None" and categorical_var != "None":
                                unique_groups = selected_df[categorical_var].dropna().unique()
                                if len(unique_groups) == 2:
                                    group1 = selected_df[selected_df[categorical_var] == unique_groups[0]][numeric_var]
                                    group2 = selected_df[selected_df[categorical_var] == unique_groups[1]][numeric_var]

                                    t_stat, p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
                                    if tail_option == "One-Tailed":
                                        p_value /= 2
                                    if t_stat < 0:
                                        p_value = 1 - p_value
                                
                                    st.write("### T-Test Results")
                                    st.write(f"T-Statistic: {t_stat}")
                                    st.write(f"P-Value: {p_value}")

                                    if p_value < 0.05:
                                        st.write(f"**Interpretation**: The means of `{numeric_var}` differ significantly between the groups in `{categorical_var}` (at the 5% significance level).")
                                    else:
                                        st.write(f"**Interpretation**: There is no significant difference in the means of `{numeric_var}` between the groups in `{categorical_var}`.")
                            else:
                                st.warning("Please select a categorical variable with exactly two unique groups for a two-sample T-Test.")
        #---------------------------------------------------------------         
        with tab2:
            
            st.info("Select two or more categorical variables to perform pairwise Chi-Square Tests.")
            col1, col2= st.columns((0.2,0.8))
            with col1:
                with st.container(border=True):
                    
                    cat_vars = selected_df.select_dtypes(include=['object', 'category']).columns
                    selected_vars = st.multiselect("Select categorical variables for Chi-Square Test", cat_vars)
                    
                    with col2:
                        with st.container(border=True):
                            
                            if len(selected_vars) >= 2:
                                for i in range(len(selected_vars)):
                                    for j in range(i + 1, len(selected_vars)):
                                        var1, var2 = selected_vars[i], selected_vars[j]
                                        
                                        subcol1, subcol2= st.columns(2)
                                        with subcol1:
                                            
                                            st.markdown(f"#### Chi-Square Test between `{var1}` and `{var2}`")
                                            contingency_table = pd.crosstab(selected_df[var1], selected_df[var2])
                                            st.markdown("Contingency Table:")
                                            st.dataframe(contingency_table)
                                        
                                        with subcol2:
                                        
                                            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

                                            st.write("#### Chi-Square Test Results")
                                            st.write(f"Chi-Square Statistic: {chi2}")
                                            st.write(f"Degrees of Freedom: {dof}")
                                            st.write(f"P-Value: {p}")
                                            st.divider()
                                            if p < 0.05:
                                                st.write("**Interpretation**: There is a significant association between the variables.")
                                            else:
                                                st.write("**Interpretation**: There is no significant association between the variables.")
        #---------------------------------------------------------------         
        with tab3:
            
            st.info("Select a numeric variable and a categorical variable with three or more groups.")
            col1, col2= st.columns((0.2,0.8))
            with col1:
                with st.container(border=True):
                    
                    numeric_var = st.selectbox("Select numeric variable for ANOVA", ["None"] + list(selected_df.select_dtypes(include=['int64', 'float64']).columns))
                    categorical_var = st.selectbox("Select categorical variable for ANOVA", ["None"] + list(selected_df.select_dtypes(include=['object', 'category']).columns))

                    with col2:
                        with st.container(border=True):
                            
                            if numeric_var != "None" and categorical_var != "None":
                                groups = [group[numeric_var].dropna() for name, group in selected_df.groupby(categorical_var)]
            
                                if len(groups) >= 3:
                                    f_statistic, p_value = stats.f_oneway(*groups)
                                    st.write("#### ANOVA Test Results")
                                    st.write(f"F-Statistic: {f_statistic}")
                                    st.write(f"P-Value: {p_value}")
                                    st.divider()
                                    if p_value < 0.05:
                                        st.write(f"**Interpretation**: There is a significant difference in `{numeric_var}` between the groups in `{categorical_var}`.")
                                    else:
                                        st.write(f"**Interpretation**: No significant difference in `{numeric_var}` across groups in `{categorical_var}`.")
                            else:
                                st.warning("Please select a categorical variable with three or more unique groups.")
        #---------------------------------------------------------------         
        with tab4:
            
            st.info("Select a numeric variable and a categorical variable with exactly two groups.")
            col1, col2= st.columns((0.2,0.8))
            with col1:
                with st.container(border=True):
                    
                    numeric_var_mw = st.selectbox("Select numeric variable for Mann-Whitney", ["None"] + list(selected_df.select_dtypes(include=['int64', 'float64']).columns),key="numeric_var_mw")
                    categorical_var_mw = st.selectbox("Select categorical variable for Mann-Whitney", ["None"] + list(selected_df.select_dtypes(include=['object', 'category']).columns),key="categorical_var_mw")

                    with col2:
                        with st.container(border=True):
                            
                            if numeric_var_mw != "None" and categorical_var_mw != "None":
                                unique_groups = selected_df[categorical_var_mw].dropna().unique()
                                
                                if len(unique_groups) == 2:
                                    group1 = selected_data[selected_data[categorical_var_mw] == unique_groups[0]][numeric_var_mw]
                                    group2 = selected_data[selected_data[categorical_var_mw] == unique_groups[1]][numeric_var_mw]

                                    u_statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                                    st.write("### Mann-Whitney U Test Results")
                                    st.write(f"U-Statistic: {u_statistic}")
                                    st.write(f"P-Value: {p_value}")

                                    if p_value < 0.05:
                                        st.write(f"**Interpretation**: The distribution of `{numeric_var_mw}` differs significantly between the two groups in `{categorical_var_mw}`.")
                                    else:
                                        st.write(f"**Interpretation**: No significant difference in `{numeric_var}` distribution between the groups in `{categorical_var_mw}`.")
                            else:
                                st.warning("Please select a categorical variable with exactly two unique groups.")
                                
        #---------------------------------------------------------------         
        with tab5:
            
            st.info("Select a numeric variable and a categorical variable with three or more groups.")
            col1, col2= st.columns((0.2,0.8))
            with col1:
                with st.container(border=True):
                    
                    numeric_var= st.selectbox("Select numeric variable for Mann-Whitney", ["None"] + list(selected_df.select_dtypes(include=['int64', 'float64']).columns),key="numeric_var_kw")
                    categorical_var = st.selectbox("Select categorical variable for Mann-Whitney", ["None"] + list(selected_df.select_dtypes(include=['object', 'category']).columns),key="categorical_var_kw")

                    with col2:
                        with st.container(border=True):
                            
                            if numeric_var != "None" and categorical_var != "None":
                                groups = [group[numeric_var].dropna() for name, group in selected_data.groupby(categorical_var)]
            
                                if len(groups) >= 3:
                                    h_statistic, p_value = stats.kruskal(*groups)
                                    st.write("#### Kruskal-Wallis Test Results")
                                    st.write(f"H-Statistic: {h_statistic}")
                                    st.write(f"P-Value: {p_value}")

                                    if p_value < 0.05:
                                        st.write(f"**Interpretation**: The distribution of `{numeric_var}` differs significantly across groups in `{categorical_var}`.")
                                    else:
                                        st.write(f"**Interpretation**: No significant difference in `{numeric_var}` distribution across groups in `{categorical_var}`.")
                            else:
                                st.warning("Please select a categorical variable with three or more unique groups.")
