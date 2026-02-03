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
from io import BytesIO, StringIO
from PIL import Image
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
#----------------------------------------
import scipy.stats as stats
from scipy.stats import gaussian_kde
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Statistics Playground | v0.2",
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
    <div class="title-small">Play with Data | v0.2</div>
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
        <p>© 2026 | Created by : <span class="highlight">Avijit Chakraborty</span> | <a href="mailto:avijit.mba18@gmail.com"> 📩 </a>  <span class="highlight">Thank you for visiting the app | Unauthorized uses or copying is strictly prohibited | For best view of the app, please zoom out the browser to 75%.</span> </p>
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

@st.cache_data(ttl="2h")
def get_plotly_download(fig, file_format="png", scale=3):
    if file_format == "png":
        buffer = BytesIO()
        fig.write_image(buffer, format="png", scale=scale)
        buffer.seek(0)
    elif file_format == "html":
        html_str = StringIO()
        fig.write_html(html_str)
        buffer = BytesIO(html_str.getvalue().encode("utf-8"))
        buffer.seek(0)
    return buffer

@st.cache_data(ttl="2h")
def plot_histograms_with_kde(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_columns) == 0:
        st.warning("No numerical columns found in the dataset to plot.")
        return
    for col in numerical_columns:
        with st.container():
            fig, ax = plt.subplots(figsize=(25,5))
            data = df[col].dropna()
            ax.hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.6, density=True)
            if len(data.unique()) > 1 and data.var() > 0:
                kde = gaussian_kde(data)
                x_vals = np.linspace(data.min(), data.max(), 1000)
                ax.plot(x_vals, kde(x_vals), color='red', lw=2, label='KDE')
            else:
                st.warning(f"KDE could not be computed for column '{col}' due to low variance.")
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig, use_container_width=True)
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

if page == "home":
    
    col1, col2 = st.columns((0.55,0.45))
    with col1: 
   
        st.markdown("""
        <div style="background-color: #F9F9FB; padding: 10px; border-radius: 8px; margin-top: 20px; box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);">    
            <h5 style="color: #0056b3; font-weight: bold;">Description</h5>
            <ul style="color: #333333; font-size: 24px,padding-left: 15px; margin: 10px 0;">
                <li>Statistics Playground provides an intuitive, user-friendly interface for comprehensive statistical analysis and visualization.</li>
                <li>Designed to simplify complex data analysis tasks, it empowers researchers and social scientists to explore, visualize, and 
                interpret data effortlessly without requiring programming skills.</li>
                <li>Supporting various statistical tests, including T-tests, Chi-Square, ANOVA, and correlation analysis</li>
                <li>Statistics Playground enables users to investigate relationships within datasets.</li>
                <li>Its customizable visualizations, proportion tables, and weighted calculations make it ideal for examining survey responses, 
                demographic distributions, and experimental results.</li>
                <li> With flexible data uploads and export options, Statistics Playground ensures a seamless analytical experience, 
                providing valuable insights for data-driven decision-making.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:  
              
        st.markdown("""
        <div style="background-color: #F9F9FB; padding: 10px; border-radius: 8px; margin-top: 20px; box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);">    
            <h5 style="color: #0056b3; font-weight: bold;">Application</h5>
            <ul style="color: #333333; font-size: 24px,padding-left: 15px; margin: 10px 0;">
                <li><b>🔍 Data Subsetting & Filtering:</b> Generate proportion tables with optional weighted calculations for accurate survey data interpretation.</li>
                <li><b>📊 Proportion Tables & Weighted Analysis:</b> Perform T-tests, Chi-Square, ANOVA, and more to analyze relationships within your data.</li>
                <li><b>📈 Comprehensive Statistical Tests:</b> Gain detailed insights through summary statistics and correlation matrices.</li>
                <li><b>📉 Descriptive Statistics & Correlation Analysis:</b> Create histograms, scatter plots, line plots, regression plots, and box plots with customizable options.</li>
                <li><b>🔢 Aggregation Functions:</b> Apply sum, mean, count, and other aggregation functions to streamline data summaries.</li>
                <li><b>📥 High-Quality Export Options:</b> Download visualizations in high-resolution PNG or interactive HTML formats for reporting and sharing.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    st.warning("Please **click** the **Analysis** button above for further details", icon="🚨")         
#---------------------------------------------------------------------------------------------------------------------------------

if page == "analysis":
    
    st.sidebar.subheader("**:blue[Contents]**",divider='blue')
    file = st.sidebar.file_uploader("**:blue[Choose a file]**",type=["csv", "xls", "xlsx"], accept_multiple_files=False, key="file_upload")
    st.sidebar.divider()
    if file is not None:
        df = load_file(file)        #for filter
        df1 = df.copy()             #for analysis
        #---------------------------------------------------------------        
        st.sidebar.markdown(
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
        st.sidebar.markdown('<div class="centered-info"><span style="margin-left: 10px;">Filters</span></div>',unsafe_allow_html=True,)
        #---------------------------------------------------------------
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
        selected_columns = st.sidebar.multiselect("**:blue[Columns to display]**", options=df.columns, default=st.session_state.selected_columns, key="selected_columns") 
        subset_option = st.sidebar.selectbox("**:blue[Subset Data Options]**", ["No Subset", "Enable Subset"])
        #---------------------------------------------------------------
        selected_df = df[selected_columns].copy()
        #---------------------------------------------------------------
        if subset_option == "Enable Subset":
            #st.header("Data Subset Filters")
            with st.popover("**:blue[Subset Filters]**", disabled=False, use_container_width=True):

                for col in selected_df.columns:
                    if pd.api.types.is_numeric_dtype(selected_df[col]):

                        st.write(f"##### Filter for Numeric Column: {col}")
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
                                selected_df = selected_df[selected_df[col] < single_val]
                            elif operator == ">":
                                selected_df = selected_df[selected_df[col] > single_val]
                            elif operator == "=":
                                selected_df = selected_df[selected_df[col] == single_val]
                        st.write('--')
                    elif pd.api.types.is_object_dtype(selected_df[col]) or pd.api.types.is_categorical_dtype(selected_df[col]):
                    
                        st.write(f"##### Filter for Categorical Column: {col}")
                        unique_vals = selected_df[col].dropna().unique()
                        selected_vals = st.multiselect(f"Select values for {col}:", options=unique_vals, default=unique_vals, key=f"filter_{col}")
                        st.session_state.categorical_filters[col] = selected_vals
                        selected_df = selected_df[selected_df[col].isin(selected_vals)]
        #---------------------------------------------------------------  
        st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Preview</span></div>',unsafe_allow_html=True,)  
        #st.markdown("")
        #---------------------------------------------------------------             
        #stats_expander = st.expander("", expanded=True)
        #with stats_expander: 
        with st.container(border=True):
            
            st.dataframe(selected_df.head(3),use_container_width=True)
        #-----------------------------------------------------------------------------------------------------------------------------------------------    
        st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Summary Statistics</span></div>',unsafe_allow_html=True,)
        #---------------------------------------------------------------
        with st.container(border=True):   
            
            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
            col1.metric('**input values (rows)**', selected_df.shape[0], help='number of rows')
            col2.metric('**variables (columns)**', selected_df.shape[1], help='number of columns')     
            col3.metric('**numerical variables**', len(selected_df.select_dtypes(include=['float64', 'int64']).columns), help='number of numerical variables')
            col4.metric('**categorical variables**', len(selected_df.select_dtypes(include=['object']).columns), help='number of categorical variables')
            col5.metric('**missing values**', selected_df.isnull().sum().sum(), help='Total missing values in the dataset')
            #col6.metric('**Unique categorical values**', sum(df.select_dtypes(include=['object']).nunique()), help='Sum of unique values in categorical variables')
         
        colSUM, colType, col3 = st.columns((0.4,0.2,0.4))
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
                        st.markdown("**Categorical Correlations**")
                        st.dataframe(proportion_table)
                    else:
                        st.warning("Please select at least one categorical variable to generate the categorical proportion table.")
                        
            with colnumPropTable: 
                with st.container(border=True):   
                    
                    selected_df_num = selected_df.select_dtypes(include=['int64', 'float64'])
                    st.markdown("**Numerical Correlations**")
                    if not selected_df_num.empty:
                        fig, ax = plt.subplots(figsize=(14,2.5))
                        sns.heatmap(selected_df_num.corr(), ax=ax, annot=True, linewidths=0.05, fmt='.2f', cmap="magma")
                        st.pyplot(fig)
                    else:
                        st.warning("No numerical columns found in the dataset.")      
        #-----------------------------------------------------------------------------------------------------------------------------------------------    
        st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Hypothesis Testing</span></div>',unsafe_allow_html=True,)
        #--------------------------------------------------------------- 
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["**T-Test**","**Chi-Square Test**","**Anova (F Statistics)**","**Mann-Whitney Test (U Statistics)**","**Kruskal-Wallis Test (h Statistics)**"])   
        #---------------------------------------------------------------         
        with tab1:
            
            st.info("Select one numeric variable and one categorical variable to perform the T-Test.")
            col1, col2= st.columns((0.2,0.8))
            with col1:
                with st.container(border=True):
                        
                    numeric_var = st.selectbox("**:blue[Select numeric variable]**", ["None"] + list(selected_df.select_dtypes(include=['int64', 'float64']).columns))
                    categorical_var = st.selectbox("**:blue[Select categorical variable]**", ["None"] + list(selected_df.select_dtypes(include=['object', 'category']).columns))
                    tail_option = st.selectbox("**:blue[Select test type]**", ["Two-Tailed", "One-Tailed"])

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
                                
                                    st.write("##### T-Test Results")
                                    t_test_df = pd.DataFrame({"Metric": ["T-Statistic", "P-Value"],"Value": [t_stat, p_value]})
                                    st.dataframe(t_test_df, hide_index=True)
                                    st.divider()
                                    if p_value < 0.05:
                                        st.write(f"**Interpretation**: The means of `{numeric_var}` differ significantly between the groups in `{categorical_var}` (at the 5% significance level).")
                                    else:
                                        st.write(f"**Interpretation**: There is no significant difference in the means of `{numeric_var}` between the groups in `{categorical_var}`.")
                            else:
                                st.warning("**Please select a categorical variable with exactly two unique groups for a two-sample T-Test.**")
        #---------------------------------------------------------------         
        with tab2:
            
            st.info("Select two or more categorical variables to perform pairwise Chi-Square Tests.")
            col1, col2= st.columns((0.2,0.8))
            with col1:
                with st.container(border=True):
                    
                    cat_vars = selected_df.select_dtypes(include=['object', 'category']).columns
                    selected_vars = st.multiselect("**:blue[Select categorical variables]**", cat_vars)
                    
                    with col2:
                        with st.container(border=True):
                            
                            if len(selected_vars) >= 2:
                                for i in range(len(selected_vars)):
                                    for j in range(i + 1, len(selected_vars)):
                                        var1, var2 = selected_vars[i], selected_vars[j]
                                        
                                        st.markdown(f"##### Chi-Square Test between `{var1}` and `{var2}`")
                                        subcol1, subcol2= st.columns(2)
                                        with subcol1:
                                            
                                            contingency_table = pd.crosstab(selected_df[var1], selected_df[var2])
                                            st.markdown("Contingency Table")
                                            st.dataframe(contingency_table)
                                        
                                        with subcol2:
                                        
                                            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

                                            st.markdown("Test Results")
                                            chi_square_df = pd.DataFrame({"Metric": ["Chi-Square Statistic", "Degrees of Freedom", "P-Value"],"Value": [chi2, dof, p]})
                                            st.dataframe(chi_square_df, hide_index=True)
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
                    ""
                    numeric_var = st.selectbox("**:blue[Select numeric variable]**", ["None"] + list(selected_df.select_dtypes(include=['int64','float64']).columns),key="numeric_var_anova")
                    categorical_var = st.selectbox("**:blue[Select categorical variable]**", ["None"] + list(selected_df.select_dtypes(include=['object', 'category']).columns),key="categorical_var_anova")

                    with col2:
                        with st.container(border=True):
                            
                            if numeric_var != "None" and categorical_var != "None":
                                groups = [group[numeric_var].dropna() for name, group in selected_df.groupby(categorical_var)]
            
                                if len(groups) >= 3:
                                    f_statistic, p_value = stats.f_oneway(*groups)
                                    
                                    st.write("##### ANOVA Results")
                                    anova_df = pd.DataFrame({"Metric": ["F-Statistic", "P-Value"],"Value": [f_statistic, p_value]})
                                    st.dataframe(anova_df, hide_index=True)                                    
                                    st.divider()
                                    if p_value < 0.05:
                                        st.write(f"**Interpretation**: There is a significant difference in `{numeric_var}` between the groups in `{categorical_var}`.")
                                    else:
                                        st.write(f"**Interpretation**: No significant difference in `{numeric_var}` across groups in `{categorical_var}`.")
                            else:
                                st.warning("**Please select a categorical variable with three or more unique groups.**")
        #---------------------------------------------------------------         
        with tab4:
            
            st.info("Select a numeric variable and a categorical variable with exactly two groups.")
            col1, col2= st.columns((0.2,0.8))
            with col1:
                with st.container(border=True):
                    
                    numeric_var_mw = st.selectbox("**:blue[Select numeric variable]**", ["None"] + list(selected_df.select_dtypes(include=['int64','float64']).columns),key="numeric_var_mw")
                    categorical_var_mw = st.selectbox("**:blue[Select categorical variable]**", ["None"] + list(selected_df.select_dtypes(include=['object','category']).columns),key="categorical_var_mw")

                    with col2:
                        with st.container(border=True):
                            
                            if numeric_var_mw != "None" and categorical_var_mw != "None":
                                unique_groups = selected_df[categorical_var_mw].dropna().unique()
                                
                                if len(unique_groups) == 2:
                                    group1 = selected_df[selected_df[categorical_var_mw] == unique_groups[0]][numeric_var_mw]
                                    group2 = selected_df[selected_df[categorical_var_mw] == unique_groups[1]][numeric_var_mw]

                                    u_statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                                    
                                    st.write("##### Mann-Whitney (U Statistics) Test Results")
                                    mw_df = pd.DataFrame({"Metric": ["U-Statistic", "P-Value"],"Value": [u_statistic, p_value]})
                                    st.dataframe(mw_df, hide_index=True)                                    
                                    st.divider()
                                    if p_value < 0.05:
                                        st.write(f"**Interpretation**: The distribution of `{numeric_var_mw}` differs significantly between the two groups in `{categorical_var_mw}`.")
                                    else:
                                        st.write(f"**Interpretation**: No significant difference in `{numeric_var}` distribution between the groups in `{categorical_var_mw}`.")
                            else:
                                st.warning("**Please select a categorical variable with exactly two unique groups.**")                
        #---------------------------------------------------------------         
        with tab5:
            
            st.info("Select a numeric variable and a categorical variable with three or more groups.")
            col1, col2= st.columns((0.2,0.8))
            with col1:
                with st.container(border=True):
                    
                    numeric_var= st.selectbox("**:blue[Select numeric variable]**", ["None"] + list(selected_df.select_dtypes(include=['int64', 'float64']).columns),key="numeric_var_kw")
                    categorical_var = st.selectbox("**:blue[Select categorical variable]**", ["None"] + list(selected_df.select_dtypes(include=['object', 'category']).columns),key="categorical_var_kw")

                    with col2:
                        with st.container(border=True):
                            
                            if numeric_var != "None" and categorical_var != "None":
                                groups = [group[numeric_var].dropna() for name, group in selected_df.groupby(categorical_var)]
            
                                if len(groups) >= 3:
                                    h_statistic, p_value = stats.kruskal(*groups)
                                    
                                    st.write("##### Kruskal-Wallis Test Results")

                                    kw_df = pd.DataFrame({"Metric": ["h-Statistic", "P-Value"],"Value": [h_statistic, p_value]})
                                    st.dataframe(mw_df, hide_index=True)                                    
                                    st.divider()
                                    if p_value < 0.05:
                                        st.write(f"**Interpretation**: The distribution of `{numeric_var}` differs significantly across groups in `{categorical_var}`.")
                                    else:
                                        st.write(f"**Interpretation**: No significant difference in `{numeric_var}` distribution across groups in `{categorical_var}`.")
                            else:
                                st.warning("**Please select a categorical variable with three or more unique groups.**")
        #-----------------------------------------------------------------------------------------------------------------------------------------------    
        st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Visualization</span></div>',unsafe_allow_html=True,)
        #--------------------------------------------------------------- 
            
        stats_expander = st.expander("**:blue[Density Distribution Plot]**", expanded=False)
        with stats_expander: 
            plot_histograms_with_kde(selected_df)

        st.divider()
        
        num_plots = st.selectbox("**:blue[Select the number of plots to generate]**", [1, 2, 3, 4], index=0)

        for i in range(num_plots):
            st.write(f"#### Plot {i + 1}")
            left_col, right_col = st.columns((0.2,0.8))

            with left_col:
                with st.container(border=True):
                
                    selected_var = st.selectbox("**:blue[Select X variable]**", selected_df.columns, key=f"x_{i}")
                    secondary_var = st.selectbox("**:blue[Select Y variable (optional)]**", ["None"] + list(selected_df.columns), key=f"y_{i}")
                    group_by_var = st.selectbox("**:blue[Select a grouping variable (optional)]**", ["None"] + list(selected_df.columns), key=f"group_{i}")

                    if secondary_var == "None":
                        plot_options = ["Histogram Plot", "Bar Plot", "Box Plot"]  
                    else:
                        plot_options = ["Histogram Plot", "Scatter Plot", "Line Plot", "Regression Plot", "Bar Plot", "Box Plot"]

                    plot_type = st.radio("**:blue[Select Plot Type]**", plot_options, index=0, key=f"type_{i}")
                    if plot_type in ["Scatter Plot", "Line Plot", "Regression Plot"] and secondary_var == "None":
                        st.warning(f"{plot_type} requires both a Y and an X variable. Please select a secondary variable (X) for this plot type.")
    
                    use_aggregation = st.checkbox(f"Apply Aggregation Function to Plot", key=f"agg_{i}")
                    if use_aggregation:
                        aggregation_function = st.selectbox(f"Select aggregation function for Plot", ["sum", "avg", "count", "min", "max"], key=f"agg_func_{i}")

                    default_title = f"{plot_type} of {selected_var}" + (f" vs {secondary_var}" if secondary_var != "None" else "") + (f" grouped by {group_by_var}" if group_by_var != "None" else "")
                    plot_title = st.text_input("**:blue[Set title for Plot]**", value=default_title, key=f"title_{i}")
                    plot_theme = st.selectbox("**:blue[Select Plot Theme]**", ["ggplot2", "seaborn", "simple_theme", "none"], key=f"theme_{i}")
                    theme = {"ggplot2": "ggplot2", "seaborn": "plotly_white", "simple_theme": "simple_white", "none": None}.get(plot_theme)

                    grouping_vars = [var for var in [secondary_var, group_by_var] if var != "None"]
                    aggregated_data = selected_df  # Default to non-aggregated data
                    if use_aggregation:
                        if grouping_vars:

                            if aggregation_function == "sum":
                                aggregated_data = selected_df.groupby(grouping_vars)[selected_var].sum().reset_index()
                            elif aggregation_function == "avg":
                                aggregated_data = selected_df.groupby(grouping_vars)[selected_var].mean().reset_index()
                            elif aggregation_function == "count":
                                aggregated_data = selected_df.groupby(grouping_vars)[selected_var].count().reset_index()
                            elif aggregation_function == "min":
                                aggregated_data = selected_df.groupby(grouping_vars)[selected_var].min().reset_index()
                            elif aggregation_function == "max":
                                aggregated_data = selected_df.groupby(grouping_vars)[selected_var].max().reset_index()
                        else:
                            st.warning("To use an aggregation function, please select a secondary variable (X) or a grouping variable.")
                    else:
                        aggregated_data = selected_df

            with right_col:
                
                    if plot_type == "Histogram Plot":
                        fig = px.histogram(aggregated_data, x=selected_var, color=group_by_var if group_by_var != "None" else None, nbins=30, template=theme,
                                   title=plot_title)
                    elif plot_type == "Scatter Plot" and secondary_var != "None":
                        fig = px.scatter(aggregated_data, x=secondary_var, y=selected_var, color=group_by_var if group_by_var != "None" else None, template=theme,
                                 title=plot_title)
                    elif plot_type == "Line Plot" and secondary_var != "None":
                        fig = px.line(aggregated_data, x=secondary_var, y=selected_var, color=group_by_var if group_by_var != "None" else None, template=theme,
                              title=plot_title)
                    elif plot_type == "Regression Plot" and secondary_var != "None":
                        fig = px.scatter(aggregated_data, x=secondary_var, y=selected_var, color=group_by_var if group_by_var != "None" else None, trendline="ols", template=theme,
                                 title=plot_title)
                    elif plot_type == "Bar Plot":
                        fig = px.bar(aggregated_data, y=selected_var, x=secondary_var if secondary_var != "None" else None, color=group_by_var if group_by_var != "None" else None,
                             template=theme, title=plot_title)
                        fig.update_layout(barmode='group' if group_by_var != "None" else 'relative')
                    elif plot_type == "Box Plot":
                        fig = px.box(aggregated_data, y=selected_var, x=secondary_var if secondary_var != "None" else None, color=group_by_var if group_by_var != "None" else None,
                             template=theme, title=plot_title)

                    st.plotly_chart(fig, use_container_width=True, key=f"plot_{i}")
            
                    #png_buffer = get_plotly_download(fig, file_format="png", scale=3)
                    #html_buffer = get_plotly_download(fig, file_format="html")

                    #st.download_button(label="Download as PNG (High Quality)",data=png_buffer,file_name=f"plot_{i + 1}.png",mime="image/png")
                    #st.download_button(label="Download as HTML (Interactive)",data=html_buffer,file_name=f"plot_{i + 1}.html",mime="text/html")
