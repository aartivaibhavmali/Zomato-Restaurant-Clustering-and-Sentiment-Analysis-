# app.py
# Zomato Restaurant Analytics & Review Sentiment Dashboard
# Keep this file in the same folder as the two CSV files and run:
# pip install streamlit pandas numpy plotly scikit-learn joblib
# streamlit run app.py

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Zomato Analytics | Aarti",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- DESIGN ---------------------------------
st.markdown("""
<style>
.main {background-color: #fbfbfd;}
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {
    padding: 1.5rem 1.8rem; border-radius: 18px;
    background: linear-gradient(110deg,#8b1538,#d94b54);
    color: white; margin-bottom: 1.2rem;
}
.hero h1 {color:white; margin:0; font-size:2.15rem;}
.hero p {color:#fff; opacity:.92; margin:.45rem 0 0 0;}
div[data-testid="stMetric"] {
    background: white; border: 1px solid #eee; padding: 14px;
    border-radius: 14px; box-shadow: 0 2px 10px rgba(0,0,0,.035);
}
section[data-testid="stSidebar"] {background:#fff7f8;}
.small-note {color:#777;font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

RESTAURANT_FILE = "Zomato Restaurant names and Metadata.csv"
REVIEW_FILE = "Zomato Restaurant reviews.csv"
SENTIMENT_MODEL_FILE = "zomato_sentiment_pipeline.joblib"
CLUSTER_MODEL_FILE = "zomato_restaurant_clustering.joblib"


# ----------------------------- DATA -----------------------------------
@st.cache_data(show_spinner="Loading Zomato datasets...")
def load_data(restaurant_path, review_path):
    restaurants = pd.read_csv(restaurant_path)
    reviews = pd.read_csv(review_path)
    restaurants = restaurants.drop_duplicates().copy()
    reviews = reviews.drop_duplicates().copy()

    required_rest = {"Name", "Cost", "Cuisines"}
    required_rev = {"Review", "Rating"}
    if not required_rest.issubset(restaurants.columns):
        raise ValueError(f"Restaurant CSV must contain: {sorted(required_rest)}")
    if not required_rev.issubset(reviews.columns):
        raise ValueError(f"Review CSV must contain: {sorted(required_rev)}")

    restaurants["Cost"] = pd.to_numeric(
        restaurants["Cost"].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )
    restaurants["Cost"] = restaurants["Cost"].fillna(restaurants["Cost"].median())
    restaurants["Cuisines"] = restaurants["Cuisines"].fillna("Unknown")
    restaurants["Collections"] = restaurants.get(
        "Collections", pd.Series("", index=restaurants.index)
    ).fillna("")
    restaurants["Timings"] = restaurants.get(
        "Timings", pd.Series("Unknown", index=restaurants.index)
    ).fillna("Unknown")

    restaurants["cuisine_list"] = restaurants["Cuisines"].apply(
        lambda x: [v.strip() for v in str(x).split(",")
                   if v.strip() and v.strip().lower() not in {"unknown", "nan", "none"}]
    )
    restaurants["n_cuisines"] = restaurants["cuisine_list"].str.len()
    restaurants["n_collections"] = restaurants["Collections"].apply(
        lambda x: len([v for v in str(x).split(",") if v.strip()])
    )
    restaurants["has_collection"] = (restaurants["n_collections"] > 0).astype(int)
    restaurants["is_open_late"] = restaurants["Timings"].str.contains(
        r"12\s*(?:midnight|am)|1\s*am|2\s*am|3\s*am",
        case=False, regex=True, na=False
    ).astype(int)
    q1, q3 = restaurants["Cost"].quantile([.25, .75])
    upper = q3 + 1.5 * (q3 - q1)
    restaurants["Cost_capped"] = restaurants["Cost"].clip(upper=upper)

    reviews = reviews.dropna(subset=["Review"]).copy()
    reviews["Review"] = reviews["Review"].astype(str)
    reviews["Rating_clean"] = pd.to_numeric(reviews["Rating"], errors="coerce")
    reviews = reviews[reviews["Rating_clean"].between(0, 5)].copy()
    reviews["Sentiment"] = np.select(
        [reviews["Rating_clean"] <= 2, reviews["Rating_clean"] == 3],
        ["Negative", "Neutral"], default="Positive"
    )
    if "Time" in reviews.columns:
        dt = pd.to_datetime(reviews["Time"], errors="coerce")
        reviews["Hour"] = dt.dt.hour
        reviews["Day"] = dt.dt.day_name()
        reviews["Weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    else:
        reviews["Hour"] = np.nan
        reviews["Day"] = "Unknown"
        reviews["Weekend"] = 0

    if "Metadata" in reviews.columns:
        def metadata_counts(value):
            m = re.search(r"(\d+)\s*Reviews?\s*,\s*(\d+)\s*Followers?",
                          str(value), flags=re.I)
            return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        reviews[["num_reviews", "num_followers"]] = reviews["Metadata"].apply(
            lambda x: pd.Series(metadata_counts(x))
        )
    else:
        reviews["num_reviews"] = 0
        reviews["num_followers"] = 0

    return restaurants, reviews


def find_file(filename, uploaded_file):
    if uploaded_file is not None:
        return uploaded_file
    if Path(filename).exists():
        return filename
    return None


@st.cache_resource(show_spinner="Preparing the review sentiment model...")
def train_sentiment_model(texts, labels):
    # Model uses the same rating-derived labels as the notebook:
    # <=2 Negative, 3 Neutral, >=4 Positive.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            min_df=2, max_df=.98, sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        ))
    ])
    model.fit(texts, labels)
    return model


@st.cache_data(show_spinner=False)
def make_restaurant_clusters(frame):
    features = frame[
        ["Cost_capped", "n_cuisines", "n_collections",
         "has_collection", "is_open_late"]
    ].fillna(0).copy()
    # Add cuisine indicators just as in the project clustering workflow.
    cuisine_dummies = frame["cuisine_list"].apply(
        lambda items: pd.Series({item: 1 for item in items})
    ).fillna(0)
    features = pd.concat(
        [features.reset_index(drop=True), cuisine_dummies.reset_index(drop=True)],
        axis=1
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    k = min(2, len(frame))
    if len(frame) >= 10:
        k = 2
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)
    result = frame.copy()
    result["KMeans_Cluster"] = labels.astype(str)
    return result


# ----------------------------- HEADER ---------------------------------
st.markdown("""
<div class="hero">
  <h1>🍽️ Zomato Restaurant Intelligence</h1>
  <p>Explore restaurant trends, customer ratings, cuisine patterns and AI-powered review sentiment.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    st.caption("Upload the two project CSV files, or keep them beside app.py.")
    restaurant_upload = st.file_uploader(
        "Restaurant metadata CSV", type=["csv"], key="restaurant_csv"
    )
    review_upload = st.file_uploader(
        "Restaurant reviews CSV", type=["csv"], key="review_csv"
    )
    st.divider()
    page = st.radio(
        "Navigate",
        ["Overview", "Restaurant Explorer", "Review Insights",
         "Sentiment Predictor", "Restaurant Clusters", "About Project"]
    )
    st.divider()
    st.caption("Project dashboard | Created by Aarti Vaibhav Mali")

rest_source = find_file(RESTAURANT_FILE, restaurant_upload)
review_source = find_file(REVIEW_FILE, review_upload)

if rest_source is None or review_source is None:
    st.warning("Please upload both CSV files in the sidebar, or place both files in the same folder as app.py.")
    st.code(f"{RESTAURANT_FILE}\n{REVIEW_FILE}")
    st.stop()

try:
    Rest, Rev = load_data(rest_source, review_source)
except Exception as exc:
    st.error(f"Could not read the datasets: {exc}")
    st.stop()

# Optional trained project artifact. If absent, a model is trained from the
# uploaded review data when the user opens Sentiment Predictor.
sentiment_model = None
if Path(SENTIMENT_MODEL_FILE).exists():
    try:
        sentiment_model = joblib.load(SENTIMENT_MODEL_FILE)
    except Exception:
        sentiment_model = None


# ----------------------------- OVERVIEW -------------------------------
if page == "Overview":
    st.subheader("Project at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Restaurants", f"{len(Rest):,}")
    c2.metric("Customer reviews", f"{len(Rev):,}")
    c3.metric("Average restaurant cost", f"₹{Rest['Cost'].mean():,.0f}")
    c4.metric("Average review rating", f"{Rev['Rating_clean'].mean():.2f}/5")

    left, right = st.columns(2)
    with left:
        rating_counts = Rev["Rating_clean"].value_counts().sort_index().rename_axis("Rating").reset_index(name="Reviews")
        fig = px.bar(rating_counts, x="Rating", y="Reviews",
                     title="Customer Rating Distribution", text_auto=True,
                     color="Rating", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        cuisine_counts = (Rest.explode("cuisine_list")["cuisine_list"]
                          .value_counts().head(10).rename_axis("Cuisine")
                          .reset_index(name="Restaurant Count"))
        fig = px.bar(cuisine_counts.sort_values("Restaurant Count"),
                     x="Restaurant Count", y="Cuisine", orientation="h",
                     title="Top 10 Cuisines", color="Restaurant Count",
                     color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = px.histogram(Rest, x="Cost_capped", nbins=30,
                           title="Restaurant Cost Distribution",
                           labels={"Cost_capped": "Cost for two (₹)"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        sentiment_counts = Rev["Sentiment"].value_counts().reindex(
            ["Positive", "Neutral", "Negative"], fill_value=0
        ).rename_axis("Sentiment").reset_index(name="Review Count")
        fig = px.pie(sentiment_counts, names="Sentiment", values="Review Count",
                     title="Rating-derived Sentiment Split", hole=.48,
                     color="Sentiment",
                     color_discrete_map={"Positive":"#2e9d68","Neutral":"#e6a23c","Negative":"#d94b54"})
        st.plotly_chart(fig, use_container_width=True)
    st.info("Sentiment categories are derived from ratings: 1–2 = Negative, 3 = Neutral, and 4–5 = Positive. They are not separate human annotations.")


# ------------------------- RESTAURANT EXPLORER -------------------------
elif page == "Restaurant Explorer":
    st.subheader("🔎 Find and compare restaurants")
    col1, col2, col3 = st.columns(3)
    max_cost = float(Rest["Cost"].max()) if len(Rest) else 0
    cost_range = col1.slider("Cost for two (₹)", 0.0, max_cost,
                              (0.0, max_cost), step=max(1.0, max_cost/100))
    cuisine_options = sorted({c for items in Rest["cuisine_list"] for c in items})
    selected_cuisines = col2.multiselect("Cuisine", cuisine_options)
    search_name = col3.text_input("Restaurant name contains")

    filtered = Rest[Rest["Cost"].between(*cost_range)].copy()
    if selected_cuisines:
        filtered = filtered[filtered["cuisine_list"].apply(
            lambda items: any(c in items for c in selected_cuisines)
        )]
    if search_name:
        filtered = filtered[filtered["Name"].astype(str).str.contains(
            search_name, case=False, na=False
        )]

    st.write(f"**{len(filtered):,} restaurants found**")
    display_cols = [c for c in [
        "Name", "Cost", "Cuisines", "Collections", "Timings"
    ] if c in filtered.columns]
    st.dataframe(filtered[display_cols].sort_values("Cost"),
                 use_container_width=True, hide_index=True)

    if not filtered.empty:
        top = filtered.nlargest(10, "Cost")
        fig = px.bar(top, x="Name", y="Cost", color="Cost",
                     title="10 Highest-cost restaurants in current selection",
                     labels={"Cost":"Cost for two (₹)"})
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------- REVIEW INSIGHTS --------------------------
elif page == "Review Insights":
    st.subheader("⭐ Customer review and rating analysis")
    col1, col2 = st.columns(2)
    with col1:
        sentiment = st.selectbox("Filter sentiment", ["All", "Positive", "Neutral", "Negative"])
    with col2:
        min_rating, max_rating = st.slider("Rating range", 0.0, 5.0, (0.0, 5.0), .5)

    data = Rev[Rev["Rating_clean"].between(min_rating, max_rating)].copy()
    if sentiment != "All":
        data = data[data["Sentiment"] == sentiment]
    st.metric("Reviews matching filters", f"{len(data):,}")

    left, right = st.columns(2)
    with left:
        day_counts = data["Day"].value_counts().rename_axis("Day").reset_index(name="Reviews")
        fig = px.bar(day_counts, x="Day", y="Reviews", title="Reviews by day of week")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if data["Hour"].notna().any():
            hour_counts = data["Hour"].value_counts().sort_index().rename_axis("Hour").reset_index(name="Reviews")
            fig = px.line(hour_counts, x="Hour", y="Reviews", markers=True,
                          title="Review activity by hour")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("The review file does not contain usable time values.")

    if "Reviewer" in data.columns and "num_followers" in data.columns:
        critics = (data.groupby("Reviewer", dropna=False)
                   .agg(Followers=("num_followers", "max"),
                        Review_Count=("Review", "count"),
                        Average_Rating=("Rating_clean", "mean"))
                   .sort_values(["Followers", "Review_Count"], ascending=False)
                   .head(10).reset_index())
        st.markdown("#### Top reviewers by follower count")
        st.dataframe(critics, use_container_width=True, hide_index=True)

    st.markdown("#### Recent review records")
    review_cols = [c for c in ["Restaurant", "Reviewer", "Review", "Rating_clean", "Sentiment"] if c in data.columns]
    st.dataframe(data[review_cols].head(300), use_container_width=True, hide_index=True)


# -------------------------- SENTIMENT PREDICTOR ------------------------
elif page == "Sentiment Predictor":
    st.subheader("🤖 AI Review Sentiment Predictor")
    st.write("Enter a customer review to predict its sentiment using the project's rating-derived training labels.")
    if sentiment_model is None:
        if len(Rev) < 20 or Rev["Sentiment"].nunique() < 2:
            st.error("There are not enough labelled reviews to train a sentiment model.")
            st.stop()
        with st.spinner("Training a Logistic Regression + TF-IDF model on the review dataset..."):
            sentiment_model = train_sentiment_model(
                Rev["Review"].astype(str).tolist(),
                Rev["Sentiment"].tolist()
            )
        st.caption("No saved model artifact was found. A model has been trained from the uploaded review data for this session.")
    else:
        st.caption(f"Loaded saved model: {SENTIMENT_MODEL_FILE}")

    review_text = st.text_area(
        "Type or paste a review",
        placeholder="Example: The food was delicious and the staff were very friendly.",
        height=130
    )
    if st.button("Predict Sentiment", type="primary", use_container_width=True):
        if len(review_text.strip()) < 3:
            st.warning("Please enter a longer review.")
        else:
            prediction = sentiment_model.predict([review_text.strip()])[0]
            colors = {"Positive":"success", "Neutral":"info", "Negative":"error"}
            getattr(st, colors.get(str(prediction), "info"))(
                f"Predicted sentiment: {prediction}"
            )
            if hasattr(sentiment_model, "predict_proba"):
                probabilities = sentiment_model.predict_proba([review_text.strip()])[0]
                classes = sentiment_model.classes_
                prob_df = pd.DataFrame({"Sentiment": classes, "Probability": probabilities})
                fig = px.bar(prob_df, x="Sentiment", y="Probability",
                             range_y=[0, 1], title="Prediction probability",
                             color="Sentiment",
                             color_discrete_map={"Positive":"#2e9d68","Neutral":"#e6a23c","Negative":"#d94b54"})
                st.plotly_chart(fig, use_container_width=True)
            st.caption("This is a model prediction, not a guarantee of the reviewer's actual emotion.")


# -------------------------- RESTAURANT CLUSTERS ------------------------
elif page == "Restaurant Clusters":
    st.subheader("🧩 Restaurant segmentation with K-Means")
    if len(Rest) < 3:
        st.warning("At least three restaurants are required for clustering.")
    else:
        clustered = make_restaurant_clusters(Rest)
        profiles = clustered.groupby("KMeans_Cluster").agg(
            Restaurant_Count=("Name", "count"),
            Average_Cost=("Cost_capped", "mean"),
            Average_Cuisine_Count=("n_cuisines", "mean"),
            Average_Collections=("n_collections", "mean")
        ).round(2).reset_index()
        st.dataframe(profiles, use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            fig = px.scatter(clustered, x="Cost_capped", y="n_cuisines",
                             color="KMeans_Cluster", hover_name="Name",
                             title="Restaurant clusters: cost vs cuisine variety",
                             labels={"Cost_capped":"Capped cost for two (₹)",
                                     "n_cuisines":"Number of cuisines"})
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = px.bar(profiles, x="KMeans_Cluster", y="Restaurant_Count",
                         color="KMeans_Cluster", title="Restaurants in each cluster")
            st.plotly_chart(fig, use_container_width=True)
        st.info("Cluster labels are exploratory groups, not verified customer segments. The notebook reported a low K-Means silhouette score (about 0.126), so the groups overlap.")
        st.markdown("#### Explore restaurants in a cluster")
        chosen_cluster = st.selectbox("Choose cluster", sorted(clustered["KMeans_Cluster"].unique()))
        st.dataframe(
            clustered.loc[clustered["KMeans_Cluster"] == chosen_cluster,
                          [c for c in ["Name", "Cost", "Cuisines", "Timings"] if c in clustered.columns]],
            use_container_width=True, hide_index=True
        )


# ----------------------------- ABOUT -----------------------------------
elif page == "About Project":
    st.subheader("📌 About this project")
    st.markdown("""
    **Project:** Zomato Restaurant Analytics and Customer Review Sentiment Analysis

    **Project modules**
    - Data cleaning and preprocessing for restaurant metadata and customer reviews.
    - Exploratory analysis of restaurant cost, cuisine variety, ratings and review activity.
    - Rating-derived sentiment classification: rating ≤ 2 = Negative, rating = 3 = Neutral, rating ≥ 4 = Positive.
    - Restaurant clustering using cost, cuisine count, collection count, late-night availability and cuisine indicators.
    - Interactive review sentiment prediction.

    **Models in the notebook**
    - Restaurant clustering: K-Means, Hierarchical Clustering and DBSCAN.
    - Review classification: Logistic Regression, Multinomial Naive Bayes, Random Forest and tuned Linear SVC.

    **Important interpretation**
    - Clustering silhouette scores and classification scores measure different tasks and should not be compared.
    - The notebook reports overlapping restaurant clusters (K-Means silhouette approximately 0.126).
    - Neutral sentiment is harder to identify than positive and negative reviews in the classification experiments.
    - Sentiment labels are generated from ratings, not independently annotated by humans.
    """)
    st.markdown("**Created by:** Aarti Vaibhav Mali")

st.divider()
st.markdown(
    "<div class='small-note'>Zomato Analytics Dashboard • Built with Streamlit, Pandas, Plotly and Scikit-learn</div>",
    unsafe_allow_html=True
)
