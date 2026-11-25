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
