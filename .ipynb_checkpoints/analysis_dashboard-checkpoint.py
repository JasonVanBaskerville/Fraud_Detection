import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Fraud Detection Analysis Dashboard",
    page_icon="🔎",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv("./fraud_sample.csv")

    return data


data = load_data()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🔎 Fraud Analysis")

st.sidebar.markdown(
    """
    This dashboard is used to analyze the patterns and characteristics of fraud transactions.
    """
)


# =========================================================
# SIDEBAR FILTER
# =========================================================

st.sidebar.subheader("Data Filters")

transaction_types = st.sidebar.multiselect(
    "Transaction Type",
    options=sorted(data["type"].unique()),
    default=sorted(data["type"].unique())
)

fraud_filter = st.sidebar.selectbox(
    "Transaction Status",
    ["All", "Fraud", "Non-Fraud"]
)


filtered_data = data[
    data["type"].isin(transaction_types)
].copy()


if fraud_filter == "Fraud":

    filtered_data = filtered_data[
        filtered_data["isFraud"] == 1
    ]

elif fraud_filter == "Non-Fraud":

    filtered_data = filtered_data[
        filtered_data["isFraud"] == 0
    ]


# =========================================================
# TITLE
# =========================================================

st.title("🔎 Fraud Detection Analysis Dashboard")

st.markdown(
    """
    This dashboard provides an exploratory analysis of transactions to identify fraud patterns and characteristics.
    """
)


# =========================================================
# OVERVIEW
# =========================================================

st.header("📊 Dataset Overview")


total_transaction = len(filtered_data)

total_fraud = (
    filtered_data["isFraud"] == 1
).sum()

total_nonfraud = (
    filtered_data["isFraud"] == 0
).sum()


fraud_rate = (
    total_fraud / total_transaction * 100
    if total_transaction > 0
    else 0
)


total_amount = filtered_data["amount"].sum()

fraud_amount = filtered_data.loc[
    filtered_data["isFraud"] == 1,
    "amount"
].sum()


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Total Transactions",
        f"{total_transaction:,}"
    )


with col2:

    st.metric(
        "Fraud Transactions",
        f"{total_fraud:,}"
    )


with col3:

    st.metric(
        "Non-Fraud Transactions",
        f"{total_nonfraud:,}"
    )


with col4:

    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )


with col5:

    st.metric(
        "Fraud Amount",
        f"${fraud_amount:,.0f}"
    )


# =========================================================
# FRAUD DISTRIBUTION
# =========================================================

st.header("🚨 Fraud Distribution")


col1, col2 = st.columns(2)


with col1:

    st.subheader("Fraud vs Non-Fraud")

    fraud_count = filtered_data["isFraud"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        ["Non-Fraud", "Fraud"],
        [
            fraud_count.get(0, 0),
            fraud_count.get(1, 0)
        ]
    )

    ax.set_ylabel("Number of Transactions")

    st.pyplot(fig)

    plt.close(fig)


with col2:

    st.subheader("Fraud Rate by Transaction Type")

    fraud_by_type = (
        filtered_data
        .groupby("type")["isFraud"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    fig, ax = plt.subplots()

    fraud_by_type.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Fraud Rate (%)")

    ax.set_xlabel("Transaction Type")

    plt.xticks(rotation=0)

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# TRANSACTION TYPE
# =========================================================

st.header("💳 Transaction Analysis")


col1, col2 = st.columns(2)


with col1:

    st.subheader("Transaction Count by Type")

    type_count = (
        filtered_data["type"]
        .value_counts()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots()

    type_count.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Number of Transactions")

    ax.set_xlabel("Transaction Type")

    plt.xticks(rotation=0)

    st.pyplot(fig)

    plt.close(fig)


with col2:

    st.subheader("Transaction Amount Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        filtered_data["amount"],
        bins=50
    )

    ax.set_xlabel("Transaction Amount")

    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# AMOUNT ANALYSIS
# =========================================================

st.header("💰 Amount Analysis")


amount_data = filtered_data.copy()

amount_data["log_amount"] = np.log1p(
    amount_data["amount"]
)


col1, col2 = st.columns(2)


with col1:

    st.subheader("Amount: Fraud vs Non-Fraud")

    fig, ax = plt.subplots()

    sns.boxplot(
        data=amount_data,
        x="isFraud",
        y="amount",
        ax=ax
    )

    ax.set_xlabel("Transaction Status")

    ax.set_ylabel("Amount")

    ax.set_xticklabels(
        ["Non-Fraud", "Fraud"]
    )

    st.pyplot(fig)

    plt.close(fig)


with col2:

    st.subheader("Log Amount: Fraud vs Non-Fraud")

    fig, ax = plt.subplots()

    sns.boxplot(
        data=amount_data,
        x="isFraud",
        y="log_amount",
        ax=ax
    )

    ax.set_xlabel("Transaction Status")

    ax.set_ylabel("log(Amount + 1)")

    ax.set_xticklabels(
        ["Non-Fraud", "Fraud"]
    )

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# TIME ANALYSIS
# =========================================================

st.header("⏱️ Time Analysis")


fraud_per_step = (
    filtered_data
    .groupby("step")
    .agg(
        total_transactions=("isFraud", "count"),
        total_fraud=("isFraud", "sum")
    )
)

fraud_per_step["fraud_rate"] = (
    fraud_per_step["total_fraud"]
    / fraud_per_step["total_transactions"]
    * 100
)


col1, col2 = st.columns(2)


with col1:

    st.subheader("Number of Fraud by Step")

    fig, ax = plt.subplots()

    ax.plot(
        fraud_per_step.index,
        fraud_per_step["total_fraud"]
    )

    ax.set_xlabel("Step")

    ax.set_ylabel("Number of Fraud")

    st.pyplot(fig)

    plt.close(fig)


with col2:

    st.subheader("Fraud Rate by Step")

    fig, ax = plt.subplots()

    ax.plot(
        fraud_per_step.index,
        fraud_per_step["fraud_rate"]
    )

    ax.set_xlabel("Step")

    ax.set_ylabel("Fraud Rate (%)")

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# BALANCE ANALYSIS
# =========================================================

st.header("💵 Balance Analysis")


col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Origin Balance: Old vs New"
    )

    sample_data = filtered_data.sample(
        min(5000, len(filtered_data)),
        random_state=42
    )

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=sample_data,
        x="oldbalanceOrg",
        y="newbalanceOrig",
        hue="isFraud",
        alpha=0.5,
        ax=ax
    )

    ax.set_xlabel("Old Balance Origin")

    ax.set_ylabel("New Balance Origin")

    st.pyplot(fig)

    plt.close(fig)


with col2:

    st.subheader(
        "Destination Balance: Old vs New"
    )

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=sample_data,
        x="oldbalanceDest",
        y="newbalanceDest",
        hue="isFraud",
        alpha=0.5,
        ax=ax
    )

    ax.set_xlabel("Old Balance Destination")

    ax.set_ylabel("New Balance Destination")

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# BALANCE CHANGE
# =========================================================

st.header("📈 Balance Change Analysis")


balance_data = filtered_data.copy()


balance_data["balance_change_origin"] = (
    balance_data["oldbalanceOrg"]
    - balance_data["newbalanceOrig"]
)


balance_data["balance_change_destination"] = (
    balance_data["newbalanceDest"]
    - balance_data["oldbalanceDest"]
)


col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "Origin Balance Change"
    )

    fig, ax = plt.subplots()

    sns.boxplot(
        data=balance_data,
        x="isFraud",
        y="balance_change_origin",
        ax=ax
    )

    ax.set_xticklabels(
        ["Non-Fraud", "Fraud"]
    )

    ax.set_xlabel("Transaction Status")

    ax.set_ylabel("Balance Change")

    st.pyplot(fig)

    plt.close(fig)


with col2:

    st.subheader(
        "Destination Balance Change"
    )

    fig, ax = plt.subplots()

    sns.boxplot(
        data=balance_data,
        x="isFraud",
        y="balance_change_destination",
        ax=ax
    )

    ax.set_xticklabels(
        ["Non-Fraud", "Fraud"]
    )

    ax.set_xlabel("Transaction Status")

    ax.set_ylabel("Balance Change")

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# TOP FRAUD TRANSACTIONS
# =========================================================

st.header("🚨 Fraud Transactions")


fraud_table = filtered_data[
    filtered_data["isFraud"] == 1
].copy()


fraud_table = fraud_table.sort_values(
    "amount",
    ascending=False
)


columns_to_show = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud"
]


columns_to_show = [
    col for col in columns_to_show
    if col in fraud_table.columns
]


st.dataframe(
    fraud_table[columns_to_show],
    use_container_width=True,
    height=400
)


# =========================================================
# SUMMARY
# =========================================================

st.header("📝 Analysis Summary")


if len(filtered_data) > 0:

    highest_fraud_type = (
        filtered_data
        .groupby("type")["isFraud"]
        .mean()
        .idxmax()
    )

    highest_fraud_rate = (
        filtered_data
        .groupby("type")["isFraud"]
        .mean()
        .max()
        * 100
    )

    highest_fraud_step = (
        fraud_per_step["total_fraud"]
        .idxmax()
    )


    st.info(
        f"""
        **Key Findings**

        • The fraud rate on filtered data is
        **{fraud_rate:.2f}%**.

        • The transaction type with the highest fraud rate is
        **{highest_fraud_type}**
        with a fraud rate of around
        **{highest_fraud_rate:.2f}%**.

        • The highest number of fraud cases occurred in
        **step {highest_fraud_step}**.

        • The total nominal value of fraud transactions reached
        **${fraud_amount:,.0f}**.
        """
    )