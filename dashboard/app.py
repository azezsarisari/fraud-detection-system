import os
import io
import requests
import pandas as pd
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

PREDICT_URL = f"{API_BASE_URL}/predict"
BATCH_URL = f"{API_BASE_URL}/predict-batch"
HEALTH_URL = f"{API_BASE_URL}/health"
MODEL_INFO_URL = f"{API_BASE_URL}/model-info"


st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="FD",
    layout="wide"
)


# =========================================================
# HELPERS
# =========================================================

def check_api():

    try:
        response = requests.get(
            HEALTH_URL,
            timeout=5
        )

        if response.status_code == 200:
            return response.json()

    except requests.RequestException:
        pass

    return {
        "status": "unavailable",
        "model_loaded": False
    }


def get_model_info():

    try:
        response = requests.get(
            MODEL_INFO_URL,
            timeout=5
        )

        if response.status_code == 200:
            return response.json()

    except requests.RequestException:
        pass

    return None


def risk_message(risk_level):

    messages = {
        "LOW": "Low fraud risk",
        "MEDIUM": "Transaction requires review",
        "HIGH": "High fraud risk",
        "CRITICAL": "Critical fraud risk detected"
    }

    return messages.get(
        risk_level,
        "Unknown risk"
    )


# =========================================================
# HEADER
# =========================================================

st.title("Fraud Detection System")

st.write(
    "End-to-end machine learning system for "
    "detecting suspicious financial transactions."
)


health = check_api()


if (
    health["status"] == "healthy"
    and health["model_loaded"]
):

    st.success(
        "API connected — model loaded"
    )

else:

    st.error(
        "FastAPI backend is not available."
    )

    st.stop()


# =========================================================
# MODEL INFO
# =========================================================

model_info = get_model_info()


if model_info:

    with st.expander(
        "Model Information"
    ):

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Model",
            model_info["model"]
        )

        col2.metric(
            "Dataset",
            model_info["dataset"]
        )

        col3.metric(
            "Threshold",
            f'{model_info["threshold"]:.2f}'
        )


        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Precision",
            f'{model_info["precision"] * 100:.2f}%'
        )

        col5.metric(
            "Recall",
            f'{model_info["recall"] * 100:.2f}%'
        )

        col6.metric(
            "F1 Score",
            f'{model_info["f1"] * 100:.2f}%'
        )


# =========================================================
# TABS
# =========================================================

single_tab, batch_tab = st.tabs(
    [
        "Single Transaction",
        "Batch CSV Analysis"
    ]
)


# =========================================================
# SINGLE TRANSACTION
# =========================================================

with single_tab:

    st.subheader(
        "Analyze a Single Transaction"
    )

    col1, col2 = st.columns(2)


    with col1:

        step = st.number_input(
            "Transaction Step",
            min_value=0,
            value=1,
            step=1
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "TRANSFER",
                "CASH_OUT",
                "PAYMENT",
                "CASH_IN",
                "DEBIT"
            ]
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=181.0,
            step=100.0
        )


    with col2:

        oldbalanceOrg = st.number_input(
            "Sender Old Balance",
            min_value=0.0,
            value=181.0
        )

        newbalanceOrig = st.number_input(
            "Sender New Balance",
            min_value=0.0,
            value=0.0
        )

        oldbalanceDest = st.number_input(
            "Receiver Old Balance",
            min_value=0.0,
            value=0.0
        )

        newbalanceDest = st.number_input(
            "Receiver New Balance",
            min_value=0.0,
            value=0.0
        )


    if st.button(
        "Analyze Transaction",
        type="primary"
    ):

        payload = {
            "step": int(step),
            "type": transaction_type,
            "amount": float(amount),
            "oldbalanceOrg": float(oldbalanceOrg),
            "newbalanceOrig": float(newbalanceOrig),
            "oldbalanceDest": float(oldbalanceDest),
            "newbalanceDest": float(newbalanceDest)
        }


        try:

            response = requests.post(
                PREDICT_URL,
                json=payload,
                timeout=30
            )

        except requests.RequestException as error:

            st.error(
                f"API request failed: {error}"
            )

        else:

            if response.status_code != 200:

                st.error(
                    response.text
                )

            else:

                result = response.json()

                prediction = result[
                    "prediction"
                ]

                probability = result[
                    "fraud_probability"
                ]

                risk_level = result[
                    "risk_level"
                ]


                st.divider()

                r1, r2, r3 = st.columns(3)

                r1.metric(
                    "Prediction",
                    prediction
                )

                r2.metric(
                    "Fraud Probability",
                    f"{probability * 100:.2f}%"
                )

                r3.metric(
                    "Risk Level",
                    risk_level
                )


                if prediction == "FRAUD":

                    st.error(
                        "Fraudulent transaction detected."
                    )

                else:

                    st.success(
                        "Transaction classified as legitimate."
                    )


                st.info(
                    risk_message(
                        risk_level
                    )
                )


# =========================================================
# BATCH ANALYSIS
# =========================================================

with batch_tab:

    st.subheader(
        "Batch Transaction Analysis"
    )

    st.write(
        "Upload a CSV containing up to "
        "10,000 transactions."
    )


    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"]
    )


    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]


    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )

        except Exception as error:

            st.error(
                f"Could not read CSV: {error}"
            )

            st.stop()


        st.write(
            f"Rows loaded: {len(batch_df):,}"
        )


        missing_columns = [
            column
            for column in required_columns
            if column not in batch_df.columns
        ]


        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(
                    missing_columns
                )
            )

        elif len(batch_df) > 10000:

            st.error(
                "Maximum batch size is "
                "10,000 transactions."
            )

        else:

            st.dataframe(
                batch_df.head(20),
                use_container_width=True
            )


            if st.button(
                "Analyze CSV",
                type="primary"
            ):

                records = (
                    batch_df[
                        required_columns
                    ]
                    .to_dict(
                        orient="records"
                    )
                )


                try:

                    response = requests.post(
                        BATCH_URL,
                        json=records,
                        timeout=120
                    )

                except requests.RequestException as error:

                    st.error(
                        f"Batch request failed: "
                        f"{error}"
                    )

                else:

                    if response.status_code != 200:

                        st.error(
                            response.text
                        )

                    else:

                        result = response.json()


                        st.divider()

                        c1, c2, c3 = st.columns(3)

                        c1.metric(
                            "Total Transactions",
                            f'{result["total_transactions"]:,}'
                        )

                        c2.metric(
                            "Fraud Detected",
                            f'{result["fraud_detected"]:,}'
                        )

                        c3.metric(
                            "Legitimate",
                            f'{result["legitimate_transactions"]:,}'
                        )


                        c4, c5, c6, c7 = st.columns(4)

                        c4.metric(
                            "Critical",
                            result["critical_risk"]
                        )

                        c5.metric(
                            "High",
                            result["high_risk"]
                        )

                        c6.metric(
                            "Medium",
                            result["medium_risk"]
                        )

                        c7.metric(
                            "Low",
                            result["low_risk"]
                        )


                        results_df = pd.DataFrame(
                            result["results"]
                        )


                        display_df = (
                            batch_df
                            .reset_index(drop=True)
                            .copy()
                        )

                        display_df[
                            "prediction"
                        ] = results_df[
                            "prediction"
                        ]

                        display_df[
                            "fraud_probability"
                        ] = results_df[
                            "fraud_probability"
                        ]

                        display_df[
                            "risk_level"
                        ] = results_df[
                            "risk_level"
                        ]


                        display_df[
                            "fraud_probability"
                        ] = (
                            display_df[
                                "fraud_probability"
                            ]
                            * 100
                        )


                        display_df = display_df.sort_values(
                            "fraud_probability",
                            ascending=False
                        )


                        st.subheader(
                            "Analysis Results"
                        )

                        st.dataframe(
                            display_df,
                            use_container_width=True
                        )


                        fraud_only = display_df[
                            display_df[
                                "prediction"
                            ] == "FRAUD"
                        ]


                        if not fraud_only.empty:

                            st.subheader(
                                "Detected Fraud"
                            )

                            st.dataframe(
                                fraud_only,
                                use_container_width=True
                            )


                        csv_buffer = io.StringIO()

                        display_df.to_csv(
                            csv_buffer,
                            index=False
                        )


                        st.download_button(
                            label=(
                                "Download Analysis Results"
                            ),
                            data=csv_buffer.getvalue(),
                            file_name=(
                                "fraud_analysis_results.csv"
                            ),
                            mime="text/csv"
                        )


