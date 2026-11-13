import sqlite3
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# V1: DATABASE SETUP
# ==========================================

DB_PATH = Path("sports_records.db")

st.set_page_config(
    page_title="Smart Campus Sports Analytics",
    page_icon="🏅",
    layout="wide"
)


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def create_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                department TEXT,
                sport TEXT,
                event TEXT,
                result REAL,
                unit TEXT,
                record_date TEXT
            )
        """)

        existing = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(performance)"
            ).fetchall()
        }

        # Add name column without deleting existing records
        if "student_name" not in existing:
            conn.execute(
                "ALTER TABLE performance ADD COLUMN student_name TEXT"
            )

        conn.commit()


def load_data():
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM performance ORDER BY record_date",
            conn
        )

    if df.empty:
        return df

    df["result"] = pd.to_numeric(
        df["result"], errors="coerce"
    )

    df["record_date"] = pd.to_datetime(
        df["record_date"], errors="coerce"
    )

    return df.dropna(subset=["result", "record_date"])


def add_record(student_id, student_name, department,
               sport, event, result, unit, record_date):

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO performance
            (student_id, department, sport, event,
             result, unit, record_date, student_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_id.strip(),
            department.strip(),
            sport.strip(),
            event.strip(),
            float(result),
            unit.strip(),
            record_date.isoformat(),
            student_name.strip()
        ))

        conn.commit()


def delete_record(record_id):
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM performance WHERE id = ?",
            (int(record_id),)
        )
        conn.commit()


create_database()


# ==========================================
# COMMON FUNCTIONS
# ==========================================

def student_display_name(row):
    name = str(row.get("student_name", "") or "").strip()

    if name and name.lower() != "nan":
        return name

    return str(row["student_id"])


def filter_data(df):
    if df.empty:
        return df

    c1, c2, c3 = st.columns(3)

    with c1:
        sports = ["All"] + sorted(
            df["sport"].dropna().astype(str).unique()
        )
        selected_sport = st.selectbox("Select Sport", sports)

    with c2:
        departments = ["All"] + sorted(
            df["department"].dropna().astype(str).unique()
        )
        selected_department = st.selectbox(
            "Select Department", departments
        )

    with c3:
        students = ["All"] + sorted(
            df["student_id"].dropna().astype(str).unique()
        )
        selected_student = st.selectbox(
            "Select Student", students
        )

    result = df.copy()

    if selected_sport != "All":
        result = result[result["sport"] == selected_sport]

    if selected_department != "All":
        result = result[
            result["department"] == selected_department
        ]

    if selected_student != "All":
        result = result[
            result["student_id"].astype(str) == selected_student
        ]

    return result


def get_student_series(df, student, sport, event, unit):

    series = df[
        (df["student_id"].astype(str) == str(student)) &
        (df["sport"].astype(str) == str(sport)) &
        (df["event"].astype(str) == str(event)) &
        (df["unit"].astype(str) == str(unit))
    ].copy()

    return series.dropna(
        subset=["result", "record_date"]
    ).sort_values("record_date")


# ==========================================
# V4: MACHINE LEARNING
# ==========================================

def train_and_predict(series, future_date):

    if len(series) < 3:
        raise ValueError(
            "At least 3 records are required."
        )

    dates = pd.to_datetime(series["record_date"])

    x = (
        dates - dates.min()
    ).dt.days.to_numpy().reshape(-1, 1)

    y = series["result"].astype(float).to_numpy()

    # Chronological training and testing
    split = max(2, int(len(series) * 0.70))

    if split >= len(series):
        split = len(series) - 1

    model = LinearRegression()

    model.fit(x[:split], y[:split])

    test_predictions = model.predict(x[split:])

    mae = mean_absolute_error(
        y[split:], test_predictions
    )

    rmse = float(np.sqrt(
        mean_squared_error(y[split:], test_predictions)
    ))

    r2 = (
        r2_score(y[split:], test_predictions)
        if len(y[split:]) >= 2 else None
    )

    # Train using all records for future prediction
    final_model = LinearRegression()

    final_model.fit(x, y)

    future_days = (
        pd.Timestamp(future_date) - dates.min()
    ).days

    prediction = float(
        final_model.predict(
            np.array([[future_days]])
        )[0]
    )

    return prediction, mae, rmse, r2


def improvement_text(series):

    if len(series) < 2:
        return "More records are needed to track improvement."

    first = float(series.iloc[0]["result"])
    latest = float(series.iloc[-1]["result"])
    unit = str(series.iloc[-1]["unit"])

    difference = latest - first

    return (
        f"Change from first to latest record: "
        f"{difference:+.2f} {unit}"
    )


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🏅 Smart Campus")

st.sidebar.write(
    "Sports Analytics & Performance Prediction"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Add Performance",
        "Student Profiles",
        "Performance Records",
        "Analytics & Trends",
        "ML Prediction",
        "Model Evaluation",
        "Reports & Data",
        "About Project"
    ]
)

df = load_data()


# ==========================================
# V1 + V6: DASHBOARD
# ==========================================

if page == "Dashboard":

    st.title("🏅 Smart Campus Sports Dashboard")

    st.write(
        "Monitor and analyse college sports performance."
    )

    if df.empty:

        st.info(
            "No records available. Add performance records first."
        )

    else:

        total_records = len(df)
        total_students = df["student_id"].nunique()
        total_sports = df["sport"].nunique()
        total_departments = df["department"].nunique()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Records", total_records)
        c2.metric("Total Students", total_students)
        c3.metric("Sports", total_sports)
        c4.metric("Departments", total_departments)

        st.subheader("Recent Performance Records")

        recent = df.sort_values(
            "record_date", ascending=False
        ).head(8).copy()

        recent["Student"] = recent.apply(
            student_display_name, axis=1
        )

        st.dataframe(
            recent[
                ["record_date", "Student", "department",
                 "sport", "event", "result", "unit"]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Records by Sport")

        st.bar_chart(
            df.groupby("sport").size()
        )

        st.subheader("Records Over Time")

        timeline = df.copy()

        timeline["month"] = (
            timeline["record_date"]
            .dt.to_period("M")
            .astype(str)
        )

        st.line_chart(
            timeline.groupby("month").size()
        )


# ==========================================
# V1 + V2: ADD PERFORMANCE
# ==========================================

elif page == "Add Performance":

    st.title("➕ Add Sports Performance")

    with st.form("performance_form"):

        c1, c2 = st.columns(2)

        with c1:

            student_id = st.text_input("Student ID")

            student_name = st.text_input(
                "Student Name (Optional)"
            )

            department = st.text_input("Department")

            sport = st.text_input(
                "Sport", placeholder="Athletics"
            )

        with c2:

            event = st.text_input(
                "Event", placeholder="100 m"
            )

            result = st.number_input(
                "Performance Result",
                min_value=0.0,
                value=0.0,
                step=0.01
            )

            unit = st.text_input(
                "Unit", placeholder="seconds / metres / points"
            )

            record_date = st.date_input(
                "Record Date", value=date.today()
            )

        submitted = st.form_submit_button(
            "Save Performance"
        )

        if submitted:

            required = [
                student_id, department,
                sport, event, unit
            ]

            if not all(value.strip() for value in required):

                st.error("Please fill in all required fields.")

            elif result <= 0:

                st.error("Enter a result greater than zero.")

            else:

                add_record(
                    student_id,
                    student_name,
                    department,
                    sport,
                    event,
                    result,
                    unit,
                    record_date
                )

                st.success("Performance record saved!")

                st.rerun()


# ==========================================
# V2: STUDENT PROFILES
# ==========================================

elif page == "Student Profiles":

    st.title("👤 Student Profiles")

    if df.empty:

        st.info("No student records available.")

    else:

        students = sorted(
            df["student_id"].dropna().astype(str).unique()
        )

        selected = st.selectbox(
            "Choose Student ID", students
        )

        student_df = df[
            df["student_id"].astype(str) == selected
        ].sort_values("record_date")

        latest = student_df.iloc[-1]

        st.subheader(student_display_name(latest))

        c1, c2, c3 = st.columns(3)

        c1.metric("Student ID", selected)

        c2.metric(
            "Department", str(latest["department"])
        )

        c3.metric(
            "Sports Records", len(student_df)
        )

        st.subheader("Sports Participated In")

        st.write(
            ", ".join(
                sorted(student_df["sport"].unique())
            )
        )

        st.subheader("Performance History")

        st.dataframe(
            student_df[
                ["record_date", "sport", "event",
                 "result", "unit"]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Latest Result for Each Event")

        latest_events = (
            student_df.sort_values("record_date")
            .groupby(
                ["sport", "event", "unit"],
                as_index=False
            ).tail(1)
        )

        st.dataframe(
            latest_events[
                ["sport", "event", "result",
                 "unit", "record_date"]
            ],
            use_container_width=True,
            hide_index=True
        )


# ==========================================
# V2 + V3: PERFORMANCE RECORDS
# ==========================================

elif page == "Performance Records":

    st.title("📋 Performance Records")

    if df.empty:

        st.info("No records available.")

    else:

        view = filter_data(df).copy()

        view["Student"] = view.apply(
            student_display_name, axis=1
        )

        st.write(f"Total records displayed: {len(view)}")

        st.dataframe(
            view.sort_values(
                "record_date", ascending=False
            )[
                ["id", "record_date", "Student",
                 "student_id", "department",
                 "sport", "event", "result", "unit"]
            ],
            use_container_width=True,
            hide_index=True
        )

        with st.expander("Delete an incorrect record"):

            st.warning(
                "Deleting a record is permanent."
            )

            record_id = st.number_input(
                "Enter Record ID",
                min_value=1,
                step=1
            )

            if st.button("Delete Record"):

                if int(record_id) in set(
                    df["id"].astype(int)
                ):

                    delete_record(int(record_id))

                    st.success("Record deleted.")

                    st.rerun()

                else:

                    st.error("Record ID not found.")


# ==========================================
# V3 + V5: ANALYTICS AND TRENDS
# ==========================================

elif page == "Analytics & Trends":

    st.title("📈 Analytics & Performance Trends")

    if df.empty:

        st.info("Add records to view analytics.")

    else:

        view = filter_data(df)

        if view.empty:

            st.warning("No matching records.")

        else:

            st.subheader("Records by Sport")

            st.bar_chart(
                view.groupby("sport").size()
            )

            st.subheader("Average Result by Event")

            summary = (
                view.groupby(
                    ["sport", "event", "unit"]
                )["result"]
                .agg(["count", "mean", "min", "max"])
                .reset_index()
            )

            st.dataframe(
                summary.round(2),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Individual Performance Trend")

            student = st.selectbox(
                "Select Student",
                sorted(
                    view["student_id"]
                    .dropna().astype(str).unique()
                ),
                key="trend_student"
            )

            student_view = view[
                view["student_id"].astype(str) == student
            ]

            combinations = student_view[
                ["sport", "event", "unit"]
            ].drop_duplicates()

            options = [
                f"{r.sport} | {r.event} | {r.unit}"
                for r in combinations.itertuples(index=False)
            ]

            selected = st.selectbox(
                "Select Sport, Event and Unit",
                options
            )

            sport, event, unit = selected.split(" | ", 2)

            series = get_student_series(
                df, student, sport, event, unit
            )

            if not series.empty:

                chart = series.set_index(
                    "record_date"
                )[["result"]].sort_index()

                st.line_chart(chart)

                st.info(improvement_text(series))


# ==========================================
# V4: ML PERFORMANCE PREDICTION
# ==========================================

elif page == "ML Prediction":

    st.title("🤖 Future Performance Prediction")

    st.write(
        "Estimate future results using Linear Regression."
    )

    st.info(
        "Predictions are estimates based on historical records, "
        "not guaranteed future results."
    )

    if df.empty:

        st.info("Add performance records first.")

    else:

        student = st.selectbox(
            "Select Student",
            sorted(
                df["student_id"].dropna().astype(str).unique()
            )
        )

        subset = df[
            df["student_id"].astype(str) == student
        ]

        combinations = subset[
            ["sport", "event", "unit"]
        ].drop_duplicates()

        options = [
            f"{r.sport} | {r.event} | {r.unit}"
            for r in combinations.itertuples(index=False)
        ]

        selected = st.selectbox(
            "Select Sport, Event and Unit",
            options
        )

        sport, event, unit = selected.split(" | ", 2)

        series = get_student_series(
            df, student, sport, event, unit
        )

        st.write(
            f"Available records: **{len(series)}**"
        )

        if len(series) < 3:

            st.warning(
                "At least 3 records are needed for prediction."
            )

        else:

            latest_date = pd.to_datetime(
                series["record_date"]
            ).max().date()

            future_date = st.date_input(
                "Prediction Date",
                value=max(date.today(), latest_date),
                min_value=latest_date
            )

            if st.button(
                "Predict Future Performance",
                type="primary"
            ):

                try:

                    prediction, mae, rmse, r2 = (
                        train_and_predict(
                            series, future_date
                        )
                    )

                    st.success("Prediction completed!")

                    st.metric(
                        "Estimated Performance",
                        f"{prediction:.2f} {unit}"
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "MAE", f"{mae:.2f} {unit}"
                    )

                    c2.metric(
                        "RMSE", f"{rmse:.2f} {unit}"
                    )

                    c3.metric(
                        "R²",
                        "Not enough test points"
                        if r2 is None else f"{r2:.3f}"
                    )

                    chart = series.copy()

                    chart["record_date"] = pd.to_datetime(
                        chart["record_date"]
                    )

                    chart = chart.set_index(
                        "record_date"
                    )[["result"]]

                    st.line_chart(chart)

                except Exception as error:

                    st.error(str(error))


# ==========================================
# V5: MODEL EVALUATION
# ==========================================

elif page == "Model Evaluation":

    st.title("🧪 Machine Learning Model Evaluation")

    st.write(
        "Evaluate predictions using historical performance records."
    )

    if df.empty:

        st.info("No records available.")

    else:

        groups = df.groupby(
            ["student_id", "sport", "event", "unit"]
        )

        rows = []

        for (sid, sport, event, unit), group in groups:

            series = (
                group.sort_values("record_date")
                .dropna(subset=["result", "record_date"])
            )

            if len(series) >= 4:

                try:

                    _, mae, rmse, r2 = train_and_predict(
                        series,
                        pd.to_datetime(
                            series["record_date"]
                        ).max().date()
                    )

                    rows.append({
                        "Student ID": sid,
                        "Sport": sport,
                        "Event": event,
                        "Unit": unit,
                        "Records": len(series),
                        "MAE": round(mae, 2),
                        "RMSE": round(rmse, 2),
                        "R2": (
                            round(r2, 3)
                            if r2 is not None else "Unavailable"
                        )
                    })

                except Exception:
                    pass

        if rows:

            evaluation = pd.DataFrame(rows)

            st.dataframe(
                evaluation,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "At least 4 records per student/event/unit "
                "are needed for evaluation."
            )

        st.warning(
            "Small datasets can produce unreliable accuracy scores. "
            "Do not use this model to rank or select students."
        )


# ==========================================
# V6: REPORTS AND CSV MANAGEMENT
# ==========================================

elif page == "Reports & Data":

    st.title("📁 Reports & Data Management")

    st.subheader("Download Performance Records")

    if not df.empty:

        export_df = df.copy()

        export_df["record_date"] = (
            export_df["record_date"]
            .dt.strftime("%Y-%m-%d")
        )

        csv_data = export_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download CSV Report",
            data=csv_data,
            file_name="sports_performance_records.csv",
            mime="text/csv"
        )

    else:

        st.info("No records to export.")

    st.divider()

    st.subheader("Import CSV Records")

    st.write(
        "Required columns: student_id, department, sport, "
        "event, result, unit, record_date. student_name is optional."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        if st.button("Import Records"):

            try:

                new_df = pd.read_csv(uploaded_file)

                required = [
                    "student_id", "department",
                    "sport", "event", "result",
                    "unit", "record_date"
                ]

                missing = [
                    column for column in required
                    if column not in new_df.columns
                ]

                if missing:

                    st.error(
                        "Missing columns: " + ", ".join(missing)
                    )

                else:

                    if "student_name" not in new_df.columns:
                        new_df["student_name"] = ""

                    new_df["result"] = pd.to_numeric(
                        new_df["result"], errors="raise"
                    )

                    new_df["record_date"] = (
                        pd.to_datetime(
                            new_df["record_date"],
                            errors="raise"
                        ).dt.strftime("%Y-%m-%d")
                    )

                    columns = [
                        "student_id", "department",
                        "sport", "event", "result",
                        "unit", "record_date", "student_name"
                    ]

                    new_df = new_df[columns]

                    if new_df.isna().any().any():

                        st.error(
                            "CSV contains empty values."
                        )

                    else:

                        with get_connection() as conn:

                            new_df.to_sql(
                                "performance",
                                conn,
                                if_exists="append",
                                index=False
                            )

                            conn.commit()

                        st.success(
                            f"Successfully imported {len(new_df)} records."
                        )

                        st.rerun()

            except Exception as error:

                st.error(
                    f"Import failed: {error}"
                )

    st.divider()

    st.subheader("Database Information")

    st.write(
        f"Database: `{DB_PATH.resolve()}`"
    )

    st.write(
        f"Total records: **{len(df)}**"
    )

    st.caption(
        "Back up your database before making major changes."
    )


# ==========================================
# V6: ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.title("ℹ️ About This Project")

    st.header(
        "Smart Campus Sports Analytics & Performance Prediction System"
    )

    st.write("""
    This project is designed to collect, analyse and track
    college students' sports performance records.

    The system uses data analytics and machine learning
    to estimate future performance.
    """)

    st.subheader("Technologies Used")

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Streamlit",
        "SQLite",
        "Matplotlib / Streamlit Charts",
        "Git and GitHub"
    ]

    for technology in technologies:
        st.write("•", technology)

    st.subheader("Project Development Versions")

    st.write("""
    **V1:** Basic application and database

    **V2:** Student profiles and performance entry

    **V3:** Data analytics and visualization

    **V4:** Machine learning prediction

    **V5:** Model evaluation and improvement tracking

    **V6:** Complete interactive dashboard and reports
    """)

    st.subheader("Project Goal")

    st.write("""
    To help college sports departments understand student
    performance trends and monitor progress using data.
    """)

    st.subheader("Data Privacy")

    st.write("""
    Collect student records only with proper permission.
    Use anonymised sample data for public demonstrations.
    Never upload private student records to a public GitHub repository.
    """)

    st.success(
        "Smart Campus Sports Analytics – Project V1 to V6"
    )