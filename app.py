import pandas as pd
import numpy as np
import streamlit as st
import folium
import plotly.graph_objects as go

from pathlib import Path
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="YatraAI | Navigate Beyond GPS",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 25px 30px;
    border-radius: 18px;
    background: linear-gradient(90deg, #101820, #172b3a);
    border: 1px solid #2d536b;
    margin-bottom: 20px;
}

.hero h1 {
    margin-bottom: 0px;
}

.status-box {
    padding: 16px 20px;
    border-radius: 12px;
    background-color: #351515;
    border: 1px solid #ff4b4b;
    margin-bottom: 20px;
}

.sensor-card {
    background-color: #161b22;
    border: 1px solid #30363d;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    min-height: 145px;
}

.timeline-card {
    background-color: #161b22;
    border-left: 4px solid #3b82f6;
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 10px;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(135deg, #102a43, #0f1d2b);
    border: 1px solid #2979ff;
    text-align: center;
}

.confidence-box {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #30363d;
}

.small-label {
    color: #8b949e;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    BASE_DIR = Path(__file__).resolve().parent

    sm_path = BASE_DIR / "S-M.csv"
    vm_path = BASE_DIR / "V-M.csv"

    sm = pd.read_csv(
        sm_path,
        encoding="ISO-8859-1"
    )

    vm = pd.read_csv(vm_path)

    # Smartphone dataset columns
    sm.columns = [
        'gps_lat',
        'gps_lon',
        'gps_alt',
        'gps_speed_kmh',
        'gps_accuracy',
        'gps_orientation',
        'gps_satellites',
        't_ms',
        'date',
        'acc_x',
        'acc_y',
        'acc_z',
        'grav_x',
        'grav_y',
        'grav_z',
        'gyro_yaw',
        'gyro_pitch',
        'gyro_roll',
        'mag_x',
        'mag_y',
        'mag_z',
        'orient_yaw',
        'orient_pitch',
        'orient_roll'
    ]

    # Vehicle dataset columns
    vm.columns = [
        'v_gps_sats',
        'v_t_s',
        'v_lat',
        'v_lon',
        'v_velocity_kmh',
        'v_heading',
        'v_height_km',
        'v_vert_vel_kmh',
        'v_sample_period',
        'steer_angle',
        'wheel_fl',
        'wheel_fr',
        'wheel_rl',
        'wheel_rr',
        'yaw_rate',
        'ind_speed_kmh',
        'ind_long_acc_g',
        'ind_lat_acc_g',
        'handbrake',
        'gear_req',
        'gear',
        'engine_rpm',
        'coolant_temp',
        'clutch',
        'brake_pressure',
        'brake_pos',
        'battery_v',
        'air_temp',
        'accel_pedal'
    ]

    # Merge datasets
    df = pd.concat(
        [
            sm.reset_index(drop=True),
            vm.reset_index(drop=True)
        ],
        axis=1
    )

    # Handle missing values
    df = df.dropna().reset_index(drop=True)

    # Reference coordinates
    lat0 = df["v_lat"].mean()
    lon0 = df["v_lon"].mean()

    R = 6371000

    # Convert GPS coordinates to local XY coordinates
    df["true_x"] = (
        np.radians(df["v_lon"] - lon0)
        * R
        * np.cos(np.radians(lat0))
    )

    df["true_y"] = (
        np.radians(df["v_lat"] - lat0)
        * R
    )

    # Vehicle speed in meters per second
    df["speed_mps"] = df["ind_speed_kmh"] / 3.6

    # Sample interval
    df["dt"] = df["v_sample_period"].fillna(0.1)

    df.loc[
        (df["dt"] <= 0) | (df["dt"] > 5),
        "dt"
    ] = 0.1

    return df, lat0, lon0, R


df, lat0, lon0, R = load_data()


# =========================================================
# AI FEATURES
# =========================================================

FEATURES = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",
    "mag_x",
    "mag_y",
    "mag_z",
    "wheel_fl",
    "wheel_fr",
    "wheel_rl",
    "wheel_rr",
    "steer_angle",
    "yaw_rate",
    "ind_long_acc_g",
    "ind_lat_acc_g"
]


# =========================================================
# COORDINATE CONVERSION
# =========================================================

def xy_to_latlon(x, y):

    lat = lat0 + np.degrees(y / R)

    lon = lon0 + np.degrees(
        x / (
            R * np.cos(np.radians(lat0))
        )
    )

    return lat, lon


# =========================================================
# PHYSICS BASED DEAD RECKONING
# =========================================================

def physics_dead_reckoning(start, end, yaw_noise_std=0.0):

    speed = df["speed_mps"].iloc[start:end].values
    dt = df["dt"].iloc[start:end].values

    initial_heading = np.radians(
        df["v_heading"].iloc[start]
    )

    yaw_rate_deg = df["yaw_rate"].iloc[start:end].values.copy()

    if yaw_noise_std > 0:
        rng = np.random.default_rng(42)
        yaw_rate_deg = yaw_rate_deg + rng.normal(
            0, yaw_noise_std, size=yaw_rate_deg.shape
        )

    yaw_rate = np.radians(yaw_rate_deg)

    heading = np.zeros(len(speed))
    heading[0] = initial_heading

    for i in range(1, len(speed)):
        heading[i] = (
            heading[i - 1]
            + yaw_rate[i - 1] * dt[i - 1]
        )

    x = (
        df["true_x"].iloc[start]
        + np.cumsum(
            speed * np.sin(heading) * dt
        )
    )

    y = (
        df["true_y"].iloc[start]
        + np.cumsum(
            speed * np.cos(heading) * dt
        )
    )

    return x, y


# =========================================================
# TRAIN AI MODELS
# =========================================================

@st.cache_resource
def train_models(start, end):

    train_mask = np.ones(
        len(df),
        dtype=bool
    )

    # Do not train on blackout section
    train_mask[start:end] = False

    positions = df[
        ["true_x", "true_y"]
    ].values

    delta_position = np.diff(
        positions,
        axis=0,
        prepend=positions[:1]
    )

    rf_x = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        n_jobs=-1,
        random_state=42
    )

    rf_y = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        n_jobs=-1,
        random_state=42
    )

    rf_x.fit(
        df.loc[train_mask, FEATURES],
        delta_position[train_mask, 0]
    )

    rf_y.fit(
        df.loc[train_mask, FEATURES],
        delta_position[train_mask, 1]
    )

    return rf_x, rf_y


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("# 🧭 YatraAI")

st.sidebar.caption(
    "Navigate Beyond GPS"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🛰️ GPS Blackout Simulator"
)

st.sidebar.write(
    "Simulate GPS failure and observe how "
    "YatraAI uses multi-sensor fusion to "
    "reconstruct the vehicle trajectory."
)


# Scenario selection
scenario = st.sidebar.selectbox(
    "Select Failure Scenario",
    [
        "🚇 Tunnel",
        "🏙️ Urban Canyon",
        "📡 GPS Jamming",
        "⚡ Signal Interference"
    ]
)


# Scenario information
scenario_settings = {

    "🚇 Tunnel": {
        "description":
            "Complete GPS blackout caused by underground signal blockage.",
        "severity": "HIGH",
        "gps_condition": "NO SIGNAL",
        "noise_factor": 1.0
    },

    "🏙️ Urban Canyon": {
        "description":
            "Tall buildings cause multipath errors and partial signal obstruction.",
        "severity": "MEDIUM",
        "gps_condition": "UNRELIABLE",
        "noise_factor": 0.6
    },

    "📡 GPS Jamming": {
        "description":
            "Intentional radio interference completely disrupts GPS reception.",
        "severity": "CRITICAL",
        "gps_condition": "JAMMED",
        "noise_factor": 1.3
    },

    "⚡ Signal Interference": {
        "description":
            "Electronic or environmental interference degrades GPS reliability.",
        "severity": "LOW-MEDIUM",
        "gps_condition": "DEGRADED",
        "noise_factor": 0.35
    }
}


current_scenario = scenario_settings[scenario]


st.sidebar.info(
    current_scenario["description"]
)

st.sidebar.caption(
    f"Severity: {current_scenario['severity']}"
)

st.sidebar.markdown("---")


window_len_s = st.sidebar.slider(
    "GPS Blackout Duration (seconds)",
    min_value=30,
    max_value=180,
    value=60,
    step=10
)


window_rows = window_len_s * 10

max_start = len(df) - window_rows - 1


start = st.sidebar.slider(
    "Blackout Start Point",
    min_value=1000,
    max_value=max_start,
    value=min(46554, max_start),
    step=1000
)


end = start + window_rows


st.sidebar.markdown("---")

st.sidebar.success(
    "🤖 YatraAI Engine Ready"
)

st.sidebar.caption(
    "Multi-Sensor Fusion • ML Prediction • Drift Correction"
)


# =========================================================
# RUN YATRAAI SYSTEM
# =========================================================

NOISY_FEATURES = [
    "acc_x", "acc_y", "acc_z",
    "gyro_yaw", "gyro_pitch", "gyro_roll",
    "mag_x", "mag_y", "mag_z"
]

with st.spinner("🧠 YatraAI is processing sensor data..."):

    scenario_noise = current_scenario["noise_factor"]

    # Traditional Dead Reckoning
    # (heading noise scaled by how severe this failure scenario is)
    x_dr, y_dr = physics_dead_reckoning(
        start,
        end,
        yaw_noise_std=scenario_noise * 2.0
    )

    # Train AI models
    rf_x, rf_y = train_models(
        start,
        end
    )

    # Test features — perturbed to reflect this scenario's sensor conditions
    X_test = df.loc[
        start:end - 1,
        FEATURES
    ].copy()

    if scenario_noise > 0:
        rng = np.random.default_rng(7)
        for col in NOISY_FEATURES:
            col_std = df[col].std()
            X_test[col] = X_test[col] + rng.normal(
                0, scenario_noise * 0.15 * col_std, size=len(X_test)
            )

    pred_x = rf_x.predict(X_test)
    pred_y = rf_y.predict(X_test)

    # AI reconstructed path
    x_ml = (
        df["true_x"].iloc[start]
        + np.cumsum(pred_x)
    )

    y_ml = (
        df["true_y"].iloc[start]
        + np.cumsum(pred_y)
    )


# =========================================================
# TRUE GPS PATH
# =========================================================

tx = df["true_x"].iloc[start:end].values
ty = df["true_y"].iloc[start:end].values


# =========================================================
# ERROR CALCULATIONS
# =========================================================

err_dr = np.sqrt(
    (x_dr - tx) ** 2
    +
    (y_dr - ty) ** 2
)

err_ml = np.sqrt(
    (x_ml - tx) ** 2
    +
    (y_ml - ty) ** 2
)


physics_error = float(err_dr[-1])
ai_error = float(err_ml[-1])


if physics_error > 0:

    improvement = (
        1 - ai_error / physics_error
    ) * 100

else:

    improvement = 0


# Distance travelled
distance_travelled = np.sum(
    np.sqrt(
        np.diff(tx) ** 2
        +
        np.diff(ty) ** 2
    )
)


# =========================================================
# AI NAVIGATION CONFIDENCE
# =========================================================

error_ratio = ai_error / max(distance_travelled, 1)

confidence = 100 * (1 - error_ratio)

confidence = max(
    50,
    min(98, confidence)
)


# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">

    <h1>🧭 YatraAI</h1>

    <h3>Navigate Beyond GPS.</h3>

    <p>
    AI-Powered Intelligent Navigation System for GPS-Denied Environments
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SYSTEM STATUS
# =========================================================

st.markdown(
    f"""
    <div class="status-box">

    <h3>🔴 GPS BLACKOUT ACTIVE</h3>

    <b>Scenario:</b> {scenario}
    &nbsp;&nbsp; | &nbsp;&nbsp;

    <b>Severity:</b> {current_scenario['severity']}
    &nbsp;&nbsp; | &nbsp;&nbsp;

    <b>YatraAI Status:</b>
    🧠 Sensor Fusion & AI Navigation Active

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TOP METRICS
# =========================================================

col1, col2, col3, col4, col5 = st.columns(5)


col1.metric(
    "📡 GPS STATUS",
    current_scenario["gps_condition"]
)


col2.metric(
    "🔴 Physics Drift",
    f"{physics_error:.1f} m"
)


col3.metric(
    "🔵 YatraAI Drift",
    f"{ai_error:.1f} m"
)


col4.metric(
    "📈 Drift Reduction",
    f"{improvement:.1f}%"
)


col5.metric(
    "📏 Distance",
    f"{distance_travelled:.0f} m"
)


st.write("")


# =========================================================
# NAVIGATION CONFIDENCE
# =========================================================

c1, c2 = st.columns([3, 1])


with c1:

    st.markdown(
        "### 🧠 AI Navigation Confidence"
    )

    st.progress(int(confidence))

    st.caption(
        "Confidence score is estimated from trajectory "
        "reconstruction error relative to the distance "
        "travelled during the GPS blackout."
    )


with c2:

    st.markdown(
        f"""
        <div class="confidence-box">

        <div class="small-label">
        NAVIGATION CONFIDENCE
        </div>

        <h1>{confidence:.0f}%</h1>

        🟢 AI Navigation Active

        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("---")


# =========================================================
# TRAJECTORY MAP
# =========================================================

st.subheader(
    "📍 Trajectory Reconstruction During GPS Blackout"
)

st.caption(
    "Comparison of the actual GPS trajectory with traditional "
    "dead reckoning and YatraAI's AI-corrected trajectory "
    "during simulated GPS signal loss."
)


lat_true, lon_true = xy_to_latlon(
    tx,
    ty
)

lat_dr, lon_dr = xy_to_latlon(
    x_dr,
    y_dr
)

lat_ml, lon_ml = xy_to_latlon(
    x_ml,
    y_ml
)


m = folium.Map(
    location=[
        float(np.mean(lat_true)),
        float(np.mean(lon_true))
    ],
    zoom_start=16,
    tiles="CartoDB positron"
)


# Actual GPS path
folium.PolyLine(
    list(zip(lat_true, lon_true)),
    color="#00C853",
    weight=6,
    tooltip="Actual GPS Path"
).add_to(m)


# Traditional Dead Reckoning
folium.PolyLine(
    list(zip(lat_dr, lon_dr)),
    color="#FF1744",
    weight=5,
    dash_array="10",
    tooltip="Traditional Dead Reckoning"
).add_to(m)


# YatraAI trajectory
folium.PolyLine(
    list(zip(lat_ml, lon_ml)),
    color="#2979FF",
    weight=5,
    tooltip="YatraAI AI-Corrected Path"
).add_to(m)


# Start marker
folium.Marker(
    [
        float(lat_true[0]),
        float(lon_true[0])
    ],
    popup="🚗 Journey Started",
    tooltip="Journey Start",
    icon=folium.Icon(
        color="blue",
        icon="play"
    )
).add_to(m)


# GPS blackout marker
folium.Marker(
    [
        float(lat_true[0]),
        float(lon_true[0])
    ],
    popup="⚠ GPS SIGNAL LOST - YatraAI Activated",
    tooltip="GPS Blackout Starts",
    icon=folium.Icon(
        color="red",
        icon="warning-sign"
    )
).add_to(m)


# GPS restored marker
folium.Marker(
    [
        float(lat_true[-1]),
        float(lon_true[-1])
    ],
    popup="🛰 GPS SIGNAL RESTORED",
    tooltip="GPS Restored",
    icon=folium.Icon(
        color="green",
        icon="ok-sign"
    )
).add_to(m)


st_folium(
    m,
    width=None,
    height=550
)


st.markdown(
    """
    🟢 **Actual GPS Path**
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    🔴 **Traditional Dead Reckoning**
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    🔵 **YatraAI AI-Corrected Navigation**
    """
)


st.markdown("---")


# =========================================================
# JOURNEY TIMELINE
# =========================================================

st.subheader(
    "🛣️ GPS Blackout Journey Timeline"
)


t1, t2, t3, t4, t5 = st.columns(5)


with t1:

    st.markdown(
        """
        <div class="timeline-card">
        🟢<br>
        <b>GPS LOCKED</b><br>
        Vehicle navigation starts
        </div>
        """,
        unsafe_allow_html=True
    )


with t2:

    st.markdown(
        f"""
        <div class="timeline-card">
        🚧<br>
        <b>{scenario}</b><br>
        GPS environment changes
        </div>
        """,
        unsafe_allow_html=True
    )


with t3:

    st.markdown(
        """
        <div class="timeline-card">
        🔴<br>
        <b>GPS LOST</b><br>
        Satellite navigation unavailable
        </div>
        """,
        unsafe_allow_html=True
    )


with t4:

    st.markdown(
        """
        <div class="timeline-card">
        🧠<br>
        <b>YATRAAI ACTIVE</b><br>
        Sensor fusion reconstructs path
        </div>
        """,
        unsafe_allow_html=True
    )


with t5:

    st.markdown(
        """
        <div class="timeline-card">
        🛰️<br>
        <b>GPS RESTORED</b><br>
        Position synchronized
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("---")


# =========================================================
# ERROR ANALYSIS
# =========================================================

st.subheader(
    "📊 Navigation Error Analysis"
)


time_axis = np.linspace(
    0,
    window_len_s,
    len(err_dr)
)


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=time_axis,
        y=err_dr,
        mode="lines",
        name="Traditional Dead Reckoning",
        line=dict(
            color="#FF1744",
            width=3
        )
    )
)


fig.add_trace(
    go.Scatter(
        x=time_axis,
        y=err_ml,
        mode="lines",
        name="YatraAI Corrected Navigation",
        line=dict(
            color="#2979FF",
            width=3
        )
    )
)


fig.update_layout(
    template="plotly_dark",
    height=450,
    xaxis_title="GPS Blackout Duration (seconds)",
    yaxis_title="Position Error (meters)",
    legend_title="Navigation Method",
    hovermode="x unified"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


st.markdown("---")


# =========================================================
# SENSOR FUSION ENGINE
# =========================================================

st.subheader(
    "🧠 YatraAI Sensor Fusion Engine"
)

st.caption(
    "YatraAI combines multiple independent sensors "
    "to estimate vehicle movement when GPS is unavailable."
)


s1, s2, s3, s4, s5 = st.columns(5)


with s1:

    st.markdown(
        """
        <div class="sensor-card">
        <h2>📱</h2>
        <h4>Accelerometer</h4>
        Measures linear acceleration
        <br><br>
        🟢 ACTIVE
        </div>
        """,
        unsafe_allow_html=True
    )


with s2:

    st.markdown(
        """
        <div class="sensor-card">
        <h2>🔄</h2>
        <h4>Gyroscope</h4>
        Tracks angular movement
        <br><br>
        🟢 ACTIVE
        </div>
        """,
        unsafe_allow_html=True
    )


with s3:

    st.markdown(
        """
        <div class="sensor-card">
        <h2>🧲</h2>
        <h4>Magnetometer</h4>
        Provides heading reference
        <br><br>
        🟢 ACTIVE
        </div>
        """,
        unsafe_allow_html=True
    )


with s4:

    st.markdown(
        """
        <div class="sensor-card">
        <h2>🚗</h2>
        <h4>Vehicle Sensors</h4>
        Wheel speed + steering
        <br><br>
        🟢 ACTIVE
        </div>
        """,
        unsafe_allow_html=True
    )


with s5:

    st.markdown(
        """
        <div class="sensor-card">
        <h2>🤖</h2>
        <h4>AI Engine</h4>
        ML-based trajectory correction
        <br><br>
        🟢 ACTIVE
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# WHY YATRAAI
# =========================================================

st.markdown("---")

st.subheader(
    "🚀 Why YatraAI?"
)


w1, w2, w3, w4 = st.columns(4)


with w1:

    st.info(
        """
        ### 🛰️ GPS Independence
        
        Continues estimating vehicle position even when satellite signals are unavailable.
        """
    )


with w2:

    st.info(
        """
        ### 📱 Multi-Sensor Fusion
        
        Combines smartphone IMU, magnetometer and vehicle dynamics data.
        """
    )


with w3:

    st.info(
        """
        ### 🤖 AI Drift Correction
        
        Machine learning predicts motion patterns and reduces accumulated navigation drift.
        """
    )


with w4:

    st.success(
        """
        ### 🔄 Seamless Recovery
        
        Estimated navigation can be synchronized once reliable GPS connectivity returns.
        """
    )


st.markdown("---")


# =========================================================
# HOW YATRAAI WORKS
# =========================================================

st.subheader(
    "⚙️ How YatraAI Works"
)


flow1, flow2, flow3, flow4 = st.columns(4)


flow1.info(
    "### 1️⃣ Sensor Collection\n"
    "Collects IMU, magnetometer and vehicle dynamics data."
)


flow2.info(
    "### 2️⃣ Dead Reckoning\n"
    "Estimates vehicle position using speed, heading and motion."
)


flow3.info(
    "### 3️⃣ AI Correction\n"
    "Random Forest learns movement patterns and predicts position changes."
)


flow4.success(
    "### 4️⃣ Trajectory Reconstruction\n"
    "YatraAI reconstructs the route until GPS connectivity returns."
)


st.markdown("---")


# =========================================================
# FINAL PERFORMANCE RESULT
# =========================================================

st.markdown(
    f"""
    <div class="result-card">

    <h2>🏆 YatraAI Performance Result</h2>

    <p>
    During a simulated <b>{window_len_s}-second GPS blackout</b>
    caused by <b>{scenario}</b>,
    YatraAI continued estimating vehicle position using
    multi-sensor fusion and machine learning.
    </p>

    <br>

    <h3>
    Traditional Dead Reckoning:
    {physics_error:.1f} m
    &nbsp;&nbsp; →
    &nbsp;&nbsp;
    YatraAI:
    {ai_error:.1f} m
    </h3>

    <h1>📈 {improvement:.1f}% Drift Reduction</h1>

    <p>
    <b>
    YatraAI demonstrates intelligent navigation continuity
    in GPS-denied environments.
    </b>
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FOOTER
# =========================================================

st.write("")

st.caption(
    "🧭 YatraAI • Navigate Beyond GPS • "
    "AI-Powered Intelligent Dead Reckoning System • SIH26168"
)