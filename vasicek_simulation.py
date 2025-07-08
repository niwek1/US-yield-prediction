import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Vasicek Simulation", layout="wide")
st.title("Vasicek Short Rate & Bond Simulation")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")
a = st.sidebar.slider("Mean Reversion Speed (a)", 0.01, 0.5, 0.1, 0.01)
b = st.sidebar.slider("Long-term Mean (b)", 0.0, 0.1, 0.03, 0.001)
sigma = st.sidebar.slider("Volatility (sigma)", 0.001, 0.05, 0.01, 0.001)
r0 = st.sidebar.slider("Initial Rate (r0)", 0.0, 0.1, 0.02, 0.001)
T = st.sidebar.slider("Horizon (Years)", 0.5, 5.0, 1.0, 0.1)
steps_per_year = st.sidebar.selectbox("Steps per Year", [252, 360, 12, 1], index=0)
num_paths = st.sidebar.number_input("Monte Carlo Paths", 100, 20000, 10000, 100)
coupon = st.sidebar.slider("Bond Coupon", 0.0, 0.1, 0.01, 0.001)
n_periods = st.sidebar.slider("Bond Periods (semiannual)", 2, 40, 20, 1)

# --- Vasicek Simulation Functions ---
def simulate_vasicek_path(a=0.1, b=0.03, sigma=0.01, r0=0.02, T=1, steps_per_year=252):
    dt = 1 / steps_per_year
    N = int(T * steps_per_year)
    r = np.zeros(N)
    r[0] = r0
    for i in range(1, N):
        dr = a * (b - r[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
        r[i] = r[i-1] + dr
    return r

def bond_price_duration(yield_rate, c=0.01, n=20):
    y = yield_rate / 2  # semiannual compounding
    discount_factors = [(1 + y) ** (-i) for i in range(1, n+1)]
    price = c * sum(discount_factors) + 1 * discount_factors[-1]
    weighted_times = [i * c * df for i, df in zip(range(1, n+1), discount_factors)]
    weighted_times[-1] += n * discount_factors[-1]  # add principal weight
    duration = sum(weighted_times) / price * 0.5  # semiannual periods to years
    return price, duration

def monte_carlo_vasicek_bond(
    num_paths=10000, a=0.1, b=0.03, sigma=0.01, r0=0.02,
    T=1, steps_per_year=252, coupon=0.01, n_periods=20
):
    prices = []
    durations = []
    yields_end = []
    for _ in range(num_paths):
        r_path = simulate_vasicek_path(a, b, sigma, r0, T, steps_per_year)
        ytm = r_path[-1]  # use last rate as YTM
        yields_end.append(ytm)
        p, d = bond_price_duration(ytm, c=coupon, n=n_periods)
        prices.append(p)
        durations.append(d)
    return np.array(prices), np.array(durations), np.array(yields_end)

def plot_histogram(data, title, xlabel, color):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(data, bins=50, alpha=0.7, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.grid(True)
    plt.tight_layout()
    return fig

# --- Run Simulation ---
if st.button("Run Vasicek Monte Carlo Simulation"):
    with st.spinner("Simulating..."):
        prices, durations, yields_end = monte_carlo_vasicek_bond(
            num_paths=num_paths, a=a, b=b, sigma=sigma, r0=r0, T=T,
            steps_per_year=steps_per_year, coupon=coupon, n_periods=n_periods
        )
    st.success("Simulation complete!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Bond Price", f"{np.mean(prices):.4f}")
        st.metric("Std Dev Bond Price", f"{np.std(prices):.4f}")
    with col2:
        st.metric("Mean Duration", f"{np.mean(durations):.4f} years")
    with col3:
        st.metric("Mean 10Y Yield (End)", f"{np.mean(yields_end)*100:.2f}%")
    st.subheader("Simulated Distributions")
    st.pyplot(plot_histogram(prices, "Simulated 10Y Bond Price Distribution", "Price", "blue"))
    st.pyplot(plot_histogram(durations, "Simulated Macaulay Duration Distribution", "Duration (Years)", "green"))
    st.pyplot(plot_histogram(yields_end, "Simulated 10Y Yield Distribution (1-Year Horizon)", "Yield", "orange"))
else:
    st.info("Adjust parameters in the sidebar and click 'Run Vasicek Monte Carlo Simulation' to begin.") 