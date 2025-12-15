import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Tuple
from scipy import optimize
import sqlite3

# Set page config and theme
st.set_page_config(page_title="Bond Valuation Dashboard", layout="wide", page_icon="💹")

# Bloomberg-style CSS
st.markdown("""
<style>
    /* Global settings */
    .main {
        background-color: #121212;
        color: #E0E0E0;
        font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #FF9F0A !important; /* Bloomberg Orange */
        font-weight: 600;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem !important;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        padding: 10px 15px;
        border-left: 3px solid #FF9F0A;
        border-radius: 2px;
    }
    label[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: #AAAAAA !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetricValue"] {
        font-family: "Roboto Mono", monospace;
        font-size: 1.25rem !important;
        color: #FFFFFF !important;
        font-weight: 500;
    }
    
    /* Buttons */
    div.stButton > button {
        border-radius: 2px;
        font-weight: 600;
        border: 1px solid #444;
        background-color: #2D2D2D;
        color: #E0E0E0;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        border-color: #FF9F0A;
        color: #FF9F0A;
    }
    div.stButton > button[kind="primary"] {
        background-color: #FF9F0A;
        color: #000;
        border: none;
    }
    
    /* Tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #333;
    }
    
    /* Spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 98%;
    }
    hr {
        margin: 1rem 0 !important;
        border-color: #333;
    }
</style>
""", unsafe_allow_html=True)

# --- Database Management ---
DB_NAME = "bond_portfolio.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Bonds Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS bonds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            start_date TEXT NOT NULL,
            principal REAL NOT NULL
        )
    ''')
    
    # Cash Flows Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS cash_flows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bond_id INTEGER NOT NULL,
            flow_date TEXT NOT NULL,
            amount REAL NOT NULL,
            FOREIGN KEY (bond_id) REFERENCES bonds (id)
        )
    ''')
    
    # Rates Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            rate REAL NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def save_bond(name: str, start_date: date, principal: float) -> int:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO bonds (name, start_date, principal) VALUES (?, ?, ?)',
              (name, start_date.strftime('%Y-%m-%d'), principal))
    bond_id = c.lastrowid
    conn.commit()
    conn.close()
    return bond_id

def update_bond(bond_id: int, name: str, start_date: date, principal: float):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('UPDATE bonds SET name = ?, start_date = ?, principal = ? WHERE id = ?',
              (name, start_date.strftime('%Y-%m-%d'), principal, bond_id))
    conn.commit()
    conn.close()

def delete_bond(bond_id: int):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM cash_flows WHERE bond_id = ?', (bond_id,))
    c.execute('DELETE FROM bonds WHERE id = ?', (bond_id,))
    conn.commit()
    conn.close()

def get_all_bonds() -> pd.DataFrame:
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM bonds", conn)
    conn.close()
    if not df.empty:
        df['start_date'] = pd.to_datetime(df['start_date']).dt.date
    return df

def get_bond_details(bond_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM bonds WHERE id = ?', (bond_id,))
    row = c.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d['start_date'] = datetime.strptime(d['start_date'], '%Y-%m-%d').date()
        return d
    return None

def save_cash_flows(bond_id: int, cash_flows: List[Tuple[date, float]]):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM cash_flows WHERE bond_id = ?', (bond_id,))
    data = [(bond_id, cf[0].strftime('%Y-%m-%d'), cf[1]) for cf in cash_flows]
    c.executemany('INSERT INTO cash_flows (bond_id, flow_date, amount) VALUES (?, ?, ?)', data)
    conn.commit()
    conn.close()

def get_bond_cash_flows(bond_id: int) -> Dict[str, float]:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT flow_date, amount FROM cash_flows WHERE bond_id = ? ORDER BY flow_date', (bond_id,))
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

# --- Rate Management ---
def get_all_rates() -> List[Dict]:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM rates ORDER BY start_date')
    rows = c.fetchall()
    conn.close()
    rates = []
    for row in rows:
        r = dict(row)
        r['start'] = datetime.strptime(r['start_date'], '%Y-%m-%d').date()
        r['end'] = datetime.strptime(r['end_date'], '%Y-%m-%d').date()
        rates.append(r)
    return rates

def save_rate(start_date: date, end_date: date, rate: float):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('INSERT INTO rates (start_date, end_date, rate) VALUES (?, ?, ?)',
              (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), rate))
    conn.commit()
    conn.close()

def delete_rate(rate_id: int):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM rates WHERE id = ?', (rate_id,))
    conn.commit()
    conn.close()

def seed_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # 1. Seed Rates
    c.execute("SELECT count(*) FROM rates")
    if c.fetchone()[0] == 0:
        rates = [
            {'start': '2022-01-01', 'end': '2022-03-31', 'rate': 0.0725},
            {'start': '2022-04-01', 'end': '2022-06-30', 'rate': 0.0875},
            {'start': '2022-07-01', 'end': '2022-09-30', 'rate': 0.0925},
            {'start': '2022-10-01', 'end': '2022-12-31', 'rate': 0.10},
            {'start': '2023-01-01', 'end': '2023-03-31', 'rate': 0.09},
            {'start': '2023-04-01', 'end': '2023-06-30', 'rate': 0.0875},
            {'start': '2023-07-01', 'end': '2023-09-30', 'rate': 0.0975},
            {'start': '2023-10-01', 'end': '2023-12-31', 'rate': 0.10},
            {'start': '2024-01-01', 'end': '2024-03-31', 'rate': 0.09},
            {'start': '2024-04-01', 'end': '2024-06-30', 'rate': 0.10},
            {'start': '2024-07-01', 'end': '2024-09-30', 'rate': 0.095},
            {'start': '2024-10-01', 'end': '2024-12-31', 'rate': 0.085},
            {'start': '2025-01-01', 'end': '2025-03-31', 'rate': 0.09},
            {'start': '2025-04-01', 'end': '2025-06-30', 'rate': 0.0925},
            {'start': '2025-07-01', 'end': '2025-09-30', 'rate': 0.0875},
            {'start': '2025-10-01', 'end': '2025-12-31', 'rate': 0.0875},
        ]
        c.executemany("INSERT INTO rates (start_date, end_date, rate) VALUES (?, ?, ?)", 
                      [(r['start'], r['end'], r['rate']) for r in rates])
        conn.commit()

    # 2. Seed Bond ISC
    c.execute("SELECT id FROM bonds WHERE name = ?", ("Bond ISC",))
    bond = c.fetchone()
    if not bond:
        start_date = '2022-07-22'
        principal = 500.0
        c.execute("INSERT INTO bonds (name, start_date, principal) VALUES (?, ?, ?)", ("Bond ISC", start_date, principal))
        bond_id = c.lastrowid
        
        flows = {
            '2022-07-22': 0, '2022-07-27': 200, '2022-08-04': 5050, '2022-08-12': 5200, '2022-08-17': 2350,
            '2022-08-29': 1000, '2022-09-09': 1700, '2022-09-21': 265, '2022-10-03': 400, '2022-10-17': 180,
            '2022-10-24': 200, '2022-11-02': 300, '2022-11-17': 200, '2022-12-05': 1600, '2023-01-17': 325,
            '2023-02-06': 750, '2023-02-16': 475, '2023-03-22': 850, '2023-03-28': 3900, '2023-04-03': 1625,
            '2023-04-11': 1375, '2023-04-20': 1175, '2023-04-26': 390, '2023-06-02': 775, '2023-07-04': 1675,
            '2023-08-15': 1475, '2023-09-04': 1100, '2023-11-30': 1800, '2023-12-15': 975, '2024-01-16': 500,
            '2024-01-29': 895, '2024-02-07': 900, '2024-02-27': 1195, '2024-03-13': 1000, '2024-03-30': 1095,
            '2024-04-17': 900, '2024-08-21': 850, '2024-09-03': 800, '2024-09-12': 1000, '2024-10-01': 1600,
            '2024-10-29': 2000, '2024-11-11': 1800, '2024-12-03': 2500, '2025-01-19': 2500, '2025-03-26': 800,
            '2025-04-29': 900, '2025-05-16': 1000, '2025-05-23': 700, '2025-06-03': 1300, '2025-07-02': 1000
        }
        data = [(bond_id, d, amt) for d, amt in flows.items() if amt > 0]
        c.executemany("INSERT INTO cash_flows (bond_id, flow_date, amount) VALUES (?, ?, ?)", data)
        conn.commit()
    
    conn.close()

def xirr(cash_flows: List[Tuple[date, float]]) -> Optional[float]:
    """
    Calculate the Extended Internal Rate of Return.
    
    Args:
        cash_flows: List of (date, amount) tuples. 
                   Investments should be negative, returns positive.
    """
    if not cash_flows:
        return None
        
    dates = [cf[0] for cf in cash_flows]
    amounts = [cf[1] for cf in cash_flows]
    
    if not amounts or sum(amounts) == 0:
        return None

    try:
        def xnpv(rate, dates, amounts):
            if rate <= -1.0:
                return float('inf')
            min_date = min(dates)
            return sum([a / ((1 + rate) ** ((d - min_date).days / 365.0)) for d, a in zip(dates, amounts)])

        return optimize.newton(lambda r: xnpv(r, dates, amounts), 0.1)
    except (RuntimeError, ValueError):
        return None

class StepRateBond:
    def __init__(self, principal: float, start_date: date, additional_cash_flows: Optional[Dict[str, float]] = None):
        """
        Initialize the bond with principal amount and start date.

        Args:
            principal: Initial investment amount
            start_date: Bond start date
            additional_cash_flows: Dictionary of {'YYYY-MM-DD': amount} for additional investments
        """
        self.principal = principal
        # Handle start_date being either string or date object
        if isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        else:
            self.start_date = start_date
            
        self.maturity_date = self.start_date + relativedelta(years=3)  # 3-year term
        self.additional_cash_flows = additional_cash_flows or {}
        self.rates = self._get_quarterly_rates()
        self.schedule = []
        
        # Unitization
        self.units = 0.0
        self.nav = 1.00000

    def _get_quarterly_rates(self) -> List[Dict]:
        """
        Define the quarterly rates based on the provided information.
        """
        return get_all_rates()

    def get_rate_for_date(self, date: date) -> float:
        """
        Get the applicable rate for a given date.
        - If date is within a defined period, return that rate.
        - If date is in the future (after all defined rates), return the last defined rate.
        - If date is in a gap, return the rate of the period immediately preceding.
        - If date is before all defined rates, return the first defined rate.
        """
        if not self.rates:
            return 0.0
            
        # 1. Exact match check
        for rate in self.rates:
            if rate['start'] <= date <= rate['end']:
                return rate['rate']
        
        # 2. Fallback: Find the last rate that started on or before the target date
        # Since self.rates is sorted by start_date:
        valid_rates = [r for r in self.rates if r['start'] <= date]
        
        if valid_rates:
            return valid_rates[-1]['rate']
            
        # 3. If date is before any rate starts, return the first available rate
        return self.rates[0]['rate']

    def get_quarter_start_end(self, dt: date) -> Tuple[date, date]:
        """Get the start and end dates of the quarter for a given date."""
        quarter = (dt.month - 1) // 3 + 1
        quarter_start = date(dt.year, 3 * quarter - 2, 1)
        next_quarter = quarter_start + relativedelta(months=3)
        quarter_end = next_quarter - timedelta(days=1)
        return quarter_start, quarter_end

    def get_book_close_date(self, quarter_end: date) -> date:
        """Get the book close date for a quarter (one month before quarter end)."""
        return quarter_end - relativedelta(months=1)

    def generate_schedule(self) -> Tuple[pd.DataFrame, List[str]]:
        """Generate a detailed schedule of bond values with daily interest accrual."""
        current_date = self.start_date
        current_principal = self.principal
        total_deposits = self.principal
        cumulative_interest = 0.0
        schedule = []
        detailed_log = []
        last_quarter = None
        
        # Initialize NAV and Units
        # Initial Principal creates units at NAV 1.00000
        self.nav = 1.00000
        self.units = current_principal / self.nav
        detailed_log.append(f"{current_date.strftime('%Y-%m-%d')}: Initial Sub: R{current_principal:,.2f} / NAV {self.nav:.5f} = {self.units:,.4f} Units")

        while current_date <= self.maturity_date:
            date_str = current_date.strftime('%Y-%m-%d')

            # Get current quarter info
            quarter_start, quarter_end = self.get_quarter_start_end(current_date)
            current_quarter = (quarter_start, quarter_end)

            # Check if we've moved to a new quarter (Capitalization Event)
            if last_quarter is not None and current_quarter != last_quarter:
                # Capitalize the interest from the previous quarter
                current_principal += cumulative_interest
                detailed_log.append(f"{last_quarter[1].strftime('%Y-%m-%d')}: Capitalized interest: R{cumulative_interest:,.2f}")
                cumulative_interest = 0.0

            last_quarter = current_quarter

            # Get the rate for the current date
            annual_rate = self.get_rate_for_date(current_date)

            # Calculate daily rate based on actual days in year (365 or 366)
            days_in_year = 366 if (current_date.month >= 3 and
                                 (current_date.year % 4 == 0 and
                                 (current_date.year % 100 != 0 or current_date.year % 400 == 0))) or \
                            (current_date.month < 3 and
                             ((current_date.year - 1) % 4 == 0 and
                             ((current_date.year - 1) % 100 != 0 or (current_date.year - 1) % 400 == 0))) else 365

            daily_rate = annual_rate / days_in_year

            # 1. Calculate Interest for the Day (on Opening Balance)
            # Interest accrues on the principal backing the units
            daily_interest = current_principal * daily_rate
            cumulative_interest += daily_interest
            
            # 2. Update Total Value and NAV
            total_value = current_principal + cumulative_interest
            if self.units > 0:
                self.nav = round(total_value / self.units, 5)
            
            # 3. Process Additional Deposits (At Prevailing NAV)
            additional_deposit = self.additional_cash_flows.get(date_str, 0)
            if additional_deposit > 0:
                # Create units at the newly calculated NAV
                new_units = additional_deposit / self.nav
                self.units += new_units
                
                # Add to principal for future interest accrual
                current_principal += additional_deposit
                total_deposits += additional_deposit
                
                detailed_log.append(f"{date_str}: Deposit R{additional_deposit:,.2f} buys {new_units:,.4f} units @ NAV {self.nav:.5f}")
                
                # Re-verify Total Value after deposit (for record keeping)
                total_value = current_principal + cumulative_interest

            # Add to schedule
            schedule.append({
                'Date': current_date,
                'Principal': current_principal,
                'Total_Deposits': total_deposits,
                'Rate': annual_rate,
                'Daily_Interest': daily_interest,
                'Cumulative_Interest': cumulative_interest,
                'Additional_Deposit': additional_deposit,
                'Is_Quarter_End': current_date == quarter_end,
                'Total_Value': total_value,
                'NAV': self.nav,
                'Units': self.units,
                'Calculation_Notes': f'Daily interest on R{current_principal:,.2f} @ {annual_rate:.2%} p.a.'
            })

            # If it's the last day of the bond term, capitalize any remaining interest
            if current_date == self.maturity_date and cumulative_interest > 0:
                current_principal += cumulative_interest
                detailed_log.append(f"{date_str}: Final interest capitalization: R{cumulative_interest:,.2f}")
                cumulative_interest = 0.0

            # Move to next day
            current_date += timedelta(days=1)

        return self._finalize_schedule(schedule), detailed_log

    def _finalize_schedule(self, schedule: List[Dict]) -> pd.DataFrame:
        """Convert the schedule to a DataFrame and add any final calculations."""
        if not schedule:
            return pd.DataFrame()

        df = pd.DataFrame(schedule)
        # Convert Date to datetime64[ns]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Explicitly ensure index is DatetimeIndex for resampling
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Ensure all numeric columns have proper types
        numeric_cols = ['Principal', 'Total_Deposits', 'Rate', 'Daily_Interest',
                       'Cumulative_Interest', 'Additional_Deposit', 'Total_Value', 'NAV', 'Units']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Forward fill any missing values for better visualization
        df.ffill(inplace=True)

        return df

def render_manage_rates():
    st.title("📉 Manage Interest Rates")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back"):
            st.session_state.view = 'portfolio'
            st.rerun()
            
    st.info("Rates define the daily interest accrual for all bonds. Ensure dates cover all active bond periods.")
    
    rates = get_all_rates()
    df_rates = pd.DataFrame(rates)
    
    # Prepare for editor
    if not df_rates.empty:
        # Sort and select cols
        df_rates = df_rates.sort_values('start')[['start', 'end', 'rate']]
    else:
        df_rates = pd.DataFrame(columns=['start', 'end', 'rate'])

    with st.form("rates_form"):
        edited_df = st.data_editor(
            df_rates,
            column_config={
                "start": st.column_config.DateColumn("Start Date", format="YYYY-MM-DD", required=True),
                "end": st.column_config.DateColumn("End Date", format="YYYY-MM-DD", required=True),
                "rate": st.column_config.NumberColumn("Annual Rate (0.10 = 10%)", format="%.4f", required=True, min_value=0.0, max_value=1.0),
            },
            num_rows="dynamic",
            use_container_width=True,
            key="rates_editor"
        )
        
        if st.form_submit_button("Save Rate Changes"):
            # Transactional update: Delete all and re-insert
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            try:
                c.execute('DELETE FROM rates')
                
                new_data = []
                for _, row in edited_df.iterrows():
                    if pd.notnull(row['start']) and pd.notnull(row['end']) and pd.notnull(row['rate']):
                        s = row['start']
                        e = row['end']
                        # Handle varied types from editor
                        if isinstance(s, datetime): s = s.date()
                        elif isinstance(s, str): s = datetime.strptime(s, '%Y-%m-%d').date()
                        if isinstance(e, datetime): e = e.date()
                        elif isinstance(e, str): e = datetime.strptime(e, '%Y-%m-%d').date()
                        
                        new_data.append((s.strftime('%Y-%m-%d'), e.strftime('%Y-%m-%d'), float(row['rate'])))
                
                c.executemany('INSERT INTO rates (start_date, end_date, rate) VALUES (?, ?, ?)', new_data)
                conn.commit()
                st.success("Rates updated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error saving rates: {e}")
            finally:
                conn.close()

def render_portfolio_dashboard():
    st.title("🏦 Portfolio Dashboard")
    
    # Control Bar
    with st.container():
        c1, c2, c3, c4 = st.columns([2, 3, 1.5, 1.5])
        with c1:
            analysis_date = st.date_input("📅 Analysis Date", value=date.today())
        with c3:
            if st.button("📉 Manage Rates", use_container_width=True):
                st.session_state.view = 'manage_rates'
                st.rerun()
        with c4:
            if st.button("➕ Add Bond", use_container_width=True, type="primary"):
                st.session_state.view = 'add_bond'
                st.rerun()

    # Check for rates configuration
    if not get_all_rates():
        st.warning("⚠️ No interest rates found in database. Portfolio calculations require rate data.")
        col_seed, _ = st.columns([1, 2])
        with col_seed:
            if st.button("Initialize Default Rates & Sample Bond", type="primary"):
                seed_db()
                st.success("Database seeded with default rates and Bond ISC.")
                st.rerun()

    bonds = get_all_bonds()
    
    if bonds.empty:
        st.info("No bonds in portfolio. Click 'Add New Bond' to get started.")
        return

    # Calculate summary metrics for all bonds
    portfolio_summary = []
    all_schedules = []
    # Collect all raw cash flows for XIRR calculation
    raw_xirr_flows = [] 
    
    for _, bond_row in bonds.iterrows():
        # Load cash flows
        flows = get_bond_cash_flows(bond_row['id'])
        
        # Run calculation
        bond_calc = StepRateBond(bond_row['principal'], bond_row['start_date'], flows)
        schedule, _ = bond_calc.generate_schedule()
        
        if not schedule.empty:
            latest = schedule.iloc[-1]
            val = latest['Total_Value']
            
            # Determine "As At" value for this specific bond for the list view
            as_at_ts = pd.Timestamp(analysis_date)
            if as_at_ts in schedule.index:
                as_at_row = schedule.loc[as_at_ts]
            elif as_at_ts > schedule.index[-1]:
                as_at_row = schedule.iloc[-1]
            elif as_at_ts < schedule.index[0]:
                as_at_row = schedule.iloc[0]
                as_at_row[:] = 0 # Pre-start
            else:
                idx = schedule.index.get_indexer([as_at_ts], method='pad')[0]
                as_at_row = schedule.iloc[idx]

            portfolio_summary.append({
                'ID': bond_row['id'],
                'Name': bond_row['name'],
                'Start Date': bond_row['start_date'],
                'Principal': bond_row['principal'],
                'Maturity Value': val,
                'As At Value': as_at_row['Total_Value'],
                'As At NAV': as_at_row['NAV'],
                'Units': latest['Units']
            })
            
            # Collect for Aggregation
            if not isinstance(schedule.index, pd.DatetimeIndex):
                schedule.set_index('Date', inplace=True)
            all_schedules.append(schedule[['Total_Value', 'Principal', 'Total_Deposits', 'Cumulative_Interest', 'Daily_Interest', 'Units']])
            
            # Collect Cash Flows
            # 1. Initial Investment (Negative)
            raw_xirr_flows.append((bond_row['start_date'], -bond_row['principal']))
            # 2. Additional Deposits (Negative)
            for dt_str, amt in flows.items():
                d = datetime.strptime(dt_str, '%Y-%m-%d').date()
                raw_xirr_flows.append((d, -amt))
            
            # 3. Maturity Payout (Positive) - If bond matures within the schedule
            # We assume if the schedule reaches maturity_date, it pays out.
            bond_maturity = bond_calc.maturity_date
            if schedule.index[-1].date() == bond_maturity:
                # Bond has matured, this is a cash outflow from the portfolio back to investor (conceptually)
                # or essentially a realized gain event for XIRR calculation.
                maturity_value = schedule.iloc[-1]['Total_Value']
                raw_xirr_flows.append((bond_maturity, maturity_value))

    if not portfolio_summary:
        st.warning("No active bond schedules found.")
        return

    # --- Portfolio Aggregation ---
    portfolio_schedule = pd.concat(all_schedules).groupby(level=0).sum().sort_index()
    
    # Calculate Blended NAV
    portfolio_schedule['NAV'] = portfolio_schedule.apply(
        lambda x: x['Total_Value'] / x['Units'] if x['Units'] > 0 else 1.0, axis=1
    )
    
    # --- Metrics Calculation ---
    
    # 1. Projected (End of Life)
    proj_row = portfolio_schedule.iloc[-1]
    proj_val = proj_row['Total_Value']
    proj_inv = proj_row['Total_Deposits']
    proj_gain = proj_val - proj_inv
    proj_roi = (proj_val / proj_inv - 1) * 100 if proj_inv > 0 else 0.0
    
    # Projected XIRR
    # raw_xirr_flows now includes maturity payouts for all bonds.
    # Since proj_val at the very end represents the value of bonds maturing on that last day,
    # and we've already added those as positive flows, we do NOT add proj_val again.
    proj_xirr = xirr(raw_xirr_flows)
    
    # 2. As At (Analysis Date)
    as_at_ts = pd.Timestamp(analysis_date)
    if as_at_ts > portfolio_schedule.index[-1]:
        as_at_row = portfolio_schedule.iloc[-1]
        # If analysis date is past portfolio maturity, value is 0 (all cashed out)
        # But our schedule drops bonds. 
        # Actually, if past last date, value should likely be considered 0 if we assume payout.
        # But for visualization we might show last known.
        # For XIRR, if past end, we rely on payouts.
    elif as_at_ts < portfolio_schedule.index[0]:
        as_at_row = pd.Series(0, index=portfolio_schedule.columns)
        as_at_row['NAV'] = 1.0
    else:
        idx = portfolio_schedule.index.get_indexer([as_at_ts], method='pad')[0]
        as_at_row = portfolio_schedule.iloc[idx]
        
    as_at_val = as_at_row['Total_Value']
    as_at_inv = as_at_row['Total_Deposits']
    as_at_gain = as_at_val - as_at_inv
    as_at_roi = (as_at_val / as_at_inv - 1) * 100 if as_at_inv > 0 else 0.0
    as_at_units = as_at_row['Units']
    
    # As At XIRR
    # Filter flows:
    # - Investments (negative) occurred on or before analysis date
    # - Payouts (positive) occurred STRICTLY BEFORE analysis date.
    #   (Payouts ON analysis date are represented by as_at_val, or if they matured, 
    #    we assume we hold the value until end of day for "As At")
    as_at_xirr_flows = []
    for d, amt in raw_xirr_flows:
        if amt < 0 and d <= analysis_date:
            as_at_xirr_flows.append((d, amt))
        elif amt > 0 and d < analysis_date:
            as_at_xirr_flows.append((d, amt))
            
    # Add current value as positive flow
    as_at_xirr_flows.append((analysis_date, as_at_val))
    as_at_xirr = xirr(as_at_xirr_flows)

    # --- Metrics Display ---
    
    # Row 1: As At
    st.markdown(f"##### 📅 Performance As At: <span style='color:#FF9F0A'>{analysis_date.strftime('%Y-%m-%d')}</span>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Value", f"R{as_at_val:,.2f}")
    c2.metric("Invested", f"R{as_at_inv:,.2f}")
    c3.metric("Gain", f"R{as_at_gain:,.2f}")
    c4.metric("Units", f"{as_at_units:,.4f}")
    c5.metric("ROI", f"{as_at_roi:.2f}%")
    c6.metric("XIRR", f"{as_at_xirr:.2%}" if as_at_xirr else "N/A")
    
    st.markdown("---")
    
    # Row 2: Projected
    st.markdown("##### 🚀 Projected (Maturity)")
    proj_units = proj_row['Units']
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Value", f"R{proj_val:,.2f}")
    c2.metric("Invested", f"R{proj_inv:,.2f}")
    c3.metric("Gain", f"R{proj_gain:,.2f}")
    c4.metric("Units", f"{proj_units:,.4f}")
    c5.metric("ROI", f"{proj_roi:.2f}%")
    c6.metric("XIRR", f"{proj_xirr:.2%}" if proj_xirr else "N/A")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Aggregate Chart ---
    st.subheader("Portfolio Composition & Growth")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}],
               [{"secondary_y": False}]]
    )
    
    dates = portfolio_schedule.index.to_numpy()
    
    # --- Row 1: Total Value & Metrics ---
    
    # 1. Total Market Value (Area)
    fig.add_trace(go.Scatter(
        x=dates, y=portfolio_schedule['Total_Value'],
        name="Total Market Value",
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(0, 104, 201, 0.4)',
        hovertemplate="Value: R%{y:,.2f}<extra></extra>"
    ), row=1, col=1, secondary_y=False)

    # 2. Principal Balance (Line)
    fig.add_trace(go.Scatter(
        x=dates, y=portfolio_schedule['Principal'],
        name="Principal Balance",
        mode='lines',
        line=dict(color='#4A90E2', width=1.5, dash='dash'),
        hovertemplate="Principal: R%{y:,.2f}<extra></extra>"
    ), row=1, col=1, secondary_y=False)

    # 3. Net Invested (Line)
    fig.add_trace(go.Scatter(
        x=dates, y=portfolio_schedule['Total_Deposits'],
        name="Net Capital Invested",
        mode='lines',
        line=dict(color='#FFD700', width=2),
        hovertemplate="Invested: R%{y:,.2f}<extra></extra>"
    ), row=1, col=1, secondary_y=False)

    # 3. Total Units (Line)
    fig.add_trace(go.Scatter(
        x=dates, y=portfolio_schedule['Units'],
        name="Total Units",
        mode='lines',
        line=dict(color='#E0E0E0', width=1.5, dash='dot'),
        hovertemplate="Units: %{y:,.4f}<extra></extra>"
    ), row=1, col=1, secondary_y=False)

    # 4. NAV (Secondary Axis)
    fig.add_trace(go.Scatter(
        x=dates, y=portfolio_schedule['NAV'],
        name="Portfolio NAV",
        mode='lines',
        line=dict(color='#00CC96', width=2, dash='solid'),
        hovertemplate="NAV: %{y:.5f}<extra></extra>"
    ), row=1, col=1, secondary_y=True)
    
    # --- Row 2: Accrued Interest ---
    
    # 5. Accrued Interest (Area)
    fig.add_trace(go.Scatter(
        x=dates, y=portfolio_schedule['Cumulative_Interest'],
        name="Accrued Interest",
        fill='tozeroy',
        mode='lines',
        line=dict(color='#FF9F0A', width=1.5),
        fillcolor='rgba(255, 159, 10, 0.2)',
        hovertemplate="Accrued: R%{y:,.2f}<extra></extra>"
    ), row=2, col=1)
    
    # Analysis Date Line (on both plots)
    for i in [1, 2]:
        fig.add_vline(x=pd.Timestamp(analysis_date).timestamp() * 1000, line_width=1, line_dash="dash", line_color="#FF9F0A", row=i, col=1)
    
    fig.add_annotation(x=analysis_date, y=1, yref="paper", text="Analysis Date", showarrow=False, font=dict(color="#FF9F0A"))

    fig.update_layout(
        height=600,
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Row 1 Axes
    fig.update_yaxes(title_text="Value (R)", row=1, col=1, secondary_y=False, gridcolor='#333')
    fig.update_yaxes(title_text="NAV", row=1, col=1, secondary_y=True, showgrid=False, tickformat='.4f')
    
    # Row 2 Axes
    fig.update_yaxes(title_text="Accrued (R)", row=2, col=1, gridcolor='#333')
    fig.update_xaxes(gridcolor='#333')
    
    st.plotly_chart(fig, use_container_width=True)

    # --- Bonds List ---
    st.subheader("Your Bonds")
    
    # Header
    with st.container():
        h1, h2, h3, h4, h5, h6 = st.columns([3, 2, 2, 2, 1.5, 1.5])
        h1.markdown("**Bond Name**")
        h2.markdown("**Start Date**")
        h3.markdown("**Principal**")
        h4.markdown(f"**Value ({analysis_date})**")
        h5.markdown("**NAV**")
        h6.markdown("")
    st.markdown("<hr style='margin: 5px 0; border-color: #333;'>", unsafe_allow_html=True)

    for item in portfolio_summary:
        with st.container():
            c1, c2, c3, c4, c5, c6 = st.columns([3, 2, 2, 2, 1.5, 1.5])
            c1.write(f"**{item['Name']}**")
            c2.write(f"{item['Start Date']}")
            c3.write(f"R{item['Principal']:,.2f}")
            c4.write(f"R{item['As At Value']:,.2f}")
            c5.write(f"{item['As At NAV']:.5f}")
            if c6.button("View", key=f"btn_{item['ID']}", use_container_width=True):
                st.session_state.selected_bond_id = item['ID']
                st.session_state.view = 'bond_detail'
                st.rerun()
            st.markdown("<hr style='margin: 5px 0; border-color: #333;'>", unsafe_allow_html=True)

def render_add_bond():
    st.title("➕ Add New Bond")
    
    if st.button("← Back"):
        st.session_state.view = 'portfolio'
        st.rerun()

    with st.form("new_bond_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Bond Name", "Client Bond A")
            principal = st.number_input("Initial Principal (R)", min_value=100.0, value=100000.0)
        with col2:
            start_date = st.date_input("Start Date", value=date(2022, 7, 22))
        
        st.markdown("### Additional Cash Flows")
        st.info("You can add top-up deposits after creating the bond.")
        
        submitted = st.form_submit_button("Create Bond", type="primary")
        
        if submitted:
            if not name:
                st.error("Please provide a bond name.")
            else:
                bond_id = save_bond(name, start_date, principal)
                st.success(f"Bond '{name}' created!")
                st.session_state.selected_bond_id = bond_id
                save_cash_flows(bond_id, [])
                st.session_state.view = 'bond_detail'
                st.rerun()

def render_bond_detail(bond_id: int):
    # (Previous implementation is fine, but I'll condense it slightly for consistency if needed, 
    # but for this edit I'll assume the previous implementation is in place.
    # Wait, I need to include it if I'm replacing the end of the file.)
    # The previous tool output showed render_bond_detail was NOT part of the block I'm replacing 
    # if I start from render_portfolio_dashboard (Line 412). 
    # BUT render_bond_detail is AFTER render_add_bond. 
    # So I must include render_bond_detail in the replacement or be very careful with start line.
    # Original file order:
    # ...
    # render_portfolio_dashboard()
    # render_add_bond()
    # render_bond_detail()
    # main()
    #
    # So I need to include ALL of them.
    # To save tokens/complexity, I will keep render_bond_detail mostly as is but ensure it's included.
    
    details = get_bond_details(bond_id)
    if not details:
        st.error("Bond not found.")
        st.session_state.view = 'portfolio'
        st.rerun()
        return

    col_nav1, col_nav2 = st.columns([1, 6])
    with col_nav1:
        if st.button("← Back"):
            st.session_state.view = 'portfolio'
            st.rerun()
            
    st.title(f"📊 {details['name']}")
    
    # ... (Rest of detail view - I'll paste the existing logic to ensure it persists) ...
    # To ensure I don't break it, I'll copy the logic from previous reads.
    
    # --- Edit Bond Details ---
    with st.expander("📝 Edit Bond Details"):
        with st.form("edit_bond_form"):
            c1, c2, c3 = st.columns(3)
            new_name = c1.text_input("Bond Name", details['name'])
            new_principal = c2.number_input("Initial Principal", min_value=0.0, value=float(details['principal']))
            new_start_date = c3.date_input("Start Date", value=details['start_date'])
            
            if st.form_submit_button("Update Details"):
                update_bond(bond_id, new_name, new_start_date, new_principal)
                st.success("Updated!")
                st.rerun()

    # --- Manage Cash Flows ---
    with st.expander("💸 Manage Cash Flows"):
        existing_flows = get_bond_cash_flows(bond_id)
        flow_data = [{"Date": datetime.strptime(d, '%Y-%m-%d').date(), "Amount": amt} 
                     for d, amt in existing_flows.items()]
        if not flow_data: flow_data = [{"Date": None, "Amount": None}]
        
        df_flows = pd.DataFrame(flow_data)
        edited_df = st.data_editor(
            df_flows,
            column_config={
                "Date": st.column_config.DateColumn("Deposit Date", format="YYYY-MM-DD"),
                "Amount": st.column_config.NumberColumn("Amount (R)", format="%.2f", min_value=0.0)
            },
            num_rows="dynamic",
            use_container_width=True,
            key="cf_editor"
        )
        if st.button("Save Flows"):
            new_flows = []
            for _, row in edited_df.iterrows():
                if pd.notnull(row['Date']) and pd.notnull(row['Amount']):
                    d = row['Date']
                    if isinstance(d, datetime): d = d.date()
                    elif isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
                    new_flows.append((d, float(row['Amount'])))
            save_cash_flows(bond_id, new_flows)
            st.success("Saved!")
            st.rerun()

    # --- Danger Zone ---
    with st.expander("🚨 Danger Zone"):
        st.warning("Permanently delete this bond?")
        if st.button("Delete Bond", type="primary"):
            delete_bond(bond_id)
            st.session_state.selected_bond_id = None
            st.session_state.view = 'portfolio'
            st.rerun()

    # --- Calculation & Charts ---
    additional_cash_flows = get_bond_cash_flows(bond_id)
    bond = StepRateBond(details['principal'], details['start_date'], additional_cash_flows)
    schedule_df, detailed_log = bond.generate_schedule()
    
    if schedule_df.empty:
        st.warning("No schedule.")
        return

    latest = schedule_df.iloc[-1]
    total_interest = latest['Total_Value'] - latest['Total_Deposits']
    total_return = (latest['Total_Value'] / latest['Total_Deposits'] - 1) * 100
    
    # XIRR
    xirr_flows = [(details['start_date'], -details['principal'])]
    for d, a in additional_cash_flows.items():
        xirr_flows.append((datetime.strptime(d, '%Y-%m-%d').date(), -a))
    xirr_flows.append((latest.name.date(), latest['Total_Value']))
    p_xirr = xirr(xirr_flows)
    xirr_disp = f"{p_xirr:.2%}" if p_xirr else "N/A"

    st.markdown("### Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Invested", f"R{latest['Total_Deposits']:,.2f}")
    c2.metric("Interest", f"R{total_interest:,.2f}")
    c3.metric("Value", f"R{latest['Total_Value']:,.2f}")
    c4.metric("XIRR", xirr_disp)
    
    c5, c6, c7 = st.columns(3)
    c5.metric("ROI", f"{total_return:.2f}%")
    c6.metric("Units", f"{latest['Units']:,.4f}")
    c7.metric("NAV", f"{latest['NAV']:.5f}")

    # Charts
    st.subheader("Growth")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    dates = schedule_df.index
    
    fig.add_trace(go.Scatter(x=dates, y=schedule_df['Total_Value'], name="Value", fill='tozeroy', fillcolor='rgba(0,104,201,0.2)'), secondary_y=False)
    fig.add_trace(go.Scatter(x=dates, y=schedule_df['Principal'], name="Principal", line=dict(color='#0068C9')), secondary_y=False)
    fig.add_trace(go.Scatter(x=dates, y=schedule_df['Total_Deposits'], name="Invested", line=dict(dash='solid', color='#FFD700')), secondary_y=False)
    fig.add_trace(go.Scatter(x=dates, y=schedule_df['NAV'], name="NAV", line=dict(color='#00CC96')), secondary_y=True)
    
    fig.update_layout(height=400, hovermode='x unified')
    fig.update_yaxes(title_text="Value", secondary_y=False)
    fig.update_yaxes(title_text="NAV", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    t1, t2 = st.tabs(["Schedule", "Log"])
    t1.dataframe(schedule_df, use_container_width=True)
    t2.text_area("Log", value="\n".join(detailed_log), height=200)

def main():
    init_db()
    seed_db()
    
    if 'view' not in st.session_state:
        st.session_state.view = 'portfolio'
    if 'selected_bond_id' not in st.session_state:
        st.session_state.selected_bond_id = None

    if st.session_state.view == 'portfolio':
        render_portfolio_dashboard()
    elif st.session_state.view == 'manage_rates':
        render_manage_rates()
    elif st.session_state.view == 'add_bond':
        render_add_bond()
    elif st.session_state.view == 'bond_detail':
        render_bond_detail(st.session_state.selected_bond_id)

if __name__ == "__main__":
    main()
