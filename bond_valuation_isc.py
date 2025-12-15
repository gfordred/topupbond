from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
import pandas as pd
from dateutil.relativedelta import relativedelta

class StepRateBond:
    def __init__(self, principal: float, start_date: str, additional_cash_flows: Optional[Dict[str, float]] = None):
        """
        Initialize the bond with principal amount and start date.

        Args:
            principal: Initial investment amount
            start_date: Bond start date in 'YYYY-MM-DD' format
            additional_cash_flows: Dictionary of {'YYYY-MM-DD': amount} for additional investments
        """
        self.principal = principal
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        self.maturity_date = self.start_date + relativedelta(years=3)  # 3-year term
        self.additional_cash_flows = additional_cash_flows or {}
        self.rates = self._get_quarterly_rates()
        self.schedule = []

    def _get_quarterly_rates(self) -> List[Dict]:
        """
        Define the quarterly rates based on the provided information.
        For 2025, we have rates for Q1-Q3, Q4 will use the last known rate.
        """
        rates = [
            {'start': '2022-01-01', 'end': '2022-03-31', 'rate': 0.0725},    # Q1 2022
            {'start': '2022-04-01', 'end': '2022-06-30', 'rate': 0.0875},  # Q2 2022
            {'start': '2022-07-01', 'end': '2022-09-30', 'rate': 0.0925},  # Q3 2022
            {'start': '2022-10-01', 'end': '2022-12-31', 'rate': 0.10},  # Q4 2022
            {'start': '2023-01-01', 'end': '2023-03-31', 'rate': 0.09},    # Q1 2023
            {'start': '2023-04-01', 'end': '2023-06-30', 'rate': 0.0875},  # Q2 2023
            {'start': '2023-07-01', 'end': '2023-09-30', 'rate': 0.0975},  # Q3 2023
            {'start': '2023-10-01', 'end': '2023-12-31', 'rate': 0.10},  # Q4 2023
            {'start': '2024-01-01', 'end': '2024-03-31', 'rate': 0.09},    # Q1 2024
            {'start': '2024-04-01', 'end': '2024-06-30', 'rate': 0.10},  # Q2 2024
            {'start': '2024-07-01', 'end': '2024-09-30', 'rate': 0.095},  # Q3 2024
            {'start': '2024-10-01', 'end': '2024-12-31', 'rate': 0.085},  # Q4 2024
            {'start': '2025-01-01', 'end': '2025-03-31', 'rate': 0.09},    # Q1 2025
            {'start': '2025-04-01', 'end': '2025-06-30', 'rate': 0.0925},  # Q2 2025
            {'start': '2025-07-01', 'end': '2025-09-30', 'rate': 0.0875},  # Q3 2025
            {'start': '2025-10-01', 'end': '2025-12-31', 'rate': 0.0875},  # Q4 2025 (using Q3 rate as default)
        ]

        # Convert string dates to date objects
        for rate in rates:
            rate['start'] = datetime.strptime(rate['start'], '%Y-%m-%d').date()
            rate['end'] = datetime.strptime(rate['end'], '%Y-%m-%d').date()

        return rates

    def get_rate_for_date(self, date: datetime.date) -> float:
        """Get the applicable rate for a given date."""
        for rate in self.rates:
            if rate['start'] <= date <= rate['end']:
                return rate['rate']
        # If no rate found, return the last known rate
        return self.rates[-1]['rate']

    def get_quarter_start_end(self, dt: date) -> tuple[date, date]:
        """Get the start and end dates of the quarter for a given date."""
        quarter = (dt.month - 1) // 3 + 1
        quarter_start = date(dt.year, 3 * quarter - 2, 1)
        next_quarter = quarter_start + relativedelta(months=3)
        quarter_end = next_quarter - timedelta(days=1)
        return quarter_start, quarter_end

    def get_book_close_date(self, quarter_end: date) -> date:
        """Get the book close date for a quarter (one month before quarter end)."""
        return quarter_end - relativedelta(months=1)

    def calculate_interest(self, principal: float, rate: float, start_date: date, end_date: date) -> float:
        """Calculate interest for a period using act/365 basis."""
        days = (end_date - start_date).days + 1  # +1 to include end date
        return principal * rate * (days / 365)

    def _finalize_schedule(self, schedule: List[Dict], detailed_log: List[str]) -> pd.DataFrame:
        """Convert the schedule to a DataFrame and add any final calculations."""
        if not schedule:
            return pd.DataFrame()

        df = pd.DataFrame(schedule)
        df.set_index('Date', inplace=True)

        # Ensure all numeric columns have proper types
        numeric_cols = ['Principal', 'Total_Deposits', 'Rate', 'Daily_Interest',
                       'Cumulative_Interest', 'Additional_Deposit', 'Total_Value']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Forward fill any missing values for better visualization
        df.ffill(inplace=True)

        return df

    def generate_schedule(self) -> pd.DataFrame:
        """Generate a detailed schedule of bond values with daily interest accrual.

        Interest is calculated daily but only capitalized at the end of each quarter.
        The daily rate is based on the quarterly rate divided by the number of days in the year.
        """
        print(f"\nStarting schedule generation from {self.start_date} to {self.maturity_date}")
        current_date = self.start_date
        current_principal = self.principal
        total_deposits = self.principal
        cumulative_interest = 0.0
        schedule = []
        detailed_log = []
        day_count = 0
        last_quarter = None

        while current_date <= self.maturity_date:
            date_str = current_date.strftime('%Y-%m-%d')

            # Get current quarter info
            quarter_start, quarter_end = self.get_quarter_start_end(current_date)
            current_quarter = (quarter_start, quarter_end)

            # Check if we've moved to a new quarter
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

            # Process any additional deposits for this date
            additional_deposit = self.additional_cash_flows.get(date_str, 0)
            if additional_deposit > 0:
                detailed_log.append(f"{date_str}: Deposit of R{additional_deposit:,.2f} added to principal")
                current_principal += additional_deposit
                total_deposits += additional_deposit

            # Calculate daily interest on current principal
            daily_interest = current_principal * daily_rate
            cumulative_interest += daily_interest

            # Add to schedule
            schedule.append({
                'Date': current_date,
                'Principal': current_principal,
                'Total_Deposits': total_deposits,
                'Rate': annual_rate,  # Show the annual rate for reference
                'Daily_Interest': daily_interest,
                'Cumulative_Interest': cumulative_interest,
                'Additional_Deposit': additional_deposit,
                'Is_Quarter_End': current_date == quarter_end,
                'Total_Value': current_principal + cumulative_interest,
                'Calculation_Notes': f'Daily interest on R{current_principal:,.2f} @ {annual_rate:.2%} p.a.'
            })

            # If it's the last day of the bond term, capitalize any remaining interest
            if current_date == self.maturity_date and cumulative_interest > 0:
                current_principal += cumulative_interest
                detailed_log.append(f"{date_str}: Final interest capitalization: R{cumulative_interest:,.2f}")
                cumulative_interest = 0.0

            # Move to next day
            current_date += timedelta(days=1)
            day_count += 1

            if day_count % 100 == 0:
                print(f"Processed {day_count} days, current date: {current_date}")

        # Finalize the schedule
        return self._finalize_schedule(schedule, detailed_log)

    def _finalize_schedule(self, schedule: list, detailed_log: list = None) -> pd.DataFrame:
        """Convert the schedule list to a DataFrame with cumulative values and print detailed information."""
        df = pd.DataFrame(schedule)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Total_Value is already calculated correctly in the schedule generation
        # No need to recalculate cumulative interest as it's tracked properly

        # Print detailed log if available
        if detailed_log:
            print("\n=== Detailed Calculation Log ===")
            print("\n".join(detailed_log))
            print("\n=== End of Calculation Log ===\n")

        # Print book close information
        print("\n=== Book Close Details ===")
        quarters = df[~df.index.duplicated(keep='last')].resample('Q').last()
        for idx, row in quarters.iterrows():
            q_start = idx - pd.offsets.QuarterBegin()
            book_close = idx - pd.offsets.MonthBegin(1)
            if book_close in df.index:
                q_principal = df.loc[book_close, 'Principal']
                print(f"\nQuarter: {q_start.strftime('%Y-%m-%d')} to {idx.strftime('%Y-%m-%d')}")
                print(f"Book Close Date: {book_close.strftime('%Y-%m-%d')}")
                print(f"Principal at Book Close: R{q_principal:,.2f}")
                print(f"Quarterly Interest: R{row['Daily_Interest']:,.2f}")

        return df

def main():
    print("Starting bond valuation script...")
    # Example usage
    principal = 500
    start_date = '2022-07-22'  # Example start date
    print(f"Principal: R{principal}, Start Date: {start_date}")

    # Example additional cash flows (date: amount)
    additional_cash_flows = {
        '2022-07-22': 0,
        '2022-07-27': 200,
        '2022-08-04': 5050,
        '2022-08-12': 5200,
        '2022-08-17': 2350,
        '2022-08-29': 1000,
        '2022-09-09': 1700,
        '2022-09-21': 265,
        '2022-10-03': 400,
        '2022-10-17': 180,
        '2022-10-24': 200,
        '2022-11-02': 300,
        '2022-11-17': 200,
        '2022-12-05': 1600,
        '2023-01-17': 325,
        '2023-02-06': 750,
        '2023-02-16': 475,
        '2023-03-22': 850,
        '2023-03-28': 3900,
        '2023-04-03': 1625,
        '2023-04-11': 1375,
        '2023-04-20': 1175,
        '2023-04-26': 390,
        '2023-06-02': 775,
        '2023-07-04': 1675,
        '2023-08-15': 1475,
        '2023-09-04': 1100,
        '2023-11-30': 1800,
        '2023-12-15': 975,
        '2024-01-16': 500,
        '2024-01-29': 895,
        '2024-02-07': 900,
        '2024-02-27': 1195,
        '2024-03-13': 1000,
        '2024-03-30': 1095,
        '2024-04-17': 900,
        '2024-08-21': 850,
        '2024-09-03': 800,
        '2024-09-12': 1000,
        '2024-10-01': 1600,
        '2024-10-29': 2000,
        '2024-11-11': 1800,
        '2024-12-03': 2500,
        '2025-01-19': 2500,
        '2025-03-26': 800,
        '2025-04-29': 900,
        '2025-05-16': 1000,
        '2025-05-23': 700,
        '2025-06-03': 1300,
        '2025-07-02': 1000,
    }

    # Initialize bond with principal, start date, and additional cash flows
    print("Initializing bond...")
    bond = StepRateBond(principal, start_date, additional_cash_flows)
    print(f"Bond initialized. Maturity date: {bond.maturity_date}")

    # Generate schedule
    print("Generating schedule...")
    schedule = bond.generate_schedule()
    print("Schedule generation complete.")

    # Display quarterly summary
    print("\nQuarterly Summary:")
    quarterly_summary = schedule.resample('Q').last()
    columns_to_show = ['Total_Deposits', 'Principal', 'Rate', 'Total_Value']
    print(quarterly_summary[columns_to_show].to_string())

    # Display final value and summary
    final_value = schedule['Total_Value'].iloc[-1]
    total_interest = final_value - schedule['Total_Deposits'].iloc[-1]
    print(f"\nSummary after 3 years:")
    print(f"Total Deposits:        R{schedule['Total_Deposits'].iloc[-1]:,.2f}")
    print(f"Total Interest Earned: R{total_interest:,.2f}")
    print(f"Final Value:           R{final_value:,.2f}")
    print(f"Total Return:          {(final_value/schedule['Total_Deposits'].iloc[-1] - 1)*100:.2f}%")

    # Save full schedule to CSV
    schedule.to_csv('bond_valuation_schedule.csv')
    print("\nFull schedule saved to 'bond_valuation_schedule.csv'")

if __name__ == "__main__":
    main()
