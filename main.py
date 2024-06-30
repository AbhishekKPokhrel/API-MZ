import pandas as pd
import numpy as np
import math
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Set up CORS to allow file:// protocol
origins = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "file://",
    "https://api-mz.onrender.com/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"CORS origins: {origins}")

# # Mount the static directory
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at the root URL
@app.get("/")
def read_root():
    return FileResponse('index.html')


# Global variable for file path
file_path_inputs = "Inputs.xlsx"

class InputData(BaseModel):
    num_employees: int
    # hourly_pct: float
    hr_min: float
    hr_median: float
    hr_max: float
    # avg_dep: int
    state: str
    # file_path_inputs: str



###############################################################################################
##################### Calculation Model #########################################
##############################################################################################


class EligibilityCalculator:
    def __init__(self, num_employees, hr_min, hr_median, hr_max, state):
        self.num_employees = num_employees
        # self.hourly_pct = hourly_pct
        self.hr_min = hr_min
        self.hr_median = hr_median
        self.hr_max = hr_max
        # self.avg_dep = avg_dep
        self.state = state
        self.df = None

    def generate_salaries(self):
        avg_weekly_hrs = 40
        weeks_in_yrs = 52

        np.random.seed(42)
        num_samples = self.num_employees - 3

        sigma_initial = 0.025 * self.hr_median
        hourly_pay_data = np.random.lognormal(mean=np.log(self.hr_median), sigma=sigma_initial, size=num_samples)
        hourly_pay_data = np.append(hourly_pay_data, [self.hr_min, self.hr_median, self.hr_max])

        while not np.all((hourly_pay_data >= self.hr_min) & (hourly_pay_data <= self.hr_max)):
            outside_indices = np.logical_or(hourly_pay_data < self.hr_min, hourly_pay_data > self.hr_max)
            hourly_pay_data[outside_indices] = np.random.lognormal(mean=np.log(self.hr_median), sigma=sigma_initial / 2, size=np.sum(outside_indices))

        salary_data = np.round(hourly_pay_data, 2)
        data = {'id': range(1, self.num_employees + 1), 'hourly_salary': salary_data}
        self.df = pd.DataFrame(data)
        self.df['annual_salary'] = self.df['hourly_salary'] * avg_weekly_hrs * weeks_in_yrs
        self.df['annual_salary'] = self.df['annual_salary'].apply(lambda x: math.floor(x))
        self.df['State/Province'] = self.state

    def generate_household_size(self):
        distribution_percentages = {1: 0.289, 2: 0.347, 3: 0.151, 4: 0.123, 5: 0.056, 6: 0.034}

        np.random.seed(42)
        cumulative_prob = {}
        cumulative = 0
        for size, percentage in distribution_percentages.items():
            cumulative += percentage
            cumulative_prob[size] = cumulative

        household_size_data = []
        for _ in range(self.num_employees):
            rand_num = np.random.rand()
            for size, prob in cumulative_prob.items():
                if rand_num <= prob:
                    household_size_data.append(size)
                    break

        self.df['household_size'] = household_size_data

    def generate_parental_status(self):
        percentages = {"Single": 0.464, "Dual": 0.3, "Single_P": 0.041, "Dual_P": 0.195}

        np.random.seed(42)
        num_single_household = self.df[self.df['household_size'] == 1].shape[0]
        adjusted_single_percentage = percentages["Single"] * (1 - num_single_household / self.num_employees)
        adjusted_percentages = {
            "Single": adjusted_single_percentage,
            "Dual": percentages["Dual"],
            "Single_P": percentages["Single_P"],
            "Dual_P": percentages["Dual_P"]
        }

        total_adjusted_percentage = sum(adjusted_percentages.values())
        remaining_percentage = 1.0 - total_adjusted_percentage

        for category in adjusted_percentages:
            if category != "Single":
                percentage = percentages[category]
                adjusted_percentages[category] += remaining_percentage * (percentage / (1 - percentages["Single"]))

        for index, row in self.df.iterrows():
            household_size = row['household_size']
            if household_size == 1:
                self.df.at[index, 'parental_status'] = 'Single'
            else:
                rand = np.random.rand()
                cumulative_prob = 0
                for status, percentage in adjusted_percentages.items():
                    cumulative_prob += percentage
                    if rand <= cumulative_prob:
                        self.df.at[index, 'parental_status'] = status
                        break

    def generate_employment_type(self):
        married_both_employed_percentage = 0.497

        np.random.seed(42)
        self.df['Marital Status'] = np.where(self.df['parental_status'].isin(['Dual', 'Dual_P']), 'M', 'S')
        
        m_indices = self.df[self.df['Marital Status'] == 'M'].index.to_numpy()
        np.random.shuffle(m_indices)
        num_be = int(len(m_indices) * married_both_employed_percentage)
        self.df.loc[m_indices[:num_be], 'M- Employment Type'] = 'BE'
        self.df.loc[m_indices[num_be:], 'M- Employment Type'] = 'SE'

        self.df['total_household_income'] = self.df['annual_salary']
        self.df.loc[self.df['M- Employment Type'] == 'BE', 'total_household_income'] *= 2

        def calculate_child_status(parental_status):
            return 1 if parental_status in ['Dual_P', 'Single_P'] else 0

        self.df['child_status'] = self.df['parental_status'].apply(lambda x: calculate_child_status(x))
    
    
    def map_fpl(self):
        FPL = {
            'household_size': [1, 2, 3, 4, 5, 6],
            'FPL': [14580, 19720, 24860, 30000, 35140, 40280]
        }
        df_FPL = pd.DataFrame(FPL)

        FPL_mapping = dict(zip(df_FPL['household_size'], df_FPL['FPL']))
        self.df['FPL'] = self.df['household_size'].map(FPL_mapping)


    def map_smi(self):
        global file_path_inputs
        sheet_name = 'SMI'
        df_SMI = pd.read_excel(file_path_inputs, sheet_name=sheet_name)
        
        # Ensure the household size columns are treated as strings for the mapping
        df_SMI.columns = df_SMI.columns.astype(str)

        # Create a dictionary to map the household size values from df_SMI
        smi_dict = df_SMI.set_index('State/Province').to_dict(orient='index')

        # Debugging: Print the SMI dictionary to check its structure
        # print("SMI dictionary keys:\n", smi_dict.keys())
        # print("Sample entry in SMI dictionary for CA:\n", smi_dict.get('CA'))

        # Define a function to get the SMI value based on State and household_size
        def get_smi(row):
            state = row['State/Province']
            household_size = str(row['household_size'])  # Convert to string to match column names in df_SMI
            if state in smi_dict:
                if household_size in smi_dict[state]:
                    return smi_dict[state][household_size]
                else:
                    print(f"Household size {household_size} not found for state {state} in SMI dictionary.")
            else:
                print(f"State {state} not found in SMI dictionary.")
            return None  # Return None if state or household_size not found in df_SMI

        # Apply the function to create the 'SMI' column in df
        self.df['SMI'] = self.df.apply(get_smi, axis=1)

        # # Check for missing SMI values
        # if self.df['SMI'].isna().any():
        #     print("Some SMI values are missing. Please verify the state names and household sizes.")
    
    def adjust_fpl_smi(self):
        global file_path_inputs
        sheet_name = 'FPL_SMI'
        df_IncAdj = pd.read_excel(file_path_inputs, sheet_name=sheet_name)

        adj_fpl_dict = df_IncAdj.set_index('State/Province')['FPL'].to_dict()
        adj_smi_dict = df_IncAdj.set_index('State/Province')['SMI'].to_dict()

        # # Check for missing values before mapping
        # print("States in self.df['State/Province']: ", self.df['State/Province'].unique())
        # print("Keys in adj_fpl_dict: ", adj_fpl_dict.keys())
        # print("Keys in adj_smi_dict: ", adj_smi_dict.keys())

        self.df['FPL_Fct'] = self.df['State/Province'].map(adj_fpl_dict)
        self.df['SMI_Fct'] = self.df['State/Province'].map(adj_smi_dict)

        # Check for NaN values after mapping
        if self.df['FPL_Fct'].isna().any():
            print("Some FPL_Fct values are still NaN. Please verify the state names and adjustment data.")
        if self.df['SMI_Fct'].isna().any():
            print("Some SMI_Fct values are still NaN. Please verify the state names and adjustment data.")

        self.df['FPL_Adj'] = self.df['FPL'] * self.df['FPL_Fct']
        self.df['SMI_Adj'] = self.df['SMI'] * self.df['SMI_Fct']

    def calculate_eligibility(self):
        self.df['FPL_check'] = (self.df['total_household_income'] <= self.df['FPL_Adj']).astype(int)
        self.df['SMI_check'] = (self.df['total_household_income'] <= self.df['SMI_Adj']).astype(int)
        self.df['FPL_elg'] = ((self.df['child_status'] == 1) & (self.df['FPL_check'] == 1)).astype(int)
        self.df['SMI_elg'] = ((self.df['child_status'] == 1) & (self.df['SMI_check'] == 1)).astype(int)

    def map_benefits(self):
        global file_path_inputs
        sheet_name = 'FPL_SMI'
        df_IncAdj = pd.read_excel(file_path_inputs, sheet_name=sheet_name)

        fpl_ben_dict = df_IncAdj.set_index('State/Province')['FPL_B'].to_dict()
        smi_ben_dict = df_IncAdj.set_index('State/Province')['SMI_B'].to_dict()

        self.df['FPL_ben'] = self.df['State/Province'].map(fpl_ben_dict)
        self.df['SMI_ben'] = self.df['State/Province'].map(smi_ben_dict)

        self.df['fpl_elg_ben'] = np.where(self.df['FPL_elg'] == 1, self.df['FPL_ben'], '')
        self.df['smi_elg_ben'] = np.where(self.df['SMI_elg'] == 1, self.df['SMI_ben'], '')

        self.df['Eligible Benefits'] = np.where(
            (self.df['fpl_elg_ben'] == '') & (self.df['smi_elg_ben'] == ''), '',
            np.where(
                (self.df['fpl_elg_ben'] != '') & (self.df['smi_elg_ben'] == ''), self.df['fpl_elg_ben'],
                np.where(
                    (self.df['fpl_elg_ben'] == '') & (self.df['smi_elg_ben'] != ''), self.df['smi_elg_ben'],
                    self.df['fpl_elg_ben'] + ', ' + self.df['smi_elg_ben']
                )
            )
        )

    def determine_child_care_eligibility(self):
        def determine_eligibility(row):
            return 'Yes' if row['Eligible Benefits'] != '' else 'No'

        self.df['Child Care Eligible?'] = self.df.apply(determine_eligibility, axis=1)
        eligible_count = self.df['Child Care Eligible?'].value_counts().get('Yes', 0)
        percentage_eligible = (eligible_count / self.num_employees) * 100
        return eligible_count, percentage_eligible

    def calculate_percentage_eligible(self):
        global file_path_inputs
        self.generate_salaries()
        self.generate_household_size()
        self.generate_parental_status()
        self.generate_employment_type()
        self.map_fpl()
        self.map_smi()
        self.adjust_fpl_smi()
        self.calculate_eligibility()
        self.map_benefits()
        eligible_count, percentage_eligible = self.determine_child_care_eligibility()
        return eligible_count, percentage_eligible


###############################################################################################
###########################             #######################################################


@app.post("/calculate_eligibility")
def calculate_eligibility(data: InputData):
    try:
        logger.info("Received data: %s", data.model_dump_json())

        calculator = EligibilityCalculator(
            num_employees=data.num_employees,
            # hourly_pct=data.hourly_pct,
            hr_min=data.hr_min,
            hr_median=data.hr_median,
            hr_max=data.hr_max,
            # avg_dep=data.avg_dep,
            state=data.state
        )
        eligible_count, percentage_eligible = calculator.calculate_percentage_eligible()

        # Convert numpy types to native Python types
        result = {
            "Total Eligible for Child Care": int(eligible_count),
            "Percentage of eligible employees": f"{float(percentage_eligible):.2f}%"
        }

        logger.info("Calculated result: %s", result)
        return result

    except Exception as e:
        logger.error("Error calculating eligibility: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    

# uvicorn main:app --reload