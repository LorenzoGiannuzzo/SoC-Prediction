import pandas as pd
import openpyxl

def excel_opener(filepath):

    df = pd.read_excel(filepath)

    return df

# IMPORT INPUT FILES

file_path = 'Phase1/1. OCV_SOC_Data/Cha_Dis_OCV_SOC_Data.xlsx'
OCV_data = excel_opener(file_path)

file_path = 'Phase1/2. HPPC_Data/EVE_HPPC_1_25degree_CHG-injectionTemplate.xlsx'
HPPC_Cha = excel_opener(file_path)

file_path = 'Phase1/2. HPPC_Data/EVE_HPPC_1_25degree_DSG-injectionTemplate.xlsx'
HPPC_Dis = excel_opener(file_path)

# IMPORT TEST FILES

file_path = 'Phase1/3. Real_World_Operational_Data/01. Scenario 1/GenerateTestData_S1_DAY0to4.xlsx'
test_1a = excel_opener(file_path)

file_path = 'Phase1/3. Real_World_Operational_Data/01. Scenario 1/GenerateTestData_S1_DAY4to7.xlsx'
test_1b = excel_opener(file_path)

file_path = 'Phase1/3. Real_World_Operational_Data/02. Scenario 2/GenerateTestData_S2_DAY0to4.xlsx'
test_2a = excel_opener(file_path)

file_path = 'Phase1/3. Real_World_Operational_Data/02. Scenario 2/GenerateTestData_S2_DAY4to7.xlsx'
test_2b = excel_opener(file_path)

file_path = 'Phase1/3. Real_World_Operational_Data/03. Scenario 3/GenerateTestData_S3_DAY0to4.xlsx'
test_3a = excel_opener(file_path)

file_path = 'Phase1/3. Real_World_Operational_Data/03. Scenario 3/GenerateTestData_S3_DAY4to7.xlsx'
test_3b = excel_opener(file_path)

file_path = 'Phase1/3. Real_World_Operational_Data/04. Scenario 4/GenerateTestData_S4_DAY0to4.xlsx'
test_4a = excel_opener(file_path)

file_path = 'Phase1/3. Real_World_Operational_Data/04. Scenario 4/GenerateTestData_S4_DAY4to7.xlsx'
test_4b = excel_opener(file_path)
